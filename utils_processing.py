import cv2
import numpy as np
import logging
from skimage.metrics import structural_similarity as ssim

logger = logging.getLogger(__name__)

def validate_image_input(img: np.ndarray) -> None:
    if img is None:
        raise ValueError("Image input cannot be None.")
    if not isinstance(img, np.ndarray):
        raise TypeError("Image input must be a numpy ndarray.")
    if img.size == 0:
        raise ValueError("Image input cannot be empty.")

def validate_points_input(pts: np.ndarray) -> None:
    if pts is None:
        raise ValueError("Points input cannot be None.")
    if not isinstance(pts, (np.ndarray, list)):
        raise TypeError("Points input must be array-like.")

def apply_multiscale_retinex(img: np.ndarray, sigmas: list = [15, 80, 250]) -> np.ndarray:
    validate_image_input(img)
    img_float = np.float32(img) + 1.0 
    retinex = np.zeros_like(img_float)
    
    for sigma in sigmas:
        # Prevent zero standard deviation crashing GaussianBlur
        if sigma <= 0: continue
        blur = cv2.GaussianBlur(img_float, (0, 0), sigma)
        # Prevent division by zero mathematically
        blur = np.clip(blur, 1.0, None)
        retinex += np.log10(img_float) - np.log10(blur)
        
    retinex = retinex / max(1, len(sigmas))
    min_val, max_val = np.min(retinex), np.max(retinex)
    if max_val - min_val > 1e-6:
        retinex = (retinex - min_val) / (max_val - min_val) * 255.0
    else:
        retinex = np.zeros_like(retinex)
        
    retinex_uint8 = np.clip(retinex, 0, 255).astype(np.uint8)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(retinex_uint8)

def filter_matches_spatially(src_pts: np.ndarray, ref_pts: np.ndarray, img_shape: tuple, grid_size: tuple = (10, 10), max_pts_per_cell: int = 50) -> tuple:
    validate_points_input(src_pts)
    validate_points_input(ref_pts)
    if len(src_pts) == 0: 
        return src_pts, ref_pts
        
    h, w = img_shape[:2]
    cell_h = max(1.0, h / grid_size[0])
    cell_w = max(1.0, w / grid_size[1])
    
    grid = {}
    for sp, rp in zip(src_pts, ref_pts):
        r = max(0, min(int(sp[1] / cell_h), grid_size[0] - 1))
        c = max(0, min(int(sp[0] / cell_w), grid_size[1] - 1))
        
        if (r, c) not in grid: grid[(r, c)] = []
        grid[(r, c)].append((sp, rp))
        
    filtered_src, filtered_ref = [], []
    for cell_idx, pairs in grid.items():
        for sp, rp in pairs[:max_pts_per_cell]:
            filtered_src.append(sp)
            filtered_ref.append(rp)
            
    filtered_src_np = np.array(filtered_src, dtype=np.float32)
    filtered_ref_np = np.array(filtered_ref, dtype=np.float32)
    
    if len(filtered_src_np) < max(20, int(len(src_pts) * 0.3)):
        return src_pts, ref_pts
        
    return filtered_src_np, filtered_ref_np

def refine_subpixel(image: np.ndarray, keypoints: np.ndarray, win_size: int = 5) -> np.ndarray:
    validate_image_input(image)
    validate_points_input(keypoints)
    if len(keypoints) == 0: return keypoints
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
    h, w = gray_image.shape[:2]
    refined_pts = np.copy(keypoints).astype(np.float32)
    margin = win_size + 1 
    
    safe_mask = (
        (refined_pts[:, 0] >= margin) & 
        (refined_pts[:, 0] < (w - margin)) & 
        (refined_pts[:, 1] >= margin) & 
        (refined_pts[:, 1] < (h - margin))
    )
    
    safe_pts = refined_pts[safe_mask]
    
    if len(safe_pts) > 0:
        safe_pts_reshaped = safe_pts.reshape(-1, 1, 2)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001)
        try:
            refined_safe = cv2.cornerSubPix(
                gray_image, safe_pts_reshaped, (win_size, win_size), (-1, -1), criteria
            ).reshape(-1, 2)
            refined_pts[safe_mask] = refined_safe
        except Exception:
            pass
            
    return refined_pts

def warp_thin_plate_spline(src_img: np.ndarray, src_pts: np.ndarray, ref_pts: np.ndarray, output_shape: tuple) -> np.ndarray:
    validate_image_input(src_img)
    validate_points_input(src_pts)
    validate_points_input(ref_pts)

    _, unique_indices = np.unique(src_pts, axis=0, return_index=True)
    src_unique = src_pts[unique_indices]
    ref_unique = ref_pts[unique_indices]

    max_tps_points = 400
    if len(src_unique) > max_tps_points:
        idx = np.random.choice(len(src_unique), max_tps_points, replace=False)
        src_subset, ref_subset = src_unique[idx], ref_unique[idx]
    else:
        src_subset, ref_subset = src_unique, ref_unique

    if len(src_subset) < 3:
        raise ValueError("Insufficient points for TPS transformation.")

    tps = cv2.createThinPlateSplineShapeTransformer()
    s = src_subset.reshape(1, -1, 2).astype(np.float32)
    r = ref_subset.reshape(1, -1, 2).astype(np.float32)
    matches = [cv2.DMatch(i, i, 0) for i in range(len(s[0]))]
    
    try:
        tps.estimateTransformation(r, s, matches)
    except cv2.error as e:
        raise ValueError(f"TPS estimation failed, likely due to collinearity. {str(e)}")
    
    h, w = output_shape[1], output_shape[0]
    
    FAST_GRID_SIZE = 800.0
    scale_factor = FAST_GRID_SIZE / float(max(h, w))
    
    if scale_factor < 1.0:
        small_w, small_h = int(w * scale_factor), int(h * scale_factor)
    else:
        small_w, small_h, scale_factor = w, h, 1.0

    X, Y = np.meshgrid(np.arange(small_w), np.arange(small_h))
    coords_small = np.float32(np.column_stack([X.ravel(), Y.ravel()]).reshape(1, -1, 2))
    
    coords_full_query = coords_small / scale_factor
    
    try:
        _, inv_coords_full = tps.applyTransformation(coords_full_query)
    except cv2.error as e:
        raise ValueError(f"TPS applyTransformation failed. {str(e)}")
        
    inv_coords_full = inv_coords_full.reshape(small_h, small_w, 2)
    
    map_x_small = inv_coords_full[:, :, 0].astype(np.float32)
    map_y_small = inv_coords_full[:, :, 1].astype(np.float32)
    
    map_x_full = cv2.resize(map_x_small, (w, h), interpolation=cv2.INTER_LINEAR)
    map_y_full = cv2.resize(map_y_small, (w, h), interpolation=cv2.INTER_LINEAR)
    
    warped = cv2.remap(src_img, map_x_full, map_y_full, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
    return warped

def calculate_rmse(src_pts: np.ndarray, ref_pts: np.ndarray, H: np.ndarray) -> float:
    if H is None or len(src_pts) == 0: return 0.0
    pts_reshaped = src_pts.reshape(-1, 1, 2)
    try:
        proj = cv2.perspectiveTransform(pts_reshaped, H).reshape(-1, 2)
        return float(np.sqrt(np.mean(np.sum((proj - ref_pts) ** 2, axis=1))))
    except Exception:
        return float('inf')

def compute_mutual_information(img1: np.ndarray, img2: np.ndarray, bins=50) -> float:
    mask = img1 > 0
    if not np.any(mask): return 0.0
    
    h2d, _, _ = np.histogram2d(img1[mask].ravel(), img2[mask].ravel(), bins=bins)
    total = np.sum(h2d)
    if total == 0: return 0.0
    
    pxy = h2d / float(total)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]
    
    nzs = (pxy > 0) & (px_py > 0)
    
    if not np.any(nzs): return 0.0
    return float(np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs])))

def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    try:
        min_dim = min(img1.shape[0], img1.shape[1])
        win_size = min(7, min_dim)
        if win_size < 3: return 0.0 
        if win_size % 2 == 0: win_size -= 1
        
        score, _ = ssim(img1, img2, full=True, data_range=255, win_size=win_size)
        return float(score)
    except Exception as e:
        logger.warning(f"SSIM computation failed: {e}")
        return 0.0