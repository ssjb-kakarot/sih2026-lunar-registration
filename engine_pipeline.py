import cv2
import numpy as np
import logging
import time
from typing import Dict, Any, Callable, Optional, Tuple

from algo_asift import ASIFT_Detector
from utils_processing import (
    apply_multiscale_retinex, refine_subpixel, 
    compute_mutual_information, compute_ssim, calculate_rmse, 
    warp_thin_plate_spline, filter_matches_spatially,
    validate_image_input
)

logger = logging.getLogger(__name__)

MAX_DIM_REFERENCE = 2500  
MAX_DIM_SOURCE = 2500     

class SparkEngine:
    def __init__(self):
        self.asift = ASIFT_Detector()

    def _safe_resize(self, img: np.ndarray, max_dim: int):
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            resized = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            return resized, 1.0 / scale 
        return img.copy(), 1.0

    def _compute_error_statistics(self, H: np.ndarray, src_pts: np.ndarray, ref_pts: np.ndarray) -> Tuple[float, float]:
        if H is None or len(src_pts) == 0:
            return float("inf"), float("inf")
        try:
            count = min(len(src_pts), len(ref_pts))
            if count == 0:
                return float("inf"), float("inf")
            projected = cv2.perspectiveTransform(
                src_pts[:count].reshape(-1, 1, 2),
                H,
            ).reshape(-1, 2)
            errors = np.linalg.norm(projected - ref_pts[:count], axis=1)
            return float(np.median(errors)), float(np.percentile(errors, 95))
        except Exception:
            return float("inf"), float("inf")

    def _compute_spatial_coverage(self, pts: np.ndarray, image_shape: tuple) -> float:
        if pts is None or len(pts) < 3:
            return 0.0
        try:
            x_min, y_min = np.min(pts, axis=0)
            x_max, y_max = np.max(pts, axis=0)
            bbox_area = (x_max - x_min) * (y_max - y_min)
            image_area = float(image_shape[0] * image_shape[1])
            if image_area <= 0:
                return 0.0
            coverage = (bbox_area / image_area) * 100.0
            return float(np.clip(coverage, 0.0, 100.0))
        except Exception:
            return 0.0

    def execute_pipeline(self, src_path: str, ref_path: str, enable_asift: bool = True, enable_tps: bool = True, enable_extreme_zoom: bool = True, progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        start_time = time.time()
        diagnostics = {'tps_fallback': False, 'tps_fallback_msg': ""}
        
        def update_progress(percent: int, message: str):
            if progress_callback:
                progress_callback(percent, message)

        try:
            update_progress(5, "Loading image matrices.")
            src_raw = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
            ref_raw = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
            
            if src_raw is None or ref_raw is None:
                return {'success': False, 'message': "File read error encountered. Ensure files are valid images."}

            update_progress(15, "Applying dimensional scaling and Retinex correction.")
            ref_proc, ref_scale_factor = self._safe_resize(ref_raw, MAX_DIM_REFERENCE)
            ref_illum = apply_multiscale_retinex(ref_proc)

            best_inliers = 0
            best_src_pts = []
            best_ref_pts = []
            best_zoom_level = 1.0
            best_src_scale_factor = 1.0

            if enable_extreme_zoom:
                scales_to_try = [4.0, 2.0, 1.0, 0.5, 0.25, 0.125]
            else:
                scales_to_try = [1.0]

            total_scales = len(scales_to_try)
            for idx, zoom in enumerate(scales_to_try):
                progress_step = 15 + int((idx / total_scales) * 40)
                update_progress(progress_step, f"Evaluating Scale Pyramid Layer: {zoom}x")
                
                target_w = int(src_raw.shape[1] * zoom)
                target_h = int(src_raw.shape[0] * zoom)

                if max(target_w, target_h) > MAX_DIM_SOURCE or min(target_w, target_h) < 100:
                    continue

                src_zoomed = cv2.resize(src_raw, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
                src_illum = apply_multiscale_retinex(src_zoomed)

                if enable_asift:
                    src_pts, ref_pts = self.asift.detect_and_match(src_illum, ref_illum)
                else:
                    sift = cv2.SIFT_create(nfeatures=15000)
                    kp1, des1 = sift.detectAndCompute(src_illum, None)
                    kp2, des2 = sift.detectAndCompute(ref_illum, None)
                    if des1 is not None and des2 is not None:
                        flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
                        try:
                            matches = flann.knnMatch(des1.astype(np.float32), des2.astype(np.float32), k=2)
                        except:
                            bf = cv2.BFMatcher(cv2.NORM_L2)
                            matches = bf.knnMatch(des1, des2, k=2)
                            
                        good = []
                        for match in matches:
                            if len(match) == 2:
                                m, n = match
                                if m.distance < 0.75 * n.distance:
                                    good.append(m)
                                    
                        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 2)
                        ref_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 2)
                    else:
                        src_pts, ref_pts = [], []

                if len(src_pts) >= 4:
                    H, mask = cv2.findHomography(src_pts, ref_pts, cv2.USAC_MAGSAC, 5.0, maxIters=5000)
                    if H is not None:
                        inliers = np.sum(mask)
                        if inliers > best_inliers:
                            best_inliers = inliers
                            best_src_pts = src_pts[mask.ravel() == 1]
                            best_ref_pts = ref_pts[mask.ravel() == 1]
                            best_zoom_level = zoom
                            best_src_scale_factor = 1.0 / zoom

            if best_inliers < 4:
                return {'success': False, 'message': "Insufficient corresponding features identified."}

            update_progress(60, "Executing spatial grid bucketing and dimension restoration.")
            src_pts_restored = best_src_pts * best_src_scale_factor
            ref_pts_restored = best_ref_pts * ref_scale_factor

            src_filtered, ref_filtered = filter_matches_spatially(
                src_pts_restored, ref_pts_restored, src_raw.shape, grid_size=(10, 10), max_pts_per_cell=50
            )

            update_progress(70, "Performing boundary-safe sub-pixel calculation.")
            src_sub = refine_subpixel(src_raw, src_filtered, win_size=5)
            ref_sub = refine_subpixel(ref_raw, ref_filtered, win_size=5)

            valid_mask = ~np.isnan(src_sub[:, 0]) & ~np.isnan(ref_sub[:, 0])
            src_sub = src_sub[valid_mask]
            ref_sub = ref_sub[valid_mask]

            update_progress(80, "Calculating robust homography matrix.")
            dynamic_thresh = max(3.0, 4.0 * max(best_src_scale_factor, ref_scale_factor))
            
            H_final = None
            mask_final = None
            if len(src_sub) >= 4:
                H_final, mask_final = cv2.findHomography(src_sub, ref_sub, cv2.USAC_MAGSAC, dynamic_thresh, maxIters=10000)

            if H_final is None or np.sum(mask_final) < 4:
                fallback_thresh = max(8.0, 8.0 * max(best_src_scale_factor, ref_scale_factor))
                if len(src_pts_restored) >= 4:
                    H_final, mask_final = cv2.findHomography(src_pts_restored, ref_pts_restored, cv2.USAC_MAGSAC, fallback_thresh, maxIters=20000)
                    
                    if H_final is None or np.sum(mask_final) < 4:
                        # Final RANSAC safety net
                        H_final, mask_final = cv2.findHomography(src_pts_restored, ref_pts_restored, cv2.RANSAC, fallback_thresh)

                    if H_final is not None and np.sum(mask_final) >= 4:
                        mask_bool = mask_final.ravel().astype(bool)
                        inlier_src = src_pts_restored[mask_bool]
                        inlier_ref = ref_pts_restored[mask_bool]
                        src_filtered, ref_filtered = src_pts_restored, ref_pts_restored
                    else:
                        return {'success': False, 'message': "Registration Sequence Terminated: Matrix generation failed."}
                else:
                    return {'success': False, 'message': "Registration Sequence Terminated: Insufficient points."}
            else:
                mask_bool = mask_final.ravel().astype(bool)
                inlier_src = src_sub[mask_bool]
                inlier_ref = ref_sub[mask_bool]

            update_progress(90, "Applying geometric deformation matrix.")
            h, w = ref_raw.shape
            
            warped_img = None
            actual_tps_used = False
            
            if enable_tps and len(inlier_src) >= 10:
                try:
                    warped_img = warp_thin_plate_spline(src_raw, inlier_src, inlier_ref, (w, h))
                    actual_tps_used = True
                except Exception as e:
                    logger.warning(f"TPS Failure: {str(e)}. Falling back to Homography.")
                    diagnostics['tps_fallback'] = True
                    diagnostics['tps_fallback_msg'] = "Non-rigid constraint failed (e.g. collinear points). Automatically fell back to Homography."
                    
            if warped_img is None:
                warped_img = cv2.warpPerspective(src_raw, H_final, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)

            update_progress(95, "Computing evaluation metrics.")
            inlier_ratio = len(inlier_src) / max(1, len(src_filtered)) 
            
            if actual_tps_used:
                tps_fwd = cv2.createThinPlateSplineShapeTransformer()
                s_pts = inlier_src.reshape(1, -1, 2).astype(np.float32)
                r_pts = inlier_ref.reshape(1, -1, 2).astype(np.float32)
                tps_fwd.estimateTransformation(s_pts, r_pts, [cv2.DMatch(i, i, 0) for i in range(len(inlier_src))])
                _, proj = tps_fwd.applyTransformation(s_pts)
                
                errors = np.linalg.norm(proj.reshape(-1, 2) - inlier_ref, axis=1)
                rmse = float(np.sqrt(np.mean(errors ** 2)))
                median_err = float(np.median(errors))
                p95_err = float(np.percentile(errors, 95))
            else:
                rmse = calculate_rmse(inlier_src, inlier_ref, H_final)
                median_err, p95_err = self._compute_error_statistics(H_final, inlier_src, inlier_ref)
                
            coverage = self._compute_spatial_coverage(inlier_ref, ref_raw.shape)
            
            if rmse < 2.5 and len(inlier_src) >= 40 and coverage > 15.0:
                quality_status = "Excellent"
            elif rmse < 6.0 and len(inlier_src) >= 15:
                quality_status = "Good"
            elif rmse < 12.0 and len(inlier_src) >= 8:
                quality_status = "Marginal"
            else:
                quality_status = "Review Required"

            metrics = {
                'RMSE': rmse,
                'Inliers': len(inlier_src),
                'Inlier_Ratio': inlier_ratio,
                'Mutual_Info': compute_mutual_information(warped_img, ref_raw),
                'SSIM': compute_ssim(warped_img, ref_raw) * 100,
                'validation_median_error': median_err,
                'validation_p95_error': p95_err,
                'spatial_coverage': coverage,
                'quality_status': quality_status,
                'runtime': time.time() - start_time,
                'subpixel_source_points': len(inlier_src),
                'subpixel_reference_points': len(inlier_ref)
            }

            update_progress(100, "Registration Complete.")
            vis_img = self._draw_matches(
                cv2.resize(src_raw, (0,0), fx=best_zoom_level, fy=best_zoom_level), 
                ref_proc, 
                best_src_pts, 
                best_ref_pts
            )

            return {
                'success': True,
                'metrics': metrics,
                'warped_img': warped_img,
                'ref_raw': ref_raw,
                'match_vis': vis_img,
                'src_shape': src_raw.shape,
                'ref_shape': ref_raw.shape,
                'best_scale_ratio': f"{best_zoom_level} : {round(1.0/ref_scale_factor, 2)}",
                'H': H_final,
                'diagnostics': diagnostics,
                'actual_tps_used': actual_tps_used
            }

        except Exception as e:
            return {'success': False, 'message': f"Execution failure: {str(e)}"}

    def _draw_matches(self, img1, img2, pts1, pts2):
        kp1 = [cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in pts1]
        kp2 = [cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in pts2]
        matches = [cv2.DMatch(i, i, 0) for i in range(len(kp1))]
        return cv2.drawMatches(img1, kp1, img2, kp2, matches, None, matchColor=(0, 255, 255), flags=2)