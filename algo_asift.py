import cv2
import numpy as np
import concurrent.futures
import os
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)

class ASIFT_Detector:
    def __init__(self):
        self.sift = cv2.SIFT_create(nfeatures=15000, nOctaveLayers=6, contrastThreshold=0.01, edgeThreshold=15)
        self.flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        self.max_features_per_tilt = 4000 
        # Cap workers to prevent memory exhaustion on smaller cloud instances
        self.max_workers = min(16, (os.cpu_count() or 1) + 4)

    def _affine_skew(self, tilt: float, phi: float, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = img.shape[:2]
        
        A = np.float32([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        # Prevent division by zero, though tilt should always be >= 1.0
        safe_tilt = max(1e-5, tilt)
        tilt_mat = np.float32([[1.0, 0], [0, 1.0 / safe_tilt]])
        
        A = np.dot(tilt_mat, A)
        
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        corners = np.dot(corners, A.T)
        x, y, w_new, h_new = cv2.boundingRect(np.int32(corners).reshape(1, -1, 2))
        
        A = np.hstack([A, np.float32([[-x], [-y]])])
        
        img_skewed = cv2.warpAffine(img, A, (w_new, h_new), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return img_skewed, A

    def _extract_from_skew(self, img: np.ndarray, tilt: float, phi: float):
        skewed, Affine_Mat = self._affine_skew(tilt, phi, img)
        kps, descs = self.sift.detectAndCompute(skewed, None)
        
        if kps is None or descs is None or len(kps) == 0:
            return [], None
            
        if len(kps) > self.max_features_per_tilt:
            # Deterministic sort for reproducibility
            sorted_indices = np.argsort([-k.response for k in kps])[:self.max_features_per_tilt]
            kps = [kps[i] for i in sorted_indices]
            descs = descs[sorted_indices]
            
        Affine_Mat_inv = cv2.invertAffineTransform(Affine_Mat)
        
        pts = np.array([k.pt for k in kps], dtype=np.float32).reshape(-1, 1, 2)
        pts_restored = cv2.transform(pts, Affine_Mat_inv).reshape(-1, 2)
        
        for i, k in enumerate(kps):
            k.pt = tuple(pts_restored[i])
            
        return kps, descs.astype(np.float32)

    def detect_and_match(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        tilts = [1.0, 1.414, 2.0, 2.8] 
        phis = [0, 72, 144, 216, 288] 
        
        params = [(t, p) for t in tilts for p in (phis if t > 1.0 else [0])]
        
        def run_all(image):
            all_kps, all_descs = [], []
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(self._extract_from_skew, image, t, p) for t, p in params]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        k, d = future.result()
                        if d is not None and len(d) > 0:
                            all_kps.extend(k)
                            all_descs.append(d)
                    except Exception as e:
                        logger.warning(f"Error extracting skew parameters: {e}")
            if len(all_descs) == 0:
                return [], None
            return all_kps, np.vstack(all_descs).astype(np.float32)

        kp1, des1 = run_all(img1)
        kp2, des2 = run_all(img2)

        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2: 
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

        try:
            matches = self.flann.knnMatch(des1, des2, k=2)
        except cv2.error:
            # FLANN fallback if internal type error occurs
            bf = cv2.BFMatcher(cv2.NORM_L2)
            matches = bf.knnMatch(des1, des2, k=2)
        
        good = []
        for match_tuple in matches:
            if len(match_tuple) == 2:
                m, n = match_tuple
                if m.distance < 0.75 * n.distance:
                    good.append(m)

        if len(good) < 4: 
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 2)
        ref_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 2)
        
        return src_pts, ref_pts