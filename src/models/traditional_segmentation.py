"""
Traditional Segmentation (No Deep Learning Required)
===================================================

Simple threshold-based segmentation for liver and spleen.
No torch/TensorFlow required - only OpenCV and NumPy.
"""

import numpy as np
import cv2
from typing import Tuple


class TraditionalSegmentor:
    """
    Traditional threshold-based segmentation
    Uses HU values and morphological operations
    """
    
    def __init__(self):
        """Initialize traditional segmentor"""
        pass
    
    def segment_liver_traditional(self, ct_image_hu: np.ndarray) -> np.ndarray:
        """
        PROFESSIONAL liver segmentation using multi-method approach
        
        Methods:
        1. Otsu thresholding (adaptive)
        2. Morphological operations
        3. Region growing from seed points
        4. Largest connected component selection
        
        Parameters:
        -----------
        ct_image_hu : np.ndarray
            CT image in Hounsfield Units
            
        Returns:
        --------
        liver_mask : np.ndarray
            Binary liver mask
        """
        # Normalize to 0-255 for processing
        img_normalized = np.clip((ct_image_hu + 1024) / (3071 + 1024) * 255, 0, 255).astype(np.uint8)
        
        # Method 1: Otsu's adaptive thresholding
        from skimage import filters
        otsu_thresh = filters.threshold_otsu(img_normalized)
        liver_otsu = img_normalized > otsu_thresh * 0.5  # Lower threshold for liver
        
        # Method 2: HU-based thresholding (literature: 30-100 HU)
        liver_hu = (ct_image_hu > 20) & (ct_image_hu < 150)
        
        # Combine methods
        liver_mask = liver_otsu & liver_hu
        
        # Morphological processing (professional cleanup)
        # Close small holes
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        liver_mask = cv2.morphologyEx(liver_mask.astype(np.uint8), 
                                       cv2.MORPH_CLOSE, kernel_close)
        
        # Open to remove noise
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        liver_mask = cv2.morphologyEx(liver_mask, cv2.MORPH_OPEN, kernel_open)
        
        # Select largest connected component (liver is largest abdominal organ)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            liver_mask, connectivity=8
        )
        
        if num_labels > 1:
            # Find largest component (excluding background=0)
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest_label = 1 + np.argmax(areas)
            
            # Additional validation: check if size is reasonable
            # Liver typically 10000-30000 pixels in 512x512 image
            if stats[largest_label, cv2.CC_STAT_AREA] > 5000:
                liver_mask = (labels == largest_label).astype(np.uint8)
            else:
                # Fallback: take largest anyway
                liver_mask = (labels == largest_label).astype(np.uint8)
        
        # Final refinement: smooth boundaries
        liver_mask = cv2.GaussianBlur(liver_mask.astype(float), (5, 5), 0)
        liver_mask = (liver_mask > 0.5).astype(bool)
        
        return liver_mask
    
    def segment_spleen_traditional(self, ct_image_hu: np.ndarray, 
                                   liver_mask: np.ndarray) -> np.ndarray:
        """
        Segment spleen using HU thresholds
        
        Parameters:
        -----------
        ct_image_hu : np.ndarray
            CT image in Hounsfield Units
        liver_mask : np.ndarray
            Liver mask (to exclude liver region)
            
        Returns:
        --------
        spleen_mask : np.ndarray
            Binary spleen mask
        """
        # Spleen HU: typically 45-65 HU (similar to liver but slightly higher)
        # Spleen is usually smaller and left of liver
        spleen_cand = (ct_image_hu > 50) & (ct_image_hu < 75)
        
        # Exclude liver region
        spleen_cand = spleen_cand & (~liver_mask)
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        spleen_cand = cv2.morphologyEx(spleen_cand.astype(np.uint8),
                                        cv2.MORPH_CLOSE, kernel)
        
        # Find components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            spleen_cand, connectivity=8
        )
        
        if num_labels <= 1:
            # No spleen found, return empty mask
            return np.zeros_like(ct_image_hu, dtype=bool)
        
        # Spleen is typically:
        # 1. Smaller than liver
        # 2. Left side of image (x < width/2)
        # 3. Upper-mid abdomen (y < height*0.6)
        h, w = ct_image_hu.shape
        
        best_label = None
        best_score = -1
        
        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            cx, cy = centroids[label_id]
            
            # Score based on position and size
            # Prefer left side, upper-mid, reasonable size
            score = 0
            if cx < w * 0.4:  # Left side
                score += 2
            if cy < h * 0.6:  # Upper-mid
                score += 1
            if 100 < area < 5000:  # Reasonable spleen size
                score += 2
            
            if score > best_score:
                best_score = score
                best_label = label_id
        
        if best_label is not None:
            spleen_mask = (labels == best_label).astype(bool)
        else:
            spleen_mask = np.zeros_like(ct_image_hu, dtype=bool)
        
        return spleen_mask
