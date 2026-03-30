"""
NASH (Non-Alcoholic Steatohepatitis) Detection Module
======================================================

This module implements NASH-specific detection algorithms combining:
1. HU (Hounsfield Unit) analysis for steatosis detection
2. Morphological features (hepatomegaly, splenomegaly)
3. Texture heterogeneity analysis
4. Frequency domain signatures from FFT

NASH Pathophysiology in CT:
----------------------------
- Steatosis (fat accumulation): Low HU values (<40 HU for fatty liver)
- Hepatomegaly: Enlarged liver volume
- Portal hypertension: Splenomegaly, collateral vessels
- Heterogeneous texture: Inflammatory changes

Author: Bülent Tuğrul
Project: Karaciğer Fibrozunun BT Görüntülerinden Non-İnvaziv Evrelendirilmesi
"""

import numpy as np
import cv2
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from scipy import ndimage
from skimage import measure, morphology, filters
import warnings
warnings.filterwarnings('ignore')


@dataclass
class NASHFeatures:
    """NASH-specific features extracted from CT imaging"""
    
    # Steatosis features (KEY for NASH)
    mean_hu: float              # Average HU value
    median_hu: float            # Median HU value
    min_hu: float               # Minimum HU
    max_hu: float               # Maximum HU
    hu_std: float               # HU standard deviation
    steatosis_percentage: float # % of pixels with HU < 40 (fat)
    liver_spleen_ratio: float   # Liver/Spleen HU ratio (<1.0 suggests steatosis)
    
    # Morphological features
    liver_volume_index: float   # Normalized liver volume
    liver_area: float           # 2D area in current slice
    spleen_area: float          # Spleen area (portal hypertension indicator)
    liver_spleen_area_ratio: float
    hepatomegaly_score: float   # >1.5 suggests enlargement
    
    # Texture heterogeneity (inflammation)
    texture_variance: float
    texture_entropy: float
    texture_uniformity: float
    coefficient_of_variation: float  # CV = std/mean
    
    # Fat distribution
    focal_fat_count: int        # Number of focal fatty areas
    fat_distribution_pattern: str  # 'diffuse', 'focal', 'geographic'
    heterogeneous_fat_score: float
    
    # Advanced features
    edge_sharpness: float       # Liver edge clarity (reduced in NASH)
    surface_nodularity: float   # Surface irregularity score
    parenchymal_heterogeneity: float
    
    # Combined NASH probability
    nash_probability_score: float  # Overall NASH likelihood (0-1)
    nash_confidence: str        # 'low', 'moderate', 'high'


class NASHDetector:
    """
    NASH Detection System using CT Imaging Features
    
    Implements multi-modal approach:
    - HU-based steatosis quantification
    - Morphological analysis
    - Texture heterogeneity
    - Machine learning-based classification
    """
    
    def __init__(self,
                 steatosis_hu_threshold: float = 40.0,
                 liver_spleen_ratio_threshold: float = 1.0,
                 hepatomegaly_threshold: float = 1.5):
        """
        Initialize NASH Detector
        
        Parameters:
        -----------
        steatosis_hu_threshold : float
            HU threshold for fat detection (typically 40)
        liver_spleen_ratio_threshold : float
            L/S ratio below this suggests steatosis
        hepatomegaly_threshold : float
            Volume index above this suggests hepatomegaly
        """
        self.steatosis_hu_threshold = steatosis_hu_threshold
        self.liver_spleen_ratio_threshold = liver_spleen_ratio_threshold
        self.hepatomegaly_threshold = hepatomegaly_threshold
        
    def extract_hu_statistics(self, 
                             image: np.ndarray,
                             liver_mask: np.ndarray) -> Dict[str, float]:
        """
        Extract HU statistics from liver region
        
        Parameters:
        -----------
        image : np.ndarray
            CT image (in HU values)
        liver_mask : np.ndarray
            Binary mask of liver region
            
        Returns:
        --------
        hu_stats : dict
            Dictionary of HU statistics
        """
        liver_pixels = image[liver_mask > 0]
        
        if len(liver_pixels) == 0:
            return self._empty_hu_stats()
        
        stats = {
            'mean_hu': float(np.mean(liver_pixels)),
            'median_hu': float(np.median(liver_pixels)),
            'min_hu': float(np.min(liver_pixels)),
            'max_hu': float(np.max(liver_pixels)),
            'hu_std': float(np.std(liver_pixels)),
            'hu_range': float(np.ptp(liver_pixels)),
        }
        
        # Steatosis percentage (HU < threshold)
        fat_pixels = liver_pixels < self.steatosis_hu_threshold
        stats['steatosis_percentage'] = float(np.sum(fat_pixels) / len(liver_pixels) * 100)
        
        # Coefficient of variation
        stats['cv'] = stats['hu_std'] / (abs(stats['mean_hu']) + 1e-8)
        
        return stats
    
    def compute_liver_spleen_ratio(self,
                                   image: np.ndarray,
                                   liver_mask: np.ndarray,
                                   spleen_mask: Optional[np.ndarray] = None) -> float:
        """
        Compute Liver-to-Spleen HU ratio
        
        This is a CRITICAL indicator for NASH:
        - Normal: L/S ratio ≈ 1.1-1.3
        - Steatosis: L/S ratio < 1.0 (fat reduces liver HU)
        - Severe steatosis: L/S ratio < 0.8
        
        Parameters:
        -----------
        image : np.ndarray
            CT image
        liver_mask : np.ndarray
            Liver segmentation mask
        spleen_mask : np.ndarray, optional
            Spleen segmentation mask
            
        Returns:
        --------
        ls_ratio : float
            Liver/Spleen HU ratio
        """
        liver_hu = np.mean(image[liver_mask > 0]) if np.any(liver_mask) else 0
        
        if spleen_mask is not None and np.any(spleen_mask):
            spleen_hu = np.mean(image[spleen_mask > 0])
        else:
            # Estimate spleen HU (typically 45-55 HU in normal cases)
            spleen_hu = 50.0
        
        ls_ratio = liver_hu / (spleen_hu + 1e-8)
        return float(ls_ratio)
    
    def analyze_morphology(self,
                          liver_mask: np.ndarray,
                          spleen_mask: Optional[np.ndarray] = None,
                          reference_area: Optional[float] = None) -> Dict[str, float]:
        """
        Analyze morphological features
        
        Parameters:
        -----------
        liver_mask : np.ndarray
            Liver segmentation
        spleen_mask : np.ndarray, optional
            Spleen segmentation
        reference_area : float, optional
            Reference normal liver area for comparison
            
        Returns:
        --------
        morph_features : dict
            Morphological features
        """
        liver_area = float(np.sum(liver_mask > 0))
        
        features = {
            'liver_area': liver_area,
            'liver_perimeter': 0.0,
            'liver_circularity': 0.0,
            'liver_solidity': 0.0,
        }
        
        # Compute perimeter and shape features
        if liver_area > 0:
            contours = measure.find_contours(liver_mask, 0.5)
            if contours:
                largest_contour = max(contours, key=len)
                perimeter = len(largest_contour)
                features['liver_perimeter'] = float(perimeter)
                
                # Circularity: 4π * Area / Perimeter²
                circularity = 4 * np.pi * liver_area / (perimeter**2 + 1e-8)
                features['liver_circularity'] = float(circularity)
                
                # Solidity: Area / Convex Hull Area
                try:
                    convex_hull = morphology.convex_hull_image(liver_mask)
                    convex_area = np.sum(convex_hull)
                    features['liver_solidity'] = float(liver_area / (convex_area + 1e-8))
                except:
                    features['liver_solidity'] = 1.0
        
        # Hepatomegaly score
        if reference_area is not None:
            features['hepatomegaly_score'] = liver_area / reference_area
        else:
            # Use typical normal liver area (~15000-20000 pixels in 512x512 image)
            typical_normal_area = 17500
            features['hepatomegaly_score'] = liver_area / typical_normal_area
        
        # Spleen analysis
        if spleen_mask is not None:
            spleen_area = float(np.sum(spleen_mask > 0))
            features['spleen_area'] = spleen_area
            features['liver_spleen_area_ratio'] = liver_area / (spleen_area + 1e-8)
            
            # Splenomegaly indicator (spleen area > 5000 pixels)
            features['splenomegaly_score'] = spleen_area / 5000.0
        else:
            features['spleen_area'] = 0.0
            features['liver_spleen_area_ratio'] = 0.0
            features['splenomegaly_score'] = 0.0
        
        return features
    
    def analyze_texture_heterogeneity(self,
                                     image: np.ndarray,
                                     liver_mask: np.ndarray) -> Dict[str, float]:
        """
        Analyze texture heterogeneity (inflammation indicator)
        
        NASH causes:
        - Increased texture variance (inflammation, fibrosis)
        - Higher entropy (disorganized tissue structure)
        - Reduced uniformity
        
        Parameters:
        -----------
        image : np.ndarray
            CT image
        liver_mask : np.ndarray
            Liver mask
            
        Returns:
        --------
        texture_features : dict
            Texture heterogeneity features
        """
        liver_pixels = image[liver_mask > 0]
        
        if len(liver_pixels) == 0:
            return {
                'texture_variance': 0.0,
                'texture_entropy': 0.0,
                'texture_uniformity': 0.0,
                'coefficient_of_variation': 0.0
            }
        
        # Variance
        variance = float(np.var(liver_pixels))
        
        # Entropy
        hist, _ = np.histogram(liver_pixels, bins=256, density=True)
        hist = hist[hist > 0]
        entropy = float(-np.sum(hist * np.log2(hist + 1e-10)))
        
        # Uniformity
        uniformity = float(np.sum(hist ** 2))
        
        # Coefficient of variation
        cv = float(np.std(liver_pixels) / (abs(np.mean(liver_pixels)) + 1e-8))
        
        features = {
            'texture_variance': variance,
            'texture_entropy': entropy,
            'texture_uniformity': uniformity,
            'coefficient_of_variation': cv
        }
        
        return features
    
    def detect_focal_fat(self,
                        image: np.ndarray,
                        liver_mask: np.ndarray) -> Tuple[int, str, float]:
        """
        Detect focal fat deposits and distribution pattern
        
        Parameters:
        -----------
        image : np.ndarray
            CT image
        liver_mask : np.ndarray
            Liver mask
            
        Returns:
        --------
        focal_count : int
            Number of focal fatty regions
        pattern : str
            Fat distribution pattern
        heterogeneity_score : float
            Fat heterogeneity score
        """
        # Create fat map (HU < threshold)
        fat_map = (image < self.steatosis_hu_threshold) & (liver_mask > 0)
        
        # Label connected components (focal fat regions)
        labeled_fat = measure.label(fat_map)
        focal_count = labeled_fat.max()
        
        # Determine pattern
        steatosis_percentage = float(np.sum(fat_map) / np.sum(liver_mask > 0) * 100)
        
        if steatosis_percentage < 5:
            pattern = 'minimal'
        elif steatosis_percentage < 30:
            if focal_count > 3:
                pattern = 'focal'
            else:
                pattern = 'geographic'
        else:
            pattern = 'diffuse'
        
        # Heterogeneity score (spatial variance of fat distribution)
        if np.any(fat_map):
            # Compute center of mass for fat regions
            fat_coords = np.argwhere(fat_map)
            fat_centroid = np.mean(fat_coords, axis=0)
            distances = np.linalg.norm(fat_coords - fat_centroid, axis=1)
            heterogeneity_score = float(np.std(distances))
        else:
            heterogeneity_score = 0.0
        
        return int(focal_count), pattern, heterogeneity_score
    
    def analyze_liver_edge(self,
                          image: np.ndarray,
                          liver_mask: np.ndarray) -> Tuple[float, float]:
        """
        Analyze liver edge characteristics
        
        NASH/fibrosis effects:
        - Reduced edge sharpness (edema, inflammation)
        - Increased surface nodularity (cirrhosis)
        
        Parameters:
        -----------
        image : np.ndarray
            CT image
        liver_mask : np.ndarray
            Liver mask
            
        Returns:
        --------
        edge_sharpness : float
            Edge sharpness score
        surface_nodularity : float
            Surface irregularity score
        """
        # Detect edges
        edges = filters.sobel(liver_mask.astype(float))
        
        # Edge sharpness: gradient magnitude at boundaries
        edge_pixels = image[edges > 0.1]
        if len(edge_pixels) > 0:
            edge_sharpness = float(np.std(edge_pixels))
        else:
            edge_sharpness = 0.0
        
        # Surface nodularity: contour irregularity
        contours = measure.find_contours(liver_mask, 0.5)
        if contours:
            largest_contour = max(contours, key=len)
            
            # Compute curvature variation
            if len(largest_contour) > 10:
                contour_smooth = ndimage.gaussian_filter1d(largest_contour, sigma=2, axis=0)
                diff = np.diff(contour_smooth, axis=0)
                curvature = np.abs(np.diff(np.arctan2(diff[:, 1], diff[:, 0])))
                surface_nodularity = float(np.std(curvature))
            else:
                surface_nodularity = 0.0
        else:
            surface_nodularity = 0.0
        
        return edge_sharpness, surface_nodularity
    
    def compute_nash_probability(self, features: Dict[str, float]) -> Tuple[float, str]:
        """
        Compute overall NASH probability score
        
        Uses weighted combination of key features:
        - Steatosis percentage (40%)
        - Liver/Spleen ratio (30%)
        - Texture heterogeneity (20%)
        - Morphology (10%)
        
        Parameters:
        -----------
        features : dict
            All extracted features
            
        Returns:
        --------
        probability : float
            NASH probability (0-1)
        confidence : str
            Confidence level
        """
        score = 0.0
        
        # Steatosis component (40%)
        steatosis_pct = features.get('steatosis_percentage', 0)
        if steatosis_pct > 30:
            steatosis_score = 1.0
        elif steatosis_pct > 10:
            steatosis_score = 0.5 + (steatosis_pct - 10) / 40
        else:
            steatosis_score = steatosis_pct / 20
        score += 0.4 * steatosis_score
        
        # Liver/Spleen ratio component (30%)
        ls_ratio = features.get('liver_spleen_ratio', 1.0)
        if ls_ratio < 0.8:
            ls_score = 1.0
        elif ls_ratio < 1.0:
            ls_score = 1.0 - (ls_ratio - 0.8) / 0.2
        else:
            ls_score = 0.0
        score += 0.3 * ls_score
        
        # Texture heterogeneity (20%)
        cv = features.get('coefficient_of_variation', 0)
        texture_entropy = features.get('texture_entropy', 0)
        heterogeneity_score = min(1.0, (cv * 2 + texture_entropy / 8) / 2)
        score += 0.2 * heterogeneity_score
        
        # Morphology (10%)
        hepatomegaly = features.get('hepatomegaly_score', 1.0)
        morph_score = min(1.0, max(0, (hepatomegaly - 1.0) / 0.5))
        score += 0.1 * morph_score
        
        # Determine confidence
        if score > 0.7:
            confidence = 'high'
        elif score > 0.4:
            confidence = 'moderate'
        else:
            confidence = 'low'
        
        return float(score), confidence
    
    def extract_all_features(self,
                            image: np.ndarray,
                            liver_mask: np.ndarray,
                            spleen_mask: Optional[np.ndarray] = None) -> NASHFeatures:
        """
        Extract ALL NASH-related features
        
        Parameters:
        -----------
        image : np.ndarray
            CT image (HU values)
        liver_mask : np.ndarray
            Liver segmentation
        spleen_mask : np.ndarray, optional
            Spleen segmentation
            
        Returns:
        --------
        features : NASHFeatures
            Complete NASH feature set
        """
        # HU statistics
        hu_stats = self.extract_hu_statistics(image, liver_mask)
        
        # Liver-Spleen ratio
        ls_ratio = self.compute_liver_spleen_ratio(image, liver_mask, spleen_mask)
        
        # Morphology
        morph_features = self.analyze_morphology(liver_mask, spleen_mask)
        
        # Texture heterogeneity
        texture_features = self.analyze_texture_heterogeneity(image, liver_mask)
        
        # Focal fat detection
        focal_count, fat_pattern, fat_heterogeneity = self.detect_focal_fat(image, liver_mask)
        
        # Edge analysis
        edge_sharpness, surface_nodularity = self.analyze_liver_edge(image, liver_mask)
        
        # Combine all features for probability calculation
        all_features = {**hu_stats, **morph_features, **texture_features,
                       'liver_spleen_ratio': ls_ratio}
        
        # Compute NASH probability
        nash_prob, nash_conf = self.compute_nash_probability(all_features)
        
        # Create feature object
        features = NASHFeatures(
            mean_hu=hu_stats['mean_hu'],
            median_hu=hu_stats['median_hu'],
            min_hu=hu_stats['min_hu'],
            max_hu=hu_stats['max_hu'],
            hu_std=hu_stats['hu_std'],
            steatosis_percentage=hu_stats['steatosis_percentage'],
            liver_spleen_ratio=ls_ratio,
            liver_volume_index=morph_features['hepatomegaly_score'],
            liver_area=morph_features['liver_area'],
            spleen_area=morph_features['spleen_area'],
            liver_spleen_area_ratio=morph_features['liver_spleen_area_ratio'],
            hepatomegaly_score=morph_features['hepatomegaly_score'],
            texture_variance=texture_features['texture_variance'],
            texture_entropy=texture_features['texture_entropy'],
            texture_uniformity=texture_features['texture_uniformity'],
            coefficient_of_variation=texture_features['coefficient_of_variation'],
            focal_fat_count=focal_count,
            fat_distribution_pattern=fat_pattern,
            heterogeneous_fat_score=fat_heterogeneity,
            edge_sharpness=edge_sharpness,
            surface_nodularity=surface_nodularity,
            parenchymal_heterogeneity=texture_features['texture_variance'],
            nash_probability_score=nash_prob,
            nash_confidence=nash_conf
        )
        
        return features
    
    def _empty_hu_stats(self) -> Dict[str, float]:
        """Return empty HU statistics"""
        return {
            'mean_hu': 0.0,
            'median_hu': 0.0,
            'min_hu': 0.0,
            'max_hu': 0.0,
            'hu_std': 0.0,
            'hu_range': 0.0,
            'steatosis_percentage': 0.0,
            'cv': 0.0
        }


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("NASH Detection Module - Spatial Domain Analysis")
    print("="*80)
    
    # Create sample data
    test_image = np.random.randn(512, 512) * 20 + 45  # Simulated CT (HU values)
    test_liver_mask = np.zeros((512, 512), dtype=bool)
    test_liver_mask[150:350, 150:350] = True  # Simple liver ROI
    
    # Initialize detector
    detector = NASHDetector()
    
    # Extract features
    print("\nExtracting NASH features...")
    features = detector.extract_all_features(test_image, test_liver_mask)
    
    # Display results
    print("\n" + "="*80)
    print("NASH DETECTION RESULTS:")
    print("="*80)
    print(f"\n{'Feature':<40} {'Value':>20}")
    print("-"*80)
    
    for field, value in features.__dict__.items():
        if isinstance(value, (int, float)):
            print(f"{field:<40} {value:>20.4f}")
        else:
            print(f"{field:<40} {str(value):>20}")
    
    print("\n" + "="*80)
    print("CLINICAL INTERPRETATION:")
    print("="*80)
    print(f"NASH Probability Score: {features.nash_probability_score:.2%}")
    print(f"Confidence Level: {features.nash_confidence.upper()}")
    print(f"\nSteatosis Level: {features.steatosis_percentage:.1f}%")
    if features.steatosis_percentage > 30:
        print("  → SEVERE steatosis detected")
    elif features.steatosis_percentage > 10:
        print("  → MODERATE steatosis detected")
    else:
        print("  → MILD/NO steatosis")
    
    print(f"\nLiver/Spleen Ratio: {features.liver_spleen_ratio:.2f}")
    if features.liver_spleen_ratio < 0.8:
        print("  → ABNORMAL - Suggests significant steatosis")
    elif features.liver_spleen_ratio < 1.0:
        print("  → BORDERLINE - Mild steatosis possible")
    else:
        print("  → NORMAL range")
    
    print("\n" + "="*80)
