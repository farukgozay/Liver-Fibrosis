"""
PyRadiomics Feature Extraction Module
======================================

Professional radiomic feature extraction using PyRadiomics library.
Extracts comprehensive texture, shape, and intensity features from liver ROI.

Integrated with:
- First-order statistics
- GLCM (Gray Level Co-occurrence Matrix)
- GLRLM (Gray Level Run Length Matrix)
- GLSZM (Gray Level Size Zone Matrix)
- GLDM (Gray Level Dependence Matrix)
- NGTDM (Neighbouring Gray Tone Difference Matrix)

Author: Bülent Tuğrul
Institution: Ankara Üniversitesi - Bilgisayar Mühendisliği
Project: Non-Invasive Liver Fibrosis Staging from CT Images
"""

import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor, setVerbosity
import pandas as pd
from typing import Dict, Optional, Union
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Suppress radiomics warnings
setVerbosity(40)


class RadiomicsExtractor:
    """
    Professional Radiomics Feature Extraction
    
    Uses PyRadiomics to extract comprehensive texture and shape features
    from liver CT images for fibrosis staging.
    """
    
    def __init__(self, 
                 bin_width: int = 25,
                 normalize: bool = True,
                 resample: bool = False):
        """
        Initialize Radiomics Extractor
        
        Parameters:
        -----------
        bin_width : int
            Bin width for histogram discretization
        normalize : bool
            Whether to normalize images
        resample : bool
            Whether to resample to isotropic voxels
        """
        self.bin_width = bin_width
        self.normalize = normalize
        self.resample = resample
        
        # Configure extractor with optimized settings
        self.settings = {
            'binWidth': bin_width,
            'resampledPixelSpacing': None if not resample else [1, 1, 1],
            'interpolator': sitk.sitkBSpline,
            'normalize': normalize,
            'normalizeScale': 100 if normalize else 1,
            'removeOutliers': None,
            
            # Enable feature classes
            'enableCExtensions': True,
        }
        
        # Initialize feature extractor
        self.extractor = featureextractor.RadiomicsFeatureExtractor(**self.settings)
        
        # Enable all feature classes
        self.extractor.enableAllImageTypes()
        self.extractor.enableAllFeatures()
        
        print("PyRadiomics Extractor initialized")
        print(f"Settings: bin_width={bin_width}, normalize={normalize}, resample={resample}")
    
    def extract_features(self,
                        image: Union[np.ndarray, sitk.Image],
                        mask: Union[np.ndarray, sitk.Image]) -> Dict[str, float]:
        """
        Extract all radiomics features
        
        Parameters:
        -----------
        image : np.ndarray or SimpleITK.Image
            Input CT image
        mask : np.ndarray or SimpleITK.Image
            Binary mask (ROI)
            
        Returns:
        --------
        features : dict
            Dictionary of radiomics features
        """
        # Convert to SimpleITK if numpy
        if isinstance(image, np.ndarray):
            image_sitk = sitk.GetImageFromArray(image.astype(np.float32))
        else:
            image_sitk = image
        
        if isinstance(mask, np.ndarray):
            mask_sitk = sitk.GetImageFromArray(mask.astype(np.uint8))
        else:
            mask_sitk = mask
        
        # Extract features
        try:
            result = self.extractor.execute(image_sitk, mask_sitk)
            
            # Convert to regular dict and filter out diagnostics
            features = {}
            for key, value in result.items():
                if not key.startswith('diagnostics_'):
                    # Store only numeric features
                    if isinstance(value, (int, float, np.number)):
                        features[key] = float(value)
            
            return features
            
        except Exception as e:
            print(f"Error extracting radiomics features: {e}")
            return {}
    
    def extract_feature_groups(self,
                              image: Union[np.ndarray, sitk.Image],
                              mask: Union[np.ndarray, sitk.Image]) -> Dict[str, Dict[str, float]]:
        """
        Extract features grouped by category
        
        Returns:
        --------
        grouped_features : dict of dict
            Features grouped by type (shape, firstorder, glcm, etc.)
        """
        all_features = self.extract_features(image, mask)
        
        # Group features by category
        groups = {
            'shape': {},
            'firstorder': {},
            'glcm': {},
            'glrlm': {},
            'glszm': {},
            'gldm': {},
            'ngtdm': {}
        }
        
        for key, value in all_features.items():
            for group_name in groups.keys():
                if group_name in key.lower():
                    # Clean feature name
                    clean_key = key.replace('original_', '').replace(f'{group_name}_', '')
                    groups[group_name][clean_key] = value
                    break
        
        return groups
    
    def extract_to_dataframe(self,
                            image: Union[np.ndarray, sitk.Image],
                            mask: Union[np.ndarray, sitk.Image],
                            patient_id: Optional[str] = None) -> pd.DataFrame:
        """
        Extract features and return as DataFrame
        
        Parameters:
        -----------
        image : np.ndarray or sitk.Image
            Input image
        mask : np.ndarray or sitk.Image
            Binary mask
        patient_id : str, optional
            Patient identifier
            
        Returns:
        --------
        df : pd.DataFrame
            Features as DataFrame row
        """
        features = self.extract_features(image, mask)
        
        # Add patient ID if provided
        if patient_id:
            features['patient_id'] = patient_id
        
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        return df
    
    def get_feature_names(self) -> list:
        """Get list of all enabled feature names"""
        return list(self.extractor.enabledFeatures.keys())
    
    def get_image_types(self) -> list:
        """Get list of all enabled image filter types"""
        return list(self.extractor.enabledImagetypes.keys())


class AdvancedTextureFeatures:
    """
    Additional advanced texture features beyond PyRadiomics
    
    Complements PyRadiomics with custom features specific to
    liver fibrosis detection.
    """
    
    @staticmethod
    def compute_fractal_dimension(image: np.ndarray, mask: np.ndarray) -> float:
        """
        Compute fractal dimension using box-counting method
        
        Useful for quantifying texture complexity in cirrhotic liver
        """
        # Extract ROI
        roi = image[mask > 0]
        
        if len(roi) == 0:
            return 0.0
        
        # Binarize
        threshold = np.median(roi)
        binary = (image > threshold).astype(int) * mask
        
        # Box-counting
        scales = np.logspace(0.5, 3, num=10, base=2, dtype=int)
        counts = []
        
        for scale in scales:
            if scale >= min(binary.shape):
                continue
            
            # Count boxes containing foreground pixels
            h, w = binary.shape[0] // scale, binary.shape[1] // scale
            boxes = binary[:h*scale, :w*scale].resh ape(h, scale, w, scale)
            box_counts = (boxes.sum(axis=(1,3)) > 0).sum()
            counts.append(box_counts)
        
        # Fit line in log-log plot
        if len(counts) > 1:
            scales_used = scales[:len(counts)]
            coeffs = np.polyfit(np.log(scales_used), np.log(counts), 1)
            fractal_dim = -coeffs[0]
            return float(fractal_dim)
        
        return 0.0
    
    @staticmethod
    def compute_local_binary_pattern(image: np.ndarray, 
                                     mask: np.ndarray,
                                     radius: int = 1,
                                     n_points: int = 8) -> Dict[str, float]:
        """
        Compute Local Binary Pattern (LBP) features
        
        Parameters:
        -----------
        image : np.ndarray
            Input image
        mask : np.ndarray
            ROI mask
        radius : int
            LBP radius
        n_points : int
            Number of points in circle
            
        Returns:
        --------
        lbp_features : dict
            LBP histogram features
        """
        from skimage import feature
        
        # Compute LBP
        lbp = feature.local_binary_pattern(image, n_points, radius, method='uniform')
        
        # Extract LBP values in ROI
        lbp_roi = lbp[mask > 0]
        
        if len(lbp_roi) == 0:
            return {'lbp_mean': 0.0, 'lbp_std': 0.0, 'lbp_energy': 0.0}
        
        # Histogram
        hist, _ = np.histogram(lbp_roi, bins=n_points + 2, density=True)
        
        features = {
            'lbp_mean': float(np.mean(lbp_roi)),
            'lbp_std': float(np.std(lbp_roi)),
            'lbp_energy': float(np.sum(hist ** 2)),
            'lbp_entropy': float(-np.sum(hist * np.log2(hist + 1e-10)))
        }
        
        return features


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("PyRadiomics Feature Extraction - Professional Module")
    print("="*80)
    
    # Create test data
    test_image = np.random.randn(256, 256) * 30 + 50
    test_mask = np.zeros((256, 256), dtype=np.uint8)
    test_mask[80:180, 80:180] = 1
    
    # Initialize extractor
    print("\nInitializing PyRadiomics extractor...")
    extractor = RadiomicsExtractor(bin_width=25, normalize=True)
    
    # Extract features
    print("\nExtracting radiomics features...")
    features = extractor.extract_features(test_image, test_mask)
    
    print(f"\n✓ Extracted {len(features)} radiomics features")
    
    # Group features
    grouped = extractor.extract_feature_groups(test_image, test_mask)
    
    print("\nFeature Groups:")
    for group, feats in grouped.items():
        if feats:
            print(f"  {group.upper()}: {len(feats)} features")
    
    # Show sample features
    print("\n" + "="*80)
    print("SAMPLE RADIOMICS FEATURES:")
    print("="*80)
    sample_features = list(features.items())[:10]
    for name, value in sample_features:
        print(f"{name:<50} {value:>15.4f}")
    
    # Advanced texture features
    print("\n" + "="*80)
    print("ADVANCED TEXTURE FEATURES:")
    print("="*80)
    
    advanced = AdvancedTextureFeatures()
    fractal_dim = advanced.compute_fractal_dimension(test_image, test_mask)
    lbp_feats = advanced.compute_local_binary_pattern(test_image, test_mask)
    
    print(f"Fractal Dimension: {fractal_dim:.4f}")
    print(f"LBP Features: {lbp_feats}")
    
    print("\n" + "="*80)
    print("✅ PyRadiomics Module Ready!")
    print("="*80)
