"""
NIFTI Conversion and Volume Processing Module
==============================================

Professional DICOM to NIFTI conversion for 3D volume analysis.
Handles multi-slice CT series and volumetric feature extraction.

Author: Bülent Tuğrul
Institution: Ankara Üniversitesi
"""

import numpy as np
import nibabel as nib
import pydicom
import SimpleITK as sitk
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import json


class NiftiConverter:
    """
    DICOM to NIFTI Conversion System
    
    Converts DICOM series to NIFTI format for volumetric analysis
    """
    
    def __init__(self):
        """Initialize converter"""
        self.reader = sitk.ImageSeriesReader()
    
    def dicom_series_to_nifti(self,
                             dicom_dir: str,
                             output_path: str,
                             series_id: Optional[str] = None) -> str:
        """
        Convert DICOM series to NIFTI
        
        Parameters:
        -----------
        dicom_dir : str
            Directory containing DICOM files
        output_path : str
            Output NIFTI file path (.nii or .nii.gz)
        series_id : str, optional
            Specific series ID to convert
            
        Returns:
        --------
        output_path : str
            Path to saved NIFTI file
        """
        dicom_dir = Path(dicom_dir)
        
        # Get series IDs
        series_ids = self.reader.GetGDCMSeriesIDs(str(dicom_dir))
        
        if not series_ids:
            raise ValueError(f"No DICOM series found in {dicom_dir}")
        
        # Select series
        if series_id and series_id in series_ids:
            selected_series = series_id
        else:
            selected_series = series_ids[0]
            print(f"Using series: {selected_series}")
        
        # Get DICOM files for series
        dicom_names = self.reader.GetGDCMSeriesFileNames(str(dicom_dir), selected_series)
        
        # Read series
        self.reader.SetFileNames(dicom_names)
        image =self.reader.Execute()
        
        # Write NIFTI
        sitk.WriteImage(image, output_path)
        
        print(f"✓ Converted {len(dicom_names)} DICOM files to {output_path}")
        print(f"  Volume size: {image.GetSize()}")
        print(f"  Spacing: {image.GetSpacing()}")
        
        return output_path
    
    def numpy_to_nifti(self,
                      volume: np.ndarray,
                      output_path: str,
                      affine: Optional[np.ndarray] = None,
                      spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> str:
        """
        Convert NumPy array to NIFTI
        
        Parameters:
        -----------
        volume : np.ndarray
            3D volume (D, H, W) or (H, W, D)
        output_path : str
            Output file path
        affine : np.ndarray, optional
            4x4 affine transformation matrix
        spacing : tuple
            Voxel spacing (x, y, z)
            
        Returns:
        --------
        output_path : str
            Path to saved file
        """
        # Create default affine if not provided
        if affine is None:
            affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0])
        
        # Create NIFTI image
        nifti_img = nib.Nifti1Image(volume, affine)
        
        # Save
        nib.save(nifti_img, output_path)
        
        print(f"✓ Saved NIFTI: {output_path}")
        print(f"  Shape: {volume.shape}")
        print(f"  Spacing: {spacing}")
        
        return output_path
    
    def nifti_to_numpy(self, nifti_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load NIFTI file to NumPy array
        
        Parameters:
        -----------
        nifti_path : str
            Path to NIFTI file
            
        Returns:
        --------
        volume : np.ndarray
            3D volume data
        affine : np.ndarray
            Affine transformation matrix
        """
        nifti_img = nib.load(nifti_path)
        volume = nifti_img.get_fdata()
        affine = nifti_img.affine
        
        return volume, affine
    
    def get_nifti_metadata(self, nifti_path: str) -> Dict:
        """
        Extract metadata from NIFTI file
        
        Parameters:
        -----------
        nifti_path : str
            Path to NIFTI file
            
        Returns:
        --------
        metadata : dict
            NIFTI metadata
        """
        nifti_img = nib.load(nifti_path)
        
        metadata = {
            'shape': nifti_img.shape,
            'affine': nifti_img.affine.tolist(),
            'header': dict(nifti_img.header),
            'pixdim': nifti_img.header['pixdim'].tolist(),
            'datatype': str(nifti_img.header.get_data_dtype())
        }
        
        return metadata


class VolumeProcessor:
    """
    3D Volume Processing for CT Data
    
    Handles volumetric analysis and feature extraction
    """
    
    @staticmethod
    def resample_volume(volume: np.ndarray,
                       original_spacing: Tuple[float, float, float],
                       new_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> np.ndarray:
        """
        Resample volume to isotropic spacing
        
        Parameters:
        -----------
        volume : np.ndarray
            Input volume
        original_spacing : tuple
            Original voxel spacing
        new_spacing : tuple
            Target voxel spacing
            
        Returns:
        --------
        resampled : np.ndarray
            Resampled volume
        """
        # Calculate new size
        original_size = volume.shape
        resize_factor = np.array(original_spacing) / np.array(new_spacing)
        new_size = (np.array(original_size) * resize_factor).astype(int)
        
        # Resample using SimpleITK
        sitk_image = sitk.GetImageFromArray(volume)
        sitk_image.SetSpacing(original_spacing[::-1])  # ITK uses (z, y, x)
        
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(new_spacing[::-1])
        resampler.SetSize(new_size[::-1].tolist())
        resampler.SetInterpolator(sitk.sitkLinear)
        
        resampled_sitk = resampler.Execute(sitk_image)
        resampled = sitk.GetArrayFromImage(resampled_sitk)
        
        return resampled
    
    @staticmethod
    def compute_volume_statistics(volume: np.ndarray,
                                  mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Compute volumetric statistics
        
        Parameters:
        -----------
        volume : np.ndarray
            3D volume
        mask : np.ndarray, optional
            ROI mask
            
        Returns:
        --------
        stats : dict
            Volume statistics
        """
        if mask is not None:
            roi_voxels = volume[mask > 0]
        else:
            roi_voxels = volume.ravel()
        
        stats = {
            'mean': float(np.mean(roi_voxels)),
            'median': float(np.median(roi_voxels)),
            'std': float(np.std(roi_voxels)),
            'min': float(np.min(roi_voxels)),
            'max': float(np.max(roi_voxels)),
            'p25': float(np.percentile(roi_voxels, 25)),
            'p75': float(np.percentile(roi_voxels, 75)),
            'volume_ml': float(np.sum(mask > 0) if mask is not None else volume.size) # Approximate
        }
        
        return stats
    
    @staticmethod
    def extract_slice_range(volume: np.ndarray,
                           start_slice: int,
                           end_slice: int) -> np.ndarray:
        """
        Extract slice range from volume
        
        Parameters:
        -----------
        volume : np.ndarray
            3D volume (slices, height, width)
        start_slice : int
            Start slice index
        end_slice : int
            End slice index
            
        Returns:
        --------
        slice_range : np.ndarray
            Extracted slices
        """
        return volume[start_slice:end_slice, :, :]
    
    @staticmethod
    def select_representative_slices(volume: np.ndarray,
                                    n_slices: int = 5,
                                    method: str = 'uniform') -> List[int]:
        """
        Select representative slices from volume
        
        Parameters:
        -----------
        volume : np.ndarray
            3D volume
        n_slices : int
            Number of slices to select
        method : str
            Selection method ('uniform', 'middle', 'variance')
            
        Returns:
        --------
        slice_indices : list
            Selected slice indices
        """
        n_total = volume.shape[0]
        
        if method == 'uniform':
            # Uniformly spaced
            indices = np.linspace(0, n_total-1, n_slices, dtype=int)
        elif method == 'middle':
            # Middle region
            mid = n_total // 2
            half_range = n_slices // 2
            indices = np.arange(mid - half_range, mid + half_range + (n_slices % 2))
        elif method == 'variance':
            # Select slices with highest variance (most information)
            variances = [np.var(volume[i]) for i in range(n_total)]
            indices = np.argsort(variances)[-n_slices:]
            indices = np.sort(indices)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return indices.tolist()


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("NIFTI Conversion & Volume Processing Module")
    print("="*80)
    
    # Create test volume
    test_volume = np.random.randn(100, 256, 256) * 30 + 50
    
    print("\nTest 1: NumPy to NIFTI Conversion")
    print("-" * 80)
    converter = NiftiConverter()
    
    output_path = "test_volume.nii.gz"
    converter.numpy_to_nifti(test_volume, output_path, spacing=(1.5, 0.7, 0.7))
    
    # Load back
    loaded_volume, affine = converter.nifti_to_numpy(output_path)
    print(f"✓ Loaded volume shape: {loaded_volume.shape}")
    
    print("\nTest 2: Volume Processing")
    print("-" * 80)
    processor = VolumeProcessor()
    
    # Statistics
    stats = processor.compute_volume_statistics(test_volume)
    print(f"Volume Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}")
    
    # Select representative slices
    slices = processor.select_representative_slices(test_volume, n_slices=5, method='middle')
    print(f"\nSelected slices (middle): {slices}")
    
    # Clean up
    import os
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"\n✓ Cleaned up test file")
    
    print("\n" + "="*80)
    print("✅ NIFTI Module Ready!")
    print("="*80)
