"""
DICOM Loader and Preprocessing Module
======================================

This module handles loading, preprocessing, and conversion of DICOM CT images
from the TCIA-LIHC dataset.

Features:
- DICOM metadata extraction
- HU (Hounsfield Unit) conversion
- Window/Level adjustment for different tissue contrasts
- Multi-phase CT handling (arterial, portal venous, delayed)
- Automatic liver/spleen ROI detection

Author: Bülent Tuğrul
"""

import pydicom
import numpy as np
import os
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import json
from dataclasses import dataclass, asdict
import cv2
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')


@dataclass
class DICOMMetadata:
    """DICOM metadata structure"""
    patient_id: str
    study_date: str
    series_description: str
    modality: str
    slice_thickness: float
    pixel_spacing: Tuple[float, float]
    image_position: Tuple[float, float, float]
    image_orientation: Tuple[float, ...]
    rows: int
    columns: int
    bits_stored: int
    rescale_slope: float
    rescale_intercept: float
    window_center: Optional[float] = None
    window_width: Optional[float] = None
    kvp: Optional[float] = None
    manufacturer: Optional[str] = None
    body_part_examined: Optional[str] = None


class DICOMLoader:
    """
    DICOM Loader for CT Images with HU conversion
    """
    
    # Standard CT window presets
    WINDOW_PRESETS = {
        'liver': {'center': 50, 'width': 150},      # Liver parenchyma
        'soft_tissue': {'center': 40, 'width': 400},  # General soft tissue
        'bone': {'center': 400, 'width': 1800},     # Bone
        'lung': {'center': -600, 'width': 1600},    # Lung
        'abdomen': {'center': 60, 'width': 400},    # Abdomen
    }
    
    def __init__(self, dataset_path: str):
        """
        Initialize DICOM Loader
        
        Parameters:
        -----------
        dataset_path : str
            Path to TCIA-DATASET-DICOM folder
        """
        self.dataset_path = Path(dataset_path)
        self.dicom_files = []
        self._scan_dataset()
    
    def _scan_dataset(self):
        """Scan dataset directory for DICOM files"""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")
        
        # Find all .dcm files recursively
        self.dicom_files = list(self.dataset_path.rglob("*.dcm"))
        print(f"Found {len(self.dicom_files)} DICOM files")
    
    def load_dicom(self, filepath: Union[str, Path]) -> Tuple[np.ndarray, DICOMMetadata]:
        """
        Load single DICOM file and convert to HU
        
        Parameters:
        -----------
        filepath : str or Path
            Path to DICOM file
            
        Returns:
        --------
        image_hu : np.ndarray
            Image in Hounsfield Units
        metadata : DICOMMetadata
            DICOM metadata
        """
        # Read DICOM
        dcm = pydicom.dcmread(filepath)
        
        # Extract pixel data
        image = dcm.pixel_array.astype(np.float64)
        
        # Convert to HU
        rescale_slope = float(getattr(dcm, 'RescaleSlope', 1.0))
        rescale_intercept = float(getattr(dcm, 'RescaleIntercept', 0.0))
        image_hu = image * rescale_slope + rescale_intercept
        
        # Extract metadata
        metadata = self._extract_metadata(dcm)
        
        return image_hu, metadata
    
    def _extract_metadata(self, dcm: pydicom.Dataset) -> DICOMMetadata:
        """Extract relevant metadata from DICOM"""
        
        # Pixel spacing
        if hasattr(dcm, 'PixelSpacing'):
            pixel_spacing = tuple(float(x) for x in dcm.PixelSpacing)
        else:
            pixel_spacing = (1.0, 1.0)
        
        # Image position
        if hasattr(dcm, 'ImagePositionPatient'):
            image_position = tuple(float(x) for x in dcm.ImagePositionPatient)
        else:
            image_position = (0.0, 0.0, 0.0)
        
        # Image orientation
        if hasattr(dcm, 'ImageOrientationPatient'):
            image_orientation = tuple(float(x) for x in dcm.ImageOrientationPatient)
        else:
            image_orientation = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        
        metadata = DICOMMetadata(
            patient_id=str(getattr(dcm, 'PatientID', 'Unknown')),
            study_date=str(getattr(dcm, 'StudyDate', 'Unknown')),
            series_description=str(getattr(dcm, 'SeriesDescription', 'Unknown')),
            modality=str(getattr(dcm, 'Modality', 'CT')),
            slice_thickness=float(getattr(dcm, 'SliceThickness', 1.0)),
            pixel_spacing=pixel_spacing,
            image_position=image_position,
            image_orientation=image_orientation,
            rows=int(dcm.Rows),
            columns=int(dcm.Columns),
            bits_stored=int(getattr(dcm, 'BitsStored', 16)),
            rescale_slope=float(getattr(dcm, 'RescaleSlope', 1.0)),
            rescale_intercept=float(getattr(dcm, 'RescaleIntercept', 0.0)),
            window_center=self._safe_float(getattr(dcm, 'WindowCenter', 40)),
            window_width=self._safe_float(getattr(dcm, 'WindowWidth', 400)),
            kvp=self._safe_float(getattr(dcm, 'KVP', 120)) if hasattr(dcm, 'KVP') else None,
            manufacturer=str(getattr(dcm, 'Manufacturer', 'Unknown')),
            body_part_examined=str(getattr(dcm, 'BodyPartExamined', 'LIVER'))
        )
        
        return metadata
    
    def _safe_float(self, value):
        """Safely convert DICOM value to float (handles multi-value)"""
        try:
            # If it's a sequence/list, take first value
            if hasattr(value, '__getitem__') and not isinstance(value, str):
                return float(value[0])
            return float(value)
        except (TypeError, IndexError, ValueError):
            return 40.0  # Default
    
    def apply_window(self,
                    image_hu: np.ndarray,
                    window_name: str = 'liver',
                    custom_center: Optional[float] = None,
                    custom_width: Optional[float] = None) -> np.ndarray:
        """
        Apply window/level adjustment
        
        Parameters:
        -----------
        image_hu : np.ndarray
            Image in HU
        window_name : str
            Preset window name
        custom_center : float, optional
            Custom window center
        custom_width : float, optional
            Custom window width
            
        Returns:
        --------
        windowed : np.ndarray
            Windowed image (0-255)
        """
        if custom_center is not None and custom_width is not None:
            center, width = custom_center, custom_width
        else:
            preset = self.WINDOW_PRESETS.get(window_name, self.WINDOW_PRESETS['liver'])
            center, width = preset['center'], preset['width']
        
        # Apply window
        min_val = center - width / 2
        max_val = center + width / 2
        
        windowed = np.clip(image_hu, min_val, max_val)
        windowed = ((windowed - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        
        return windowed
    
    def normalize_hu(self, image_hu: np.ndarray, 
                    hu_min: float = -200, 
                    hu_max: float = 300) -> np.ndarray:
        """
        Normalize HU values to [0, 1] range
        
        Parameters:
        -----------
        image_hu : np.ndarray
            Image in HU
        hu_min : float
            Minimum HU for normalization
        hu_max : float
            Maximum HU for normalization
            
        Returns:
        --------
        normalized : np.ndarray
            Normalized image [0, 1]
        """
        clipped = np.clip(image_hu, hu_min, hu_max)
        normalized = (clipped - hu_min) / (hu_max - hu_min)
        return normalized
    
    def load_series(self, patient_id: str, 
                   series_description_filter: Optional[str] = None) -> Tuple[np.ndarray, List[DICOMMetadata]]:
        """
        Load complete series for a patient
        
        Parameters:
        -----------
        patient_id : str
            Patient ID
        series_description_filter : str, optional
            Filter by series description (e.g., 'Portal', 'Arterial')
            
        Returns:
        --------
        volume : np.ndarray
            3D volume (slices, rows, cols)
        metadata_list : list
            List of metadata for each slice
        """
        # Find files for this patient
        patient_files = [f for f in self.dicom_files if patient_id in f.parts]
        
        if not patient_files:
            raise ValueError(f"No files found for patient {patient_id}")
        
        # Load and sort by slice location
        slices_data = []
        for filepath in patient_files:
            try:
                dcm = pydicom.dcmread(filepath)
                
                # Filter by series description if specified
                if series_description_filter:
                    series_desc = str(getattr(dcm, 'SeriesDescription', ''))
                    if series_description_filter.lower() not in series_desc.lower():
                        continue
                
                image_hu, metadata = self.load_dicom(filepath)
                slice_location = float(getattr(dcm, 'SliceLocation', 0.0))
                
                slices_data.append({
                    'slice_location': slice_location,
                    'image': image_hu,
                    'metadata': metadata
                })
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                continue
        
        # Sort by slice location
        slices_data.sort(key=lambda x: x['slice_location'])
        
        # Stack into volume
        volume = np.stack([s['image'] for s in slices_data], axis=0)
        metadata_list = [s['metadata'] for s in slices_data]
        
        return volume, metadata_list
    
    def detect_liver_roi(self, image_hu: np.ndarray, 
                        method: str = 'threshold') -> np.ndarray:
        """
        Simple liver ROI detection
        
        Parameters:
        -----------
        image_hu : np.ndarray
            CT image in HU
        method : str
            Detection method ('threshold', 'otsu')
            
        Returns:
        --------
        liver_mask : np.ndarray
            Binary liver mask
        """
        # Apply liver window
        windowed = self.apply_window(image_hu, 'liver')
        
        if method == 'threshold':
            # Simple thresholding (liver typically 40-70 HU)
            mask = (image_hu > 30) & (image_hu < 100)
        elif method == 'otsu':
            # Otsu thresholding
            _, mask = cv2.threshold(windowed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mask = mask > 0
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Keep largest connected component
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels > 1:
            # Find largest component (excluding background)
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = (labels == largest_label).astype(np.uint8)
        
        return mask.astype(bool)
    
    def save_preprocessed(self, 
                         image: np.ndarray,
                         metadata: DICOMMetadata,
                         output_path: Union[str, Path],
                         format: str = 'npz'):
        """
        Save preprocessed image with metadata
        
        Parameters:
        -----------
        image : np.ndarray
            Preprocessed image
        metadata : DICOMMetadata
            Image metadata
        output_path : str or Path
            Output file path
        format : str
            Save format ('npz', 'png', 'npy')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'npz':
            np.savez_compressed(output_path, 
                              image=image,
                              metadata=asdict(metadata))
        elif format == 'npy':
            np.save(output_path, image)
            # Save metadata separately as JSON
            json_path = output_path.with_suffix('.json')
            with open(json_path, 'w') as f:
                json.dump(asdict(metadata), f, indent=2)
        elif format == 'png':
            # Normalize to 0-255 for PNG
            if image.dtype != np.uint8:
                image_norm = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
            else:
                image_norm = image
            cv2.imwrite(str(output_path), image_norm)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        print(f"Saved to: {output_path}")


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("DICOM Loader - TCIA-LIHC Dataset")
    print("="*80)
    
    # Example path (update with actual path)
    dataset_path = "data/raw/TCIA-DATASET-DICOM"
    
    try:
        loader = DICOMLoader(dataset_path)
        print(f"\nDataset loaded: {len(loader.dicom_files)} files found")
        
        if loader.dicom_files:
            
            first_file = loader.dicom_files[0]
            print(f"\nLoading example file: {first_file.name}")
            
            image_hu, metadata = loader.load_dicom(first_file)
            
            print("\n" + "="*80)
            print("IMAGE INFORMATION:")
            print("="*80)
            print(f"Shape: {image_hu.shape}")
            print(f"HU Range: [{image_hu.min():.1f}, {image_hu.max():.1f}]")
            print(f"Mean HU: {image_hu.mean():.1f}")
            print(f"\nMetadata:")
            print(f"  Patient ID: {metadata.patient_id}")
            print(f"  Study Date: {metadata.study_date}")
            print(f"  Series: {metadata.series_description}")
            print(f"  Slice Thickness: {metadata.slice_thickness}mm")
            print(f"  Pixel Spacing: {metadata.pixel_spacing}")
            
            # Apply liver window
            windowed = loader.apply_window(image_hu, 'liver')
            print(f"\nLiver windowed image range: [0, 255]")
            
            # Detect liver ROI
            liver_mask = loader.detect_liver_roi(image_hu)
            liver_area = np.sum(liver_mask)
            print(f"Detected liver area: {liver_area} pixels")
            
    except FileNotFoundError as e:
        print(f"\nNote: {e}")
        print("Please update dataset_path with actual TCIA data location")
    
    print("\n" + "="*80)
    print("DICOM Loader ready for use!")
    print("="*80)
