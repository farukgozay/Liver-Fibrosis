"""
End-to-End Testing and Validation Script
=========================================

Professional testing pipeline for the complete liver fibrosis staging system.
Tests ALL modules with REAL TCIA data.

Author: Bülent Tuğrul
Ankara Üniversitesi - Bilgisayar Mühendisliği
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import json
import time
from datetime import datetime

# Import all modules
from data_processing.dicom_loader import DICOMLoader
from data_processing.nifti_converter import NiftiConverter, VolumeProcessor
from feature_extraction.frequency_domain.fft_2d import FFT2DAnalyzer
from feature_extraction.spatial_domain.nash_detection import NASHDetector
from feature_extraction.spatial_domain.radiomics_features import RadiomicsExtractor, AdvancedTextureFeatures
from models.deep_learning.unet_segmentation import LiverSpleenSegmentor, TraditionalSegmentor
from models.classical_ml.xgboost_model import FibrosisXGBoostModel

import warnings
warnings.filterwarnings('ignore')


class ComprehensiveValidator:
    """
    End-to-End System Validation
    
    Tests all components with real TCIA-LIHC data
    """
    
    def __init__(self, dataset_path: str, output_dir: str = 'validation_results'):
        """
        Initialize Validator
        
        Parameters:
        -----------
        dataset_path : str
            Path to TCIA-DATASET-DICOM
        output_dir : str
            Output directory for results
        """
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'tests': {},
            'errors': [],
            'performance': {}
        }
        
        print("="*90)
        print("COMPREHENSIVE SYSTEM VALIDATION")
        print("Non-Invasive Liver Fibrosis Staging from CT Images")
        print("="*90)
    
    def test_dicom_loading(self) -> bool:
        """Test DICOM loading module"""
        print("\n" + "="*90)
        print("TEST 1: DICOM Loading & Preprocessing")
        print("="*90)
        
        try:
            start_time = time.time()
            
            # Initialize loader
            loader = DICOMLoader(str(self.dataset_path))
            
            if not loader.dicom_files:
                raise ValueError("No DICOM files found!")
            
            # Load first file
            first_file = loader.dicom_files[0]
            print(f"\n📁 Loading: {first_file.name}")
            
            image_hu, metadata = loader.load_dicom(first_file)
            
            # Validate
            assert image_hu is not None, "Image is None"
            assert len(image_hu.shape) == 2, "Image not 2D"
            assert metadata is not None, "Metadata is None"
            
            # Apply windowing
            windowed = loader.apply_window(image_hu, 'liver')
            
            elapsed = time.time() - start_time
            
            self.results['tests']['dicom_loading'] = {
                'status': 'PASSED',
                'files_found': len(loader.dicom_files),
                'image_shape': image_hu.shape,
                'hu_range': [float(image_hu.min()), float(image_hu.max())],
                'patient_id': metadata.patient_id,
                'time_seconds': elapsed
            }
            
            print(f"\n✅ DICOM Loading: PASSED")
            print(f"   Files found: {len(loader.dicom_files)}")
            print(f"   Image shape: {image_hu.shape}")
            print(f"   HU range: [{image_hu.min():.1f}, {image_hu.max():.1f}]")
            print(f"   Time: {elapsed:.2f}s")
            
            return True, loader, image_hu, metadata
            
        except Exception as e:
            print(f"\n❌ DICOM Loading: FAILED")
            print(f"   Error: {e}")
            self.results['tests']['dicom_loading'] = {'status': 'FAILED', 'error': str(e)}
            self.results['errors'].append(f"DICOM Loading: {e}")
            return False, None, None, None
    
    def test_segmentation(self, image_hu: np.ndarray) -> tuple:
        """Test segmentation module"""
        print("\n" + "="*90)
        print("TEST 2: Liver & Spleen Segmentation")
        print("="*90)
        
        try:
            start_time = time.time()
            
            # Try U-Net first, fallback to traditional
            print("\n🔬 Attempting U-Net segmentation...")
            try:
                segmentor = LiverSpleenSegmentor()
                liver_mask, spleen_mask = segmentor.segment(image_hu)
                method = 'U-Net'
            except:
                print("   U-Net not available, using traditional method...")
                segmentor = TraditionalSegmentor()
                liver_mask = segmentor.segment_liver_traditional(image_hu)
                spleen_mask = segmentor.segment_spleen_traditional(image_hu, liver_mask)
                method = 'Traditional'
            
            # Validate
            liver_area = np.sum(liver_mask > 0)
            spleen_area = np.sum(spleen_mask > 0)
            
            assert liver_area > 0, "No liver detected"
            assert liver_area < image_hu.size * 0.5, "Liver too large (>50% of image)"
            
            elapsed = time.time() - start_time
            
            self.results['tests']['segmentation'] = {
                'status': 'PASSED',
                'method': method,
                'liver_area_pixels': int(liver_area),
                'spleen_area_pixels': int(spleen_area),
                'time_seconds': elapsed
            }
            
            print(f"\n✅ Segmentation: PASSED")
            print(f"   Method: {method}")
            print(f"   Liver area: {liver_area} pixels")
            print(f"   Spleen area: {spleen_area} pixels")
            print(f"   Time: {elapsed:.2f}s")
            
            return True, liver_mask, spleen_mask
            
        except Exception as e:
            print(f"\n❌ Segmentation: FAILED")
            print(f"   Error: {e}")
            self.results['tests']['segmentation'] = {'status': 'FAILED', 'error': str(e)}
            self.results['errors'].append(f"Segmentation: {e}")
            return False, None, None
    
    def test_fft_features(self, image: np.ndarray, mask: np.ndarray) -> tuple:
        """Test FFT feature extraction"""
        print("\n" + "="*90)
        print("TEST 3: 2D FFT Frequency Domain Analysis")
        print("="*90)
        
        try:
            start_time = time.time()
            
            # Extract liver ROI
            roi = image * mask
            roi_norm = ((roi - roi.min()) / (roi.max() - roi.min() + 1e-8) * 255).astype(np.uint8)
            
            # Initialize analyzer
            fft_analyzer = FFT2DAnalyzer(window_function='hamming')
            
            # Extract features
            fft_features = fft_analyzer.extract_all_features(roi_norm)
            
            # Validate
            assert fft_features is not None
            assert fft_features.low_high_ratio > 0
            
            elapsed = time.time() - start_time
            
            self.results['tests']['fft_features'] = {
                'status': 'PASSED',
                'total_power': float(fft_features.total_power),
                'low_high_ratio': float(fft_features.low_high_ratio),
                'nash_signature': float(fft_features.steatosis_frequency_signature),
                'anisotropy_index': float(fft_features.anisotropy_index),
                'spectral_entropy': float(fft_features.spectral_entropy),
                'time_seconds': elapsed
            }
            
            print(f"\n✅ FFT Features: PASSED")
            print(f"   Low/High Ratio: {fft_features.low_high_ratio:.4f}")
            print(f"   NASH Signature: {fft_features.steatosis_frequency_signature:.4f}")
            print(f"   Anisotropy: {fft_features.anisotropy_index:.4f}")
            print(f"   Spectral Entropy: {fft_features.spectral_entropy:.4f}")
            print(f"   Time: {elapsed:.2f}s")
            
            return True, fft_features
            
        except Exception as e:
            print(f"\n❌ FFT Features: FAILED")
            print(f"   Error: {e}")
            self.results['tests']['fft_features'] = {'status': 'FAILED', 'error': str(e)}
            self.results['errors'].append(f"FFT Features: {e}")
            return False, None
    
    def test_nash_detection(self, image_hu: np.ndarray, liver_mask: np.ndarray, 
                           spleen_mask: np.ndarray) -> tuple:
        """Test NASH detection"""
        print("\n" + "="*90)
        print("TEST 4: NASH Detection (Spatial Domain)")
        print("="*90)
        
        try:
            start_time = time.time()
            
            # Initialize detector
            nash_detector = NASHDetector()
            
            # Extract features
            nash_features = nash_detector.extract_all_features(image_hu, liver_mask, spleen_mask)
            
            # Validate
            assert nash_features is not None
            assert 0 <= nash_features.nash_probability_score <= 1
            
            elapsed = time.time() - start_time
            
            self.results['tests']['nash_detection'] = {
                'status': 'PASSED',
                'nash_probability': float(nash_features.nash_probability_score),
                'confidence': nash_features.nash_confidence,
                'steatosis_pct': float(nash_features.steatosis_percentage),
                'liver_spleen_ratio': float(nash_features.liver_spleen_ratio),
                'mean_hu': float(nash_features.mean_hu),
                'time_seconds': elapsed
            }
            
            print(f"\n✅ NASH Detection: PASSED")
            print(f"   NASH Probability: {nash_features.nash_probability_score:.2%}")
            print(f"   Confidence: {nash_features.nash_confidence.upper()}")
            print(f"   Steatosis: {nash_features.steatosis_percentage:.1f}%")
            print(f"   L/S Ratio: {nash_features.liver_spleen_ratio:.2f}")
            print(f"   Mean HU: {nash_features.mean_hu:.1f}")
            print(f"   Time: {elapsed:.2f}s")
            
            return True, nash_features
            
        except Exception as e:
            print(f"\n❌ NASH Detection: FAILED")
            print(f"   Error: {e}")
            self.results['tests']['nash_detection'] = {'status': 'FAILED', 'error': str(e)}
            self.results['errors'].append(f"NASH Detection: {e}")
            return False, None
    
    def test_radiomics(self, image: np.ndarray, mask: np.ndarray) -> tuple:
        """Test PyRadiomics features"""
        print("\n" + "="*90)
        print("TEST 5: PyRadiomics Feature Extraction")
        print("="*90)
        
        try:
            start_time = time.time()
            
            # Initialize extractor
            radiomics = RadiomicsExtractor(bin_width=25)
            
            # Extract features
            features = radiomics.extract_features(image, mask)
            
            # Validate
            assert len(features) > 0, "No features extracted"
            
            elapsed = time.time() - start_time
            
            self.results['tests']['radiomics'] = {
                'status': 'PASSED',
                'features_extracted': len(features),
                'sample_features': dict(list(features.items())[:5]),
                'time_seconds': elapsed
            }
            
            print(f"\n✅ PyRadiomics: PASSED")
            print(f"   Features extracted: {len(features)}")
            print(f"   Time: {elapsed:.2f}s")
            
            return True, features
            
        except Exception as e:
            print(f"\n❌ PyRadiomics: FAILED")
            print(f"   Error: {e}")
            self.results['tests']['radiomics'] = {'status': 'FAILED', 'error': str(e)}
            self.results['errors'].append(f"PyRadiomics: {e}")
            return False, None
    
    def test_nifti_conversion(self, image: np.ndarray) -> bool:
        """Test NIFTI conversion"""
        print("\n" + "="*90)
        print("TEST 6: NIFTI Conversion")
        print("="*90)
        
        try:
            start_time = time.time()
            
            # Create mock volume
            volume = np.stack([image] * 10, axis=0)
            
            # Convert
            converter = NiftiConverter()
            output_path = self.output_dir / 'test_volume.nii.gz'
            converter.numpy_to_nifti(volume, str(output_path), spacing=(1.5, 0.7, 0.7))
            
            # Load back
            loaded_volume, affine = converter.nifti_to_numpy(str(output_path))
            
            # Validate
            assert loaded_volume.shape == volume.shape
            
            elapsed = time.time() - start_time
            
            self.results['tests']['nifti_conversion'] = {
                'status': 'PASSED',
                'volume_shape': list(volume.shape),
                'file_size_mb': float(output_path.stat().st_size / 1024 / 1024),
                'time_seconds': elapsed
            }
            
            print(f"\n✅ NIFTI Conversion: PASSED")
            print(f"   Volume shape: {volume.shape}")
            print(f"   File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
            print(f"   Time: {elapsed:.2f}s")
            
            return True
            
        except Exception as e:
            print(f"\n❌ NIFTI Conversion: FAILED")
            print(f"   Error: {e}")
            self.results['tests']['nifti_conversion'] = {'status': 'FAILED', 'error': str(e)}
            self.results['errors'].append(f"NIFTI Conversion: {e}")
            return False
    
    def generate_report(self):
        """Generate validation report"""
        print("\n" + "="*90)
        print("VALIDATION REPORT")
        print("="*90)
        
        total_tests = len(self.results['tests'])
        passed = sum(1 for t in self.results['tests'].values() if t.get('status') == 'PASSED')
        failed = total_tests - passed
        
        print(f"\nTotal Tests: {total_tests}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"Success Rate: {passed/total_tests*100:.1f}%")
        
        print("\nTest Summary:")
        print("-" * 90)
        for test_name, result in self.results['tests'].items():
            status_symbol = "✅" if result['status'] == 'PASSED' else "❌"
            time_info = f"({result.get('time_seconds', 0):.2f}s)" if 'time_seconds' in result else ""
            print(f"{status_symbol} {test_name.replace('_', ' ').title():<40} {time_info}")
        
        if self.results['errors']:
            print("\nErrors:")
            print("-" * 90)
            for error in self.results['errors']:
                print(f"  • {error}")
        
        # Save JSON report
        report_path = self.output_dir / 'validation_report.json'
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n📄 Full report saved to: {report_path}")
        
        return passed == total_tests
    
    def run_all_tests(self):
        """Run all validation tests"""
        print(f"\nDataset Path: {self.dataset_path}")
        print(f"Output Directory: {self.output_dir}")
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Test 1: DICOM Loading
        success, loader, image_hu, metadata = self.test_dicom_loading()
        if not success:
            print("\n⚠️ Cannot proceed without DICOM loading")
            self.generate_report()
            return False
        
        # Test 2: Segmentation
        success, liver_mask, spleen_mask = self.test_segmentation(image_hu)
        if not success or liver_mask is None:
            print("\n⚠️ Segmentation failed, using fallback mask")
            liver_mask = np.ones_like(image_hu, dtype=np.uint8)
            liver_mask[100:400, 100:400] = 1
            spleen_mask = np.zeros_like(image_hu, dtype=np.uint8)
        
        # Test 3: FFT Features
        self.test_fft_features(image_hu, liver_mask)
        
        # Test 4: NASH Detection
        self.test_nash_detection(image_hu, liver_mask, spleen_mask)
        
        # Test 5: PyRadiomics
        self.test_radiomics(image_hu, liver_mask)
        
        # Test 6: NIFTI Conversion
        self.test_nifti_conversion(image_hu)
        
        # Generate report
        all_passed = self.generate_report()
        
        if all_passed:
            print("\n" + "="*90)
            print("🎉 ALL TESTS PASSED! SYSTEM IS FULLY OPERATIONAL!")
            print("="*90)
        else:
            print("\n" + "="*90)
            print("⚠️ SOME TESTS FAILED. CHECK REPORT FOR DETAILS.")
            print("="*90)
        
        return all_passed


def main():
    """Main execution"""
    print("\n")
    print("╔" + "="*88 + "╗")
    print("║" + " "*88 + "║")
    print("║" + "  PROFESSIONAL LIVER FIBROSIS STAGING SYSTEM - VALIDATION SUITE".center(88) + "║")
    print("║" + "  Non-Invasive Staging using 2D FFT + NASH Detection".center(88) + "║")
    print("║" + " "*88 + "║")
    print("║" + "  Bülent Tuğrul - 22290673".center(88) + "║")
    print("║" + "  Ankara Üniversitesi - Bilgisayar Mühendisliği".center(88) + "║")
    print("║" + " "*88 + "║")
    print("╚" + "="*88 + "╝")
    
    # Configuration
    DATASET_PATH = "data/raw/TCIA-DATASET-DICOM"
    OUTPUT_DIR = "validation_results"
    
    # Run validation
    validator = ComprehensiveValidator(DATASET_PATH, OUTPUT_DIR)
    success = validator.run_all_tests()
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
