"""
Main Pipeline: Liver Fibrosis Staging with FFT + NASH Detection
================================================================

This script integrates all components:
1. DICOM loading
2. FFT-based frequency domain feature extraction
3. NASH detection (spatial domain)
4. XGBoost classification
5. SHAP explainability

Author: Bülent Tuğrul
Project: Karaciğer Fibrozunun BT Görüntülerinden Non-İnvaziv Evrelendirilmesi
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
from typing import Dict, List

# Add src to path
sys.path.append(str(Path(__file__).parents[1]))

from data_processing.dicom_loader import DICOMLoader
from feature_extraction.frequency_domain.fft_2d import FFT2DAnalyzer, FFTFeatures
from feature_extraction.spatial_domain.nash_detection import NASHDetector, NASHFeatures
from models.classical_ml.xgboost_model import FibrosisXGBoostModel

import warnings
warnings.filterwarnings('ignore')


class LiverFibrosisPipeline:
    """
    Complete Pipeline for Liver Fibrosis Staging
    
    Integrates:
    - DICOM preprocessing
    - Frequency domain (FFT) features
    - Spatial domain (NASH) features
    - XGBoost classification
    - SHAP explanations
    """
    
    def __init__(self,
                 dataset_path: str,
                 output_dir: str = 'results'):
        """
        Initialize Pipeline
        
        Parameters:
        -----------
        dataset_path : str
            Path to TCIA-DATASET-DICOM folder
        output_dir : str
            Output directory for results
        """
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        print("Initializing Pipeline Components...")
        self.dicom_loader = DICOMLoader(str(dataset_path))
        self.fft_analyzer = FFT2DAnalyzer(window_function='hamming')
        self.nash_detector = NASHDetector()
        self.model = None
        
        print("Pipeline initialized successfully!")
    
    def extract_features_from_image(self,
                                    image_hu: np.ndarray,
                                    liver_mask: np.ndarray,
                                    spleen_mask: np.ndarray = None) -> Dict[str, float]:
        """
        Extract all features from a single image
        
        Parameters:
        -----------
        image_hu : np.ndarray
            CT image in HU
        liver_mask : np.ndarray
            Liver segmentation mask
        spleen_mask : np.ndarray, optional
            Spleen segmentation mask
            
        Returns:
        --------
        features : dict
            Complete feature dictionary
        """
        # Extract liver ROI
        liver_roi = image_hu * liver_mask
        liver_roi_pixels = image_hu[liver_mask > 0]
        
        # Normalize for FFT (FFT works on grayscale image)
        liver_roi_norm = ((liver_roi - liver_roi.min()) / 
                         (liver_roi.max() - liver_roi.min() + 1e-8) * 255).astype(np.uint8)
        
        # 1. FFT Features (Frequency Domain)
        print("  Extracting FFT features...")
        fft_features = self.fft_analyzer.extract_all_features(liver_roi_norm)
        fft_dict = {f'fft_{k}': v for k, v in fft_features.__dict__.items()}
        
        # 2. NASH Features (Spatial Domain)
        print("  Extracting NASH features...")
        nash_features = self.nash_detector.extract_all_features(
            image_hu, liver_mask, spleen_mask
        )
        nash_dict = {f'nash_{k}': v for k, v in nash_features.__dict__.items() 
                    if isinstance(v, (int, float))}
        
        # 3. Basic HU Statistics
        hu_stats = {
            'hu_mean': float(np.mean(liver_roi_pixels)),
            'hu_median': float(np.median(liver_roi_pixels)),
            'hu_std': float(np.std(liver_roi_pixels)),
            'hu_min': float(np.min(liver_roi_pixels)),
            'hu_max': float(np.max(liver_roi_pixels)),
            'hu_p25': float(np.percentile(liver_roi_pixels, 25)),
            'hu_p75': float(np.percentile(liver_roi_pixels, 75)),
        }
        
        # Combine all features
        all_features = {**fft_dict, **nash_dict, **hu_stats}
        
        return all_features
    
    def process_patient(self,
                       patient_id: str,
                       series_filter: str = 'Portal') -> pd.DataFrame:
        """
        Process all images for a patient
        
        Parameters:
        -----------
        patient_id : str
            Patient ID
        series_filter : str
            Series description filter
            
        Returns:
        --------
        features_df : pd.DataFrame
            Features for all slices
        """
        print(f"\nProcessing Patient: {patient_id}")
        print("="*60)
        
        try:
            # Load series
            print(f"Loading {series_filter} phase series...")
            volume, metadata_list = self.dicom_loader.load_series(
                patient_id, series_filter
            )
            
            n_slices = volume.shape[0]
            print(f"Loaded {n_slices} slices")
            
            # Process middle slices (avoid top/bottom)
            mid_start = n_slices // 3
            mid_end = 2 * n_slices // 3
            
            features_list = []
            
            for idx in range(mid_start, mid_end):
                print(f"\n  Processing slice {idx}/{n_slices}...")
                
                # Get image
                image_slice = volume[idx]
                
                # Detect liver ROI
                liver_mask = self.dicom_loader.detect_liver_roi(image_slice)
                
                if not np.any(liver_mask):
                    print(f"    Warning: No liver detected in slice {idx}, skipping...")
                    continue
                
                # Extract features
                slice_features = self.extract_features_from_image(
                    image_slice, liver_mask
                )
                
                # Add metadata
                slice_features['patient_id'] = patient_id
                slice_features['slice_idx'] = idx
                
                features_list.append(slice_features)
            
            # Create DataFrame
            features_df = pd.DataFrame(features_list)
            
            print(f"\nExtracted features from {len(features_df)} slices")
            print(f"Total features per slice: {len(features_df.columns) - 2}")  # Exclude patient_id, slice_idx
            
            return features_df
            
        except Exception as e:
            print(f"Error processing patient {patient_id}: {e}")
            return pd.DataFrame()
    
    def aggregate_patient_features(self, features_df: pd.DataFrame) -> Dict[str, float]:
        """
        Aggregate slice-level features to patient-level
        
        Parameters:
        -----------
        features_df : pd.DataFrame
           Features from all slices
            
        Returns:
        --------
        aggregated : dict
            Patient-level features
        """
        # Exclude metadata columns
        feature_cols = [c for c in features_df.columns 
                       if c not in ['patient_id', 'slice_idx']]
        
        aggregated = {}
        
        for col in feature_cols:
            # Skip non-numeric
            if not pd.api.types.is_numeric_dtype(features_df[col]):
                continue
            
            # Compute statistics
            aggregated[f'{col}_mean'] = features_df[col].mean()
            aggregated[f'{col}_std'] = features_df[col].std()
            aggregated[f'{col}_median'] = features_df[col].median()
            aggregated[f'{col}_max'] = features_df[col].max()
            aggregated[f'{col}_min'] = features_df[col].min()
        
        return aggregated
    
    def train_model(self,
                   features_df: pd.DataFrame,
                   labels_df: pd.DataFrame,
                   task: str = 'multiclass',
                   optimize: bool = True):
        """
        Train fibrosis staging model
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            Patient-level features
        labels_df : pd.DataFrame
            Ground truth labels (patient_id, fibrosis_stage)
        task : str
            Classification task
        optimize : bool
            Whether to optimize hyperparameters
        """
        print("\n" + "="*80)
        print("TRAINING MODEL")
        print("="*80)
        
        # Merge features with labels
        data = features_df.merge(labels_df, on='patient_id', how='inner')
        
        print(f"Total samples: {len(data)}")
        print(f"Feature dimensions: {len(data.columns) - 2}")  # Exclude patient_id, fibrosis_stage
        
        # Split features and labels
        X = data.drop(['patient_id', 'fibrosis_stage'], axis=1).values
        y = data['fibrosis_stage'].values
        feature_names = list(data.drop(['patient_id', 'fibrosis_stage'], axis=1).columns)
        
        # Train/test split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Initialize and train model
        self.model = FibrosisXGBoostModel(task=task, use_gpu=False)
        
        history = self.model.train(
            X_train, y_train,
            feature_names=feature_names,
            optimize_hyperparams=optimize,
            n_trials=30 if optimize else 0
        )
        
        # Evaluate
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)
        
        metrics = self.model.evaluate(X_test, y_test)
        
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        if metrics.get('auc'):
            print(f"AUC: {metrics['auc']:.4f}")
        
        print("\nConfusion Matrix:")
        print(metrics['confusion_matrix'])
        
        # Feature importance
        print("\n" + "="*80)
        print("TOP 20 IMPORTANT FEATURES")
        print("="*80)
        
        importance_df = self.model.get_feature_importance(top_n=20)
        print(importance_df.to_string(index=False))
        
        # Save model
        model_path = self.output_dir / f'xgboost_liver_fibrosis_{task}.pkl'
        self.model.save_model(str(model_path))
        
        # Save metrics
        metrics_path = self.output_dir / f'metrics_{task}.json'
        import json
        with open(metrics_path, 'w') as f:
            # Convert numpy arrays to lists for JSON
            metrics_json = {
                'accuracy': float(metrics['accuracy']),
                'f1_score': float(metrics['f1_score']),
                'auc': float(metrics['auc']) if metrics.get('auc') else None,
                'confusion_matrix': metrics['confusion_matrix'].tolist()
            }
            json.dump(metrics_json, f, indent=2)
        
        print(f"\nModel saved to: {model_path}")
        print(f"Metrics saved to: {metrics_path}")
        
        return metrics


def main():
    """Main execution function"""
    print("="*80)
    print("LIVER FIBROSIS STAGING PIPELINE")
    print("FFT-Based Frequency Domain + NASH Detection")
    print("Bülent Tuğrul - Ankara Üniversitesi")
    print("="*80)
    
    # Configuration
    DATASET_PATH = "data/raw/TCIA-DATASET-DICOM"
    OUTPUT_DIR = "results"
    
    # Initialize pipeline
    pipeline = LiverFibrosisPipeline(DATASET_PATH, OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("PIPELINE COMPONENTS INITIALIZED")
    print("="*80)
    print("✓ DICOM Loader")
    print("✓ 2D FFT Analyzer (Frequency Domain)")
    print("✓ NASH Detector (Spatial Domain)")
    print("✓ XGBoost Classifier")
    
    print("\n" + "="*80)
    print("READY FOR DATA PROCESSING")
    print("="*80)
    print("\nTo process a patient:")
    print("  features_df = pipeline.process_patient('PATIENT_ID')")
    print("\nTo train model:")
    print("  pipeline.train_model(features_df, labels_df)")
    
    # Example: Process first patient (if exists)
    if pipeline.dicom_loader.dicom_files:
        # Get unique patient IDs
        patient_ids = set()
        for f in pipeline.dicom_loader.dicom_files[:100]:  # Sample first 100 files
            try:
                dcm = __import__('pydicom').dcmread(f)
                patient_ids.add(str(getattr(dcm, 'PatientID', 'Unknown')))
            except:
                continue
        
        if patient_ids:
            example_patient = list(patient_ids)[0]
            print(f"\nExample patient found: {example_patient}")
            print("To process this patient, run:")
            print(f"  features = pipeline.process_patient('{example_patient}')")
    
    return pipeline


if __name__ == "__main__":
    pipeline = main()
    
    print("\n" + "="*80)
    print("PIPELINE READY!")
    print("="*80)
    print("\n🎯 CAN ALICI NOKTALAR:")
    print("  1. 2D FFT Frekans Domain Analizi")
    print("  2. NASH Tespiti (HU + Morfometri)")
    print("  3. Hibrit Feature Fusion (Spatial + Frequency)")
    print("  4. XGBoost + SHAP Açıklanabilirlik")
    print("\n✅ Proje hazır!")
