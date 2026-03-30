"""
Single Patient Test Script
===========================

Test the pipeline on a single patient by ID.
Useful for debugging and demonstration.
"""

import sys
import argparse
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_processing.dicom_loader import DICOMLoader
from feature_extraction.frequency_domain.fft_2d import FFT2DAnalyzer
from feature_extraction.spatial_domain.nash_detection import NASHDetector

def test_single_patient(patient_id: str):
    """Test pipeline on single patient"""
    
    print("="*80)
    print(f"SINGLE PATIENT TEST: {patient_id}")
    print("="*80)
    
    # Find DICOM directory
    dicom_candidates = [
        Path("TCIA-DATASET-DICOM/manifest-1768695854784/TCGA-LIHC"),
        Path("data/raw/TCIA-DATASET-DICOM"),
    ]
    
    dicom_dir = None
    for candidate in dicom_candidates:
        if candidate.exists():
            dicom_dir = candidate
            break
    
    if not dicom_dir:
        print("❌ DICOM directory not found!")
        return
    
    patient_dir = dicom_dir / patient_id
    
    if not patient_dir.exists():
        print(f"❌ Patient directory not found: {patient_dir}")
        print(f"\nAvailable patients:")
        for p in list(dicom_dir.glob("TCGA-*"))[:10]:
            print(f"  - {p.name}")
        return
    
    print(f"✓ Found patient directory: {patient_dir}\n")
    
    # Load DICOM
    print("[1/4] Loading DICOM...")
    try:
        # Find series directories
        series_dirs = [d for d in patient_dir.iterdir() if d.is_dir()]
        
        if not series_dirs:
            print("❌ No series found")
            return
        
        # Load DICOMs from first series (search recursively)
        series_dir = series_dirs[0]
        dicom_files = list(series_dir.rglob("*.dcm"))  # Recursive search
        
        if not dicom_files:
            print("❌ No DICOM files in series")
            print(f"   Searched in: {series_dir}")
            return
        
        print(f"✓ Found {len(dicom_files)} DICOM files in series")
        
        # Load middle slice
        mid_idx = len(dicom_files) // 2
        loader = DICOMLoader(dataset_path=str(dicom_dir))
        ct_slice, metadata = loader.load_dicom(dicom_files[mid_idx])
        
        print(f"✓ Loaded slice {mid_idx}/{len(dicom_files)}")
        print(f"✓ Image shape: {ct_slice.shape}")
        print(f"✓ HU range: [{ct_slice.min():.1f}, {ct_slice.max():.1f}]")
        
    except Exception as e:
        print(f"❌ DICOM loading failed: {e}")
        return
    
    # FFT Analysis
    print("\n[2/4] FFT Analysis...")
    try:
        fft = FFT2DAnalyzer()
        fft_features = fft.extract_all_features(ct_slice)
        
        print(f"✓ FFT features extracted")
        print(f"  Low/High Ratio: {fft_features.low_high_ratio:.4f}")
        print(f"  Spectral Entropy: {fft_features.spectral_entropy:.4f}")
        print(f"  Anisotropy: {fft_features.anisotropy_index:.4f}")
        
    except Exception as e:
        print(f"❌ FFT failed: {e}")
        fft_features = None
    
    # NASH Detection
    print("\n[3/4] NASH Detection...")
    try:
        nash = NASHDetector()
        
        # Simple liver mask (threshold-based)
        liver_mask = (ct_slice > 0) & (ct_slice < 200)
        
        nash_features = nash.extract_all_features(ct_slice, liver_mask, liver_mask)
        
        print(f"✓ NASH features extracted")
        print(f"  Steatosis: {nash_features.steatosis_percentage:.1f}%")
        print(f"  L/S Ratio: {nash_features.liver_spleen_ratio:.2f}")
        print(f"  Mean HU: {nash_features.mean_hu:.1f}")
        
    except Exception as e:
        print(f"❌ NASH failed: {e}")
        nash_features = None
    
    # Summary
    print("\n[4/4] Summary & Prediction...")
    print("="*80)
    print(f"Patient: {patient_id}")
    
    # Load ground truth if available
    actual_score = None
    try:
        import pandas as pd
        clinical_path = Path("cleaned_clinical_data.csv")
        if clinical_path.exists():
            df_clinical = pd.read_csv(clinical_path)
            df_clinical = df_clinical.dropna(subset=['diagnoses.submitter_id'])
            
            # Find patient
            patient_row = df_clinical[df_clinical['diagnoses.submitter_id'] == patient_id]
            
            if not patient_row.empty:
                actual_score = int(patient_row.iloc[0]['diagnoses.ishak_fibrosis_score'])
                age = int(patient_row.iloc[0]['demographic.age_at_index']) if pd.notna(patient_row.iloc[0]['demographic.age_at_index']) else 'N/A'
                gender = patient_row.iloc[0]['demographic.gender']
                race = patient_row.iloc[0]['demographic.race']
                
                print(f"\n📋 CLINICAL DATA:")
                print(f"  Age: {age}")
                print(f"  Gender: {gender}")
                print(f"  Race: {race}")
            else:
                print(f"\n⚠️  No clinical data found for {patient_id}")
        else:
            print(f"\n⚠️  Clinical data file not found")
    except Exception as e:
        print(f"\n⚠️  Could not load clinical data: {e}")
    
    print(f"\n📊 FEATURE EXTRACTION:")
    print(f"DICOM: ✓ Loaded")
    print(f"FFT: {'✓' if fft_features else '✗'}")
    print(f"NASH: {'✓' if nash_features else '✗'}")
    
    # Model prediction
    print(f"\n🤖 MODEL PREDICTION:")
    try:
        if fft_features and nash_features and actual_score is not None:
            # Load model if exists
            model_path = Path("results/final_experiment/xgboost_model.json")
            
            if model_path.exists():
                import xgboost as xgb
                from sklearn.preprocessing import StandardScaler
                import pickle
                
                # Load model
                model = xgb.XGBClassifier()
                model.load_model(str(model_path))
                
                # Load scaler if exists
                scaler_path = Path("results/final_experiment/scaler.pkl")
                if scaler_path.exists():
                    with open(scaler_path, 'rb') as f:
                        scaler = pickle.load(f)
                else:
                    scaler = StandardScaler()
                
                # Prepare features (matching training order)
                features = np.array([[
                    age if age != 'N/A' else 60,  # age
                    1 if gender == 'male' else 0,  # gender
                    0,  # race (simplified)
                    nash_features.steatosis_percentage,
                    nash_features.liver_spleen_ratio,
                    nash_features.mean_hu,
                    nash_features.hu_std,  # Correct attribute name
                    nash_features.parenchymal_heterogeneity,  # heterogeneity_index
                    nash_features.hepatomegaly_score,
                    nash_features.surface_nodularity,
                    nash_features.edge_sharpness,
                    nash_features.focal_fat_count,  # focal_fat_deposition
                    nash_features.liver_spleen_area_ratio,  # caudate_right_lobe_ratio approximation
                    fft_features.low_high_ratio,
                    fft_features.spectral_entropy,
                    fft_features.anisotropy_index,
                    fft_features.steatosis_frequency_signature,
                    fft_features.heterogeneity_index,
                    fft_features.low_freq_power,
                    fft_features.mid_freq_power,
                    fft_features.dominant_frequency,
                    fft_features.phase_coherence,
                    fft_features.spectral_flatness,
                ]])
                
                # Normalize (if scaler was loaded, use it; otherwise just use raw)
                try:
                    features_scaled = scaler.transform(features)
                except:
                    features_scaled = features
                
                # Predict
                prediction = model.predict(features_scaled)[0]
                proba = model.predict_proba(features_scaled)[0]
                
                # Display results
                fibrosis_names = ['No fibrosis', 'Mild fibrosis', 'Moderate fibrosis', 'Advanced fibrosis', 'Cirrhosis']
                
                print(f"  ⭐ ACTUAL:    F{actual_score} ({fibrosis_names[actual_score]})")
                print(f"  🎯 PREDICTED: F{int(prediction)} ({fibrosis_names[int(prediction)]})")
                print(f"  {'  ✓ CORRECT!' if int(prediction) == actual_score else '  ✗ INCORRECT'}")
                
                print(f"\n  Class Probabilities:")
                for i, prob in enumerate(proba):
                    bar = '█' * int(prob * 20)
                    print(f"    F{i}: {prob:5.1%} {bar}")
                    
            else:
                print(f"  ⚠️  No trained model found")
                print(f"     Run: python scripts\\run_complete_pipeline.py")
                if actual_score is not None:
                    print(f"\n  ⭐ ACTUAL: F{actual_score} ({fibrosis_names[actual_score]})")
        else:
            if actual_score is not None:
                fibrosis_names = ['No fibrosis', 'Mild fibrosis', 'Moderate fibrosis', 'Advanced fibrosis', 'Cirrhosis']
                print(f"  ⭐ ACTUAL: F{actual_score} ({fibrosis_names[actual_score]})")
            print(f"  ⚠️  Prediction skipped (incomplete features or no ground truth)")
            
    except Exception as e:
        print(f"  ⚠️  Prediction failed: {e}")
        if actual_score is not None:
            fibrosis_names = ['No fibrosis', 'Mild fibrosis', 'Moderate fibrosis', 'Advanced fibrosis', 'Cirrhosis']
            print(f"  ⭐ ACTUAL: F{actual_score} ({fibrosis_names[actual_score]})")
    
    print("="*80)
    
    print("\n✅ Test complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test single patient')
    parser.add_argument('--patient', type=str, required=True,
                       help='Patient ID (e.g., TCGA-DD-A114)')
    
    args = parser.parse_args()
    
    test_single_patient(args.patient)
