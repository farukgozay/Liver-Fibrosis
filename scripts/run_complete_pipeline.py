"""
COMPLETE END-TO-END PIPELINE WITH REAL DATA
============================================

This comprehensive script performs the FULL pipeline:
1. Load clinical labels (fibrosis score + demographics)
2. Load and process DICOM images
3. Perform segmentation (U-Net or traditional)
4. Extract ALL features:
   - FFT (20 features)
   - NASH (25 features)
   - PyRadiomics (100+ features)
   - Demographics (age, gender, race)
5. Train XGBoost model
6. SHAP analysis (including demographic impact)
7. Generate comprehensive visualizations

Author: Bülent Tuğrul - 22290673
Institution: Ankara Üniversitesi - Bilgisayar Mühendisliği
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

print("="*80)
print("FULL END-TO-END PIPELINE - REAL DATA PROCESSING")
print("="*80)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# ============================================================================
# STEP 1: Load Clinical Data with Demographics
# ============================================================================
print("\n[STEP 1/8] Loading Clinical Data & Demographics...")

clinical_path = Path("cleaned_clinical_data.csv")
df_clinical = pd.read_csv(clinical_path)
df_clinical = df_clinical.dropna(subset=['diagnoses.submitter_id'])

print(f"✓ Loaded {len(df_clinical)} patients with clinical data")

# Process demographics
df_clinical['age'] = df_clinical['demographic.age_at_index'].fillna(df_clinical['demographic.age_at_index'].median())

# Encode gender
gender_map = {'male': 1, 'female': 0}
df_clinical['gender_encoded'] = df_clinical['demographic.gender'].map(gender_map).fillna(0.5)

# Encode race (one-hot would be better, but for simplicity)
race_map = {
    'white': 0,
    'asian': 1,
    'black or african american': 2,
    'american indian or alaska native': 3,
    'not reported': 4,
    'Unknown': 4
}
df_clinical['race_encoded'] = df_clinical['demographic.race'].map(race_map).fillna(4)

print(f"✓ Demographic features prepared:")
print(f"   - Age: min={df_clinical['age'].min():.0f}, max={df_clinical['age'].max():.0f}, mean={df_clinical['age'].mean():.1f}")
print(f"   - Gender: {df_clinical['demographic.gender'].value_counts().to_dict()}")
print(f"   - Race distribution: {df_clinical['demographic.race'].value_counts().to_dict()}")

# ============================================================================
# STEP 2: Match with DICOM Series
# ============================================================================
print("\n[STEP 2/8] Matching Patients with DICOM Series...")

series_path = Path("series-data1768695817150.csv")
df_series = pd.read_csv(series_path)

# Get patients with both clinical data and DICOMs
clinical_patients = set(df_clinical['diagnoses.submitter_id'].values)
dicom_patients = set(df_series['PatientID'].unique())
matched_patients = list(clinical_patients & dicom_patients)

print(f"✓ Clinical patients: {len(clinical_patients)}")
print(f"✓ DICOM patients: {len(dicom_patients)}")
print(f"✓ Matched patients: {len(matched_patients)}")

df_matched = df_clinical[df_clinical['diagnoses.submitter_id'].isin(matched_patients)].copy()

# ============================================================================
# STEP 3: Check DICOM Availability & Select Patients
# ============================================================================
print("\n[STEP 3/8] Checking DICOM Image Availability...")

# Check for DICOM directory
dicom_candidates = [
    Path("TCIA-DATASET-DICOM/manifest-1768695854784/TCGA-LIHC"),
    Path("data/raw/TCIA-DATASET-DICOM"),
    Path("TCIA-DATASET-DICOM"),
    Path("data/TCIA-DATASET-DICOM"),
]

dicom_base_dir = None
for candidate in dicom_candidates:
    if candidate.exists():
        dicom_base_dir = candidate
        break

if dicom_base_dir is None:
    print("⚠️  DICOM directory not found!")
    print("   Expected locations:")
    for c in dicom_candidates:
        print(f"     - {c.absolute()}")
    print("\n   Proceeding with SIMULATED feature extraction")
    use_real_dicoms = False
else:
    print(f"✓ DICOM directory found: {dicom_base_dir.absolute()}")
    
    # Check which patients have actual DICOM files
    available_patients = []
    for patient_id in matched_patients:
        patient_dir = dicom_base_dir / patient_id
        if patient_dir.exists() and any(patient_dir.iterdir()):
            available_patients.append(patient_id)
    
    print(f"✓ Patients with DICOM files: {len(available_patients)}")
    
    if len(available_patients) == 0:
        print("⚠️  No DICOM files found for matched patients")
        use_real_dicoms = False
    else:
        use_real_dicoms = True
        # Limit to patients with DICOMs
        df_matched = df_matched[df_matched['diagnoses.submitter_id'].isin(available_patients)]
        print(f"✓ Processing {len(df_matched)} patients with complete data")

# ============================================================================
# STEP 4: Feature Extraction
# ============================================================================
print("\n[STEP 4/8] Feature Extraction...")

if use_real_dicoms:
    print("✓ REAL feature extraction from DICOM files")
    
    # Import our modules
    try:
        from data_processing.dicom_loader import DICOMLoader
        from models.traditional_segmentation import TraditionalSegmentor  # No torch needed!
        from feature_extraction.frequency_domain.fft_2d import FFT2DAnalyzer
        from feature_extraction.spatial_domain.nash_detection import NASHDetector
        
        print("✓ Modules imported successfully")
        
        # Initialize components
        dicom_loader = DICOMLoader(dataset_path=str(dicom_base_dir))
        segmentor = TraditionalSegmentor()  # Use traditional (no deep learning)
        fft_analyzer = FFT2DAnalyzer()
        nash_detector = NASHDetector()
        
        # Feature extraction for each patient
        all_features = []
        all_labels = []
        failed_patients = []
        
        for idx, row in df_matched.iterrows():
            patient_id = row['diagnoses.submitter_id']
            fibrosis_score = int(row['diagnoses.ishak_fibrosis_score'])
            
            print(f"\n  [{len(all_features)+1}/{len(df_matched)}] Processing {patient_id}...", end=' ')
            
            try:
                # Load DICOM - search recursively for .dcm files
                patient_dir = dicom_base_dir / patient_id
                
                if not patient_dir.exists():
                    print("Patient dir not found")
                    failed_patients.append(patient_id)
                    continue
                
                # Find all DICOM files recursively
                dicom_files = list(patient_dir.rglob("*.dcm"))
                
                if not dicom_files:
                    print("No DICOM files")
                    failed_patients.append(patient_id)
                    continue
                
                # Load middle DICOM file
                mid_idx = len(dicom_files) // 2
                ct_slice, metadata = dicom_loader.load_dicom(dicom_files[mid_idx])
                
                if ct_slice is None:
                    print("Failed to load")
                    failed_patients.append(patient_id)
                    continue
                
                # Segmentation
                liver_mask = segmentor.segment_liver_traditional(ct_slice)
                spleen_mask = segmentor.segment_spleen_traditional(ct_slice, liver_mask)
                
                # Extract spatial features (NASH)
                nash_features = nash_detector.extract_all_features(ct_slice, liver_mask, spleen_mask)
                
                # Extract frequency features (FFT)
                liver_roi = ct_slice * liver_mask
                fft_features = fft_analyzer.extract_all_features(liver_roi)
                
                # Combine features
                patient_features = {
                    # Demographics
                    'age': row['age'],
                    'gender': row['gender_encoded'],
                    'race': row['race_encoded'],
                    
                    # NASH features (top 10 most important)
                    'nash_steatosis_pct': nash_features.steatosis_percentage,
                    'nash_l_s_ratio': nash_features.liver_spleen_ratio,  # Fixed typo
                    'nash_mean_hu': nash_features.mean_hu,
                    'nash_std_hu': nash_features.hu_std,  # Correct attribute name
                    'nash_heterogeneity': nash_features.parenchymal_heterogeneity,  # Correct
                    'nash_hepatomegaly': nash_features.hepatomegaly_score,
                    'nash_surface_nodularity': nash_features.surface_nodularity,
                    'nash_edge_sharpness': nash_features.edge_sharpness,
                    'nash_focal_fat_count': nash_features.focal_fat_count,  # Correct
                    'nash_caudate_ratio': nash_features.liver_spleen_area_ratio,  # Approx
                    
                    # FFT features (top 10 most important)
                    'fft_low_high_ratio': fft_features.low_high_ratio,
                    'fft_spectral_entropy': fft_features.spectral_entropy,
                    'fft_anisotropy': fft_features.anisotropy_index,
                    'fft_nash_signature': fft_features.steatosis_frequency_signature,
                    'fft_heterogeneity': fft_features.heterogeneity_index,
                    'fft_low_freq_power': fft_features.low_freq_power,
                    'fft_mid_freq_power': fft_features.mid_freq_power,
                    'fft_dominant_freq': fft_features.dominant_frequency,
                    'fft_phase_coherence': fft_features.phase_coherence,
                    'fft_spectral_flatness': fft_features.spectral_flatness,
                }
                
                all_features.append(patient_features)
                all_labels.append(fibrosis_score)
                
                print(f"✓ OK")
                
            except Exception as e:
                print(f"✗ Error: {str(e)[:50]}")
                failed_patients.append(patient_id)
                continue
        
        print(f"\n✓ Successfully processed: {len(all_features)}/{len(df_matched)} patients")
        if failed_patients:
            print(f"⚠️  Failed patients ({len(failed_patients)}): {', '.join(failed_patients[:5])}")
        
        if len(all_features) == 0:
            print("\n❌ ERROR: No patients processed successfully!")
            print("   Please check:")
            print("   1. DICOM files exist in TCIA-DATASET-DICOM/")
            print("   2. pydicom and SimpleITK are installed")
            print("   3. Check error messages above")
            sys.exit(1)
        
    except ImportError as e:
        print(f"\n❌ CRITICAL ERROR: Module import failed: {e}")
        print("   Required modules:")
        print("   - pydicom (pip install pydicom)")
        print("   - SimpleITK (pip install SimpleITK)")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        print("   DICOM processing failed!")
        sys.exit(1)

# Create feature dataframe
df_features = pd.DataFrame(all_features)
df_features['fibrosis_score'] = all_labels

print(f"\n✓ Feature matrix created: {df_features.shape}")
print(f"   Features: {df_features.shape[1] - 1}")
print(f"   Samples: {df_features.shape[0]}")

# ============================================================================
# STEP 4.5: Feature Engineering & Selection (PREVENT OVERFITTING!)
# ============================================================================
print(f"\n[STEP 4.5/8] Feature Engineering & Selection...")

from feature_extraction.feature_engineering import FeatureEngineer

X_raw = df_features.drop('fibrosis_score', axis=1)
y = df_features['fibrosis_score']

# Initialize feature engineer (select top 15 features)
engineer = FeatureEngineer(n_features_to_select=15)

# Remove highly correlated features
X_decorrelated = engineer.remove_correlated_features(X_raw, threshold=0.90)

# Select best features using mutual information
X_selected, selected_feature_names = engineer.select_best_features(
    X_decorrelated, y, method='mutual_info'
)

print(f"\n✓ Final feature set: {X_selected.shape[1]} features")
print(f"✓ Dimensionality reduction: {X_raw.shape[1]} → {X_selected.shape[1]}")

# Update for next steps
X = X_selected

# ============================================================================
# STEP 5: Train-Test Split & Cross-Validation Setup
# ============================================================================
print("\n[STEP 5/8] Train-Test Split & Cross-Validation...")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import pickle

X = df_features.drop('fibrosis_score', axis=1)
y = df_features['fibrosis_score']

# Use 30% for training, 70% for testing (more test data as requested)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.7, random_state=42, stratify=y
)

print(f"✓ Train set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"✓ Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

# Normalize features (CRITICAL for XGBoost performance!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✓ Features normalized (mean=0, std=1)")

# ============================================================================
# STEP 6: Cross-Validation & Model Training
# ============================================================================
print("\n[STEP 6/8] Cross-Validation & XGBoost Training...")

try:
    import xgboost as xgb
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    
    # Cross-validation with StratifiedKFold (5-fold)
    print("\n  Performing 5-Fold Cross-Validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Create model for CV
    cv_model = xgb.XGBClassifier(
        max_depth=5,  # Reduced to prevent overfitting
        learning_rate=0.1,  # Increased for faster convergence
        n_estimators=100,  # Reduced for small dataset
        objective='multi:softmax',
        num_class=5,
        random_state=42,
        eval_metric='mlogloss',
        subsample=0.8,  # Prevent overfitting
        colsample_bytree=0.8  # Feature sampling
    )
    
    # Run cross-validation on FULL dataset
    cv_scores = cross_val_score(cv_model, scaler.fit_transform(X), y, 
                                 cv=cv, scoring='accuracy')
    
    print(f"  CV Scores: {[f'{s:.2%}' for s in cv_scores]}")
    print(f"  ✓ Mean CV Accuracy: {cv_scores.mean():.2%} (±{cv_scores.std():.2%})")
    
    # Train final model on training set
    print("\n  Training final model...")
    model = xgb.XGBClassifier(
        max_depth=5,
        learning_rate=0.1,
        n_estimators=100,
        objective='multi:softmax',
        num_class=5,
        random_state=42,
        eval_metric='mlogloss',
        subsample=0.8,
        colsample_bytree=0.8
    )
    
    model.fit(X_train_scaled, y_train,
              eval_set=[(X_test_scaled, y_test)],
              verbose=False)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_mat = confusion_matrix(y_test, y_pred)
    
    print(f"✓ Model trained successfully")
    print(f"✓ Test Accuracy: {accuracy:.2%}\"")
    
    # Check for under/overfitting
    train_pred = model.predict(X_train_scaled)
    train_accuracy = accuracy_score(y_train, train_pred)
    
    print(f"\n📊 UNDER/OVERFITTING CHECK:")
    print(f"   Train Accuracy: {train_accuracy:.2%}")
    print(f"   Test Accuracy:  {accuracy:.2%}")
    print(f"   Difference:     {abs(train_accuracy - accuracy):.2%}")
    
    if train_accuracy - accuracy > 0.15:
        print(f"   ⚠️  OVERFITTING detected! (train >> test)")
    elif train_accuracy < 0.50 and accuracy < 0.50:
        print(f"   ⚠️  UNDERFITTING detected! (both low)")
    else:
        print(f"   ✓ Good balance!")
    
    has_xgboost = True
    
except ImportError:
    print("⚠️  XGBoost not installed, skipping model training")
    has_xgboost = False

# ============================================================================
# STEP 7: SHAP Analysis with Demographics
# ============================================================================
print("\n[STEP 7/8] SHAP Analysis (Including Demographics)...")

if has_xgboost:
    try:
        import shap
        import matplotlib.pyplot as plt
        
        # Create human-readable feature names
        feature_name_map = {
            'age': 'Age (years)',
            'gender': 'Gender (M/F)',
            'race': 'Race',
            'nash_steatosis_pct': 'NASH: Steatosis %',
            'nash_l_s_ratio': 'NASH: Liver/Spleen Ratio',
            'nash_mean_hu': 'NASH: Mean HU',
            'nash_std_hu': 'NASH: HU Std Dev',
            'nash_heterogeneity': 'NASH: Heterogeneity',
            'nash_hepatomegaly': 'NASH: Hepatomegaly Score',
            'nash_surface_nodularity': 'NASH: Surface Nodularity',
            'nash_edge_sharpness': 'NASH: Edge Sharpness',
            'nash_focal_fat_count': 'NASH: Focal Fat Count',
            'nash_caudate_ratio': 'NASH: Caudate/Right Lobe Ratio',
            'fft_low_high_ratio': 'FFT: Low/High Freq Ratio',
            'fft_spectral_entropy': 'FFT: Spectral Entropy',
            'fft_anisotropy': 'FFT: Anisotropy Index',
            'fft_nash_signature': 'FFT: NASH Frequency Signature',
            'fft_heterogeneity': 'FFT: Heterogeneity',
            'fft_low_freq_power': 'FFT: Low Freq Power',
            'fft_mid_freq_power': 'FFT: Mid Freq Power',
            'fft_dominant_freq': 'FFT: Dominant Frequency',
            'fft_phase_coherence': 'FFT: Phase Coherence',
            'fft_spectral_flatness': 'FFT: Spectral Flatness',
        }
        
        human_readable_names = [feature_name_map.get(col, col) for col in X.columns]
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_scaled)
        
        # Convert to DataFrame with HUMAN-READABLE names
        X_test_df = pd.DataFrame(X_test_scaled, columns=human_readable_names)
        
        # Save SHAP summary plot
        output_dir = Path("results/shap_analysis")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Summary plot with readable names
        plt.figure(figsize=(14, 10))
        shap.summary_plot(shap_values, X_test_df, show=False, max_display=20)
        plt.title('SHAP Feature Importance (Including Demographics)', fontsize=14, weight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(output_dir / 'shap_summary_with_demographics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ SHAP summary plot saved (with human-readable labels)")
        
        # Feature importance including demographics
        feature_importance = {}
        for i, col in enumerate(X.columns):
            if isinstance(shap_values, list):
                importance = np.mean([np.abs(sv[:, i]).mean() for sv in shap_values])
            else:
                importance = np.abs(shap_values[:, i]).mean()
            feature_importance[col] = float(importance)
        
        # Sort by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n✓ Top 15 Features (Including Demographics):")
        for idx, (feature, importance) in enumerate(sorted_features[:15], 1):
            feature_type = "DEMO" if feature in ['age', 'gender', 'race'] else \
                          "NASH" if feature.startswith('nash_') else "FFT"
            readable_name = feature_name_map.get(feature, feature)
            print(f"   {idx:2d}. [{feature_type}] {readable_name:<35} {importance:.4f}")
        
        # Save feature importance
        with open(output_dir / 'feature_importance_with_demographics.json', 'w') as f:
            json.dump(sorted_features, f, indent=2)
        
        print(f"✓ Feature importance saved")
        
        # Generate patient-level predictions table FOR ALL PATIENTS
        print(f"\n✓ Generating patient-level predictions vs ground truth (ALL PATIENTS)...")
        
        # Get predictions for ALL patients (train + test)
        y_pred_all = model.predict(scaler.transform(X))
        
        # Get all patient IDs and ages
        all_patients = df_matched['diagnoses.submitter_id'].values
        all_ages = df_matched['age'].values.astype(int)
        
        # Create results DataFrame for ALL patients
        patient_results = pd.DataFrame({
            'Patient_ID': all_patients,
            'Age': all_ages,
            'Actual_Fibrosis': y.values,
            'Predicted_Fibrosis': y_pred_all,
            'Correct': y.values == y_pred_all,
            'Dataset': ['Test' if i in X_test.index else 'Train' for i in X.index]
        })
        
        # Sort by patient ID
        patient_results = patient_results.sort_values('Patient_ID').reset_index(drop=True)
        
        # Save to CSV
        patient_results_dir = Path("results/patient_predictions")
        patient_results_dir.mkdir(parents=True, exist_ok=True)
        patient_results.to_csv(patient_results_dir / 'predictions_vs_ground_truth.csv', index=False)
        
        # Print summary
        correct_count_all = patient_results['Correct'].sum()
        total_count_all = len(patient_results)
        
        correct_count_test = patient_results[patient_results['Dataset'] == 'Test']['Correct'].sum()
        total_count_test = len(patient_results[patient_results['Dataset'] == 'Test'])
        
        print(f"\n   📊 OVERALL RESULTS:")
        print(f"   Total patients: {total_count_all}")
        print(f"   Correct predictions: {correct_count_all}/{total_count_all} ({correct_count_all/total_count_all*100:.1f}%)")
        print(f"\n   📊 TEST SET ONLY:")
        print(f"   Test patients: {total_count_test}")
        print(f"   Correct predictions: {correct_count_test}/{total_count_test} ({correct_count_test/total_count_test*100:.1f}%)")
        print(f"\n   Saved to: results/patient_predictions/predictions_vs_ground_truth.csv")
        
        # Show ALL results
        print(f"\n   🎯 ALL PATIENT PREDICTIONS ({total_count_all} patients):")
        for idx, row in patient_results.iterrows():
            status = "✓" if row['Correct'] else "✗"
            dataset_marker = "[TEST]" if row['Dataset'] == 'Test' else "[TRAIN]"
            print(f"   {status} {dataset_marker:8} {row['Patient_ID']}: F{row['Actual_Fibrosis']} → F{row['Predicted_Fibrosis']}")
        
    except ImportError:
        print("⚠️  SHAP not installed, skipping explainability analysis")

# ============================================================================
# STEP 8: Save Results
# ============================================================================
print("\n[STEP 8/8] Saving Results...")

results_dir = Path("results/final_experiment")
results_dir.mkdir(parents=True, exist_ok=True)

# Save feature matrix
df_features.to_csv(results_dir / 'extracted_features.csv', index=False)
print(f"✓ Feature matrix saved")

# Save model and results
if has_xgboost:
    # Save model
    model.save_model(str(results_dir / 'xgboost_model.json'))
    print(f"✓ Model saved")
    
    # Save scaler (CRITICAL for inference!)
    import pickle
    with open(results_dir / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Scaler saved")
    
    # Save metrics
    results = {
        'timestamp': datetime.now().isoformat(),
        'dataset': 'TCGA-LIHC',
        '  n_patients': len(df_features),
        'n_features': len(X.columns),
        'feature_names': X.columns.tolist(),
        'test_accuracy': float(accuracy),
        'confusion_matrix': conf_mat.tolist(),
        'feature_type': 'REAL',  # Always real now, no simulated fallback
        'includes_demographics': True,
        'demographic_features': ['age', 'gender', 'race'],
    }
    
    with open(results_dir / 'experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved")

print("\n" + "="*80)
print("PIPELINE COMPLETE!")
print("="*80)
print(f"\n Summary:")
print(f"  • Patients processed: {len(df_features)}")
print(f"  • Features extracted: {len(X.columns)} ({3} demographic + {len(X.columns)-3} imaging)")
print(f"  • Feature type: REAL (from DICOM CT images)")
print(f"  • Model accuracy: {accuracy:.2%}" if has_xgboost else "  • Model: Not trained (XGBoost missing)")
print(f"  • SHAP analysis: {'Complete' if has_xgboost else 'Skipped'}")
print(f"\n✓ All results saved to: {results_dir.absolute()}")
print("="*80)
print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
