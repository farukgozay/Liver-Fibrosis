# 🏥 Karaciğer Fibrozunun BT Görüntülerinden Non-İnvaziv Evrelendirilmesi

## 📋 Proje Özeti

Bu bitirme projesi, **NASH (Non-Alcoholic Steatohepatitis)** hastalarında karaciğer fibrozunun evrelendirilmesi için **2D FFT (Fast Fourier Transform)** tabanlı frekans domain analizi ve derin öğrenme yöntemlerini kullanmaktadır.

### 🎯 Temel Hedefler

1. **NASH Tespiti**: Non-alkolik steatohepatit hastalığının BT görüntülerinden otomatik tespiti
2. **Fibroz Evrelemesi**: F0-F4 arası fibroz evrelerinin sınıflandırılması (METAVIR/Ishak skorlama sistemi)
3. **Frekans Domain Analizi**: 2D FFT kullanarak doku dokusunun frekans uzayında karakterizasyonu
4. **Açıklanabilir AI**: SHAP değerleri ile model kararlarının yorumlanması

### 🔬 Yenilikçi Yaklaşım: 2D FFT + Uzamsal Domain Hibrit Analiz

#### **Frekans Domain (Frequency Domain) Analizi**
- **2D FFT Uygulaması**: BT görüntülerinin frekans uzayına dönüştürülmesi
- **Texture Periodicity**: Fibroz dokusunun periyodik yapılarının tespiti
- **Spektral Güç Analizi**: Düşük ve yüksek frekanslı bileşenlerin ayrıştırılması
- **Gabor Filtreleme**: Yönlü doku özelliklerinin çıkarılması

#### **Uzamsal Domain (Spatial Domain) Analizi**
- **HU (Hounsfield Unit) Değerleri**: Doku yoğunluğu ölçümleri
- **Texture Features**: GLCM, GLRLM, GLSZM matrisleri
- **Morphological Features**: Karaciğer ve dalak morfolojik özellikleri
- **Radiomics**: Şekil, yoğunluk ve doku özellikleri

### 📊 Dataset: TCIA-LIHC

- **Kaynak**: The Cancer Imaging Archive - Liver Hepatocellular Carcinoma
- **Veri Sayısı**: 3,344 DICOM dosyası
- **Hasta Sayısı**: ~100 hasta
- **Modalite**: Kontrastlı ve kontrastsız abdomen BT
- **Fazlar**: Portal venous, arterial, delayed phases

### 🧠 Model Mimarisi

```
Input: CT DICOM Images
  ↓
┌─────────────────────────────────────┐
│  PREPROCESSING PIPELINE             │
│  - DICOM Loading & Normalization    │
│  - ROI Segmentation (Liver/Spleen)  │
│  - Window-Level Adjustment           │
└─────────────────────────────────────┘
  ↓
┌─────────────────┬───────────────────┐
│ SPATIAL DOMAIN  │ FREQUENCY DOMAIN  │
│                 │                   │
│ • Radiomics     │ • 2D FFT         │
│ • HU Values     │ • Power Spectrum │
│ • Texture GLCM  │ • Gabor Filters  │
│ • Morphology    │ • Wavelet Trans. │
└─────────────────┴───────────────────┘
  ↓
┌─────────────────────────────────────┐
│  FEATURE FUSION                     │
│  Spatial + Frequency Features       │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│  CLASSIFICATION MODELS              │
│  • XGBoost (Primary)                │
│  • ResNet-50 (Deep Learning)        │
│  • Ensemble Methods                 │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│  EXPLAINABILITY (XAI)               │
│  • SHAP Values                      │
│  • Attention Maps                   │
│  • Feature Importance               │
└─────────────────────────────────────┘
  ↓
Output: NASH Detection + Fibrosis Stage (F0-F4)
```

### 📁 Proje Yapısı

```
bitirme/
├── data/
│   ├── raw/                          # TCIA DICOM files
│   ├── processed/                    # Preprocessed images
│   │   ├── spatial/                  # Spatial domain images
│   │   └── frequency/                # FFT transformed images
│   ├── segmentations/                # Liver/spleen masks
│   └── metadata/                     # Clinical data & labels
│
├── src/
│   ├── data_processing/
│   │   ├── dicom_loader.py          # DICOM reading & parsing
│   │   ├── preprocessing.py          # Normalization, windowing
│   │   ├── segmentation.py           # Auto liver/spleen segmentation
│   │   └── data_augmentation.py      # Augmentation strategies
│   │
│   ├── feature_extraction/
│   │   ├── frequency_domain/
│   │   │   ├── fft_2d.py            # 2D FFT implementation
│   │   │   ├── power_spectrum.py     # Spectral analysis
│   │   │   ├── gabor_filters.py      # Gabor feature extraction
│   │   │   └── wavelet_transform.py  # Wavelet decomposition
│   │   │
│   │   ├── spatial_domain/
│   │   │   ├── radiomics_features.py # PyRadiomics integration
│   │   │   ├── texture_glcm.py       # GLCM features
│   │   │   ├── texture_glrlm.py      # Run-length matrix
│   │   │   ├── texture_glszm.py      # Size-zone matrix
│   │   │   ├── hu_statistics.py      # HU value analysis
│   │   │   └── morphological.py      # Shape features
│   │   │
│   │   └── feature_fusion.py         # Spatial + Frequency fusion
│   │
│   ├── models/
│   │   ├── classical_ml/
│   │   │   ├── xgboost_model.py     # XGBoost classifier
│   │   │   ├── random_forest.py      # RF classifier
│   │   │   └── svm_classifier.py     # SVM classifier
│   │   │
│   │   ├── deep_learning/
│   │   │   ├── resnet_fibrosis.py   # ResNet-50 fine-tuned
│   │   │   ├── dual_stream_cnn.py    # Spatial+Freq CNN
│   │   │   ├── attention_unet.py     # Segmentation model
│   │   │   └── custom_cnn.py         # Custom architecture
│   │   │
│   │   └── ensemble/
│   │       ├── voting_classifier.py  # Ensemble voting
│   │       └── stacking_model.py     # Stacking ensemble
│   │
│   ├── explainability/
│   │   ├── shap_analysis.py         # SHAP value computation
│   │   ├── lime_explainer.py         # LIME explanations
│   │   ├── grad_cam.py               # Attention visualization
│   │   └── feature_importance.py     # Feature ranking
│   │
│   ├── evaluation/
│   │   ├── metrics.py                # Accuracy, AUC, F1, etc.
│   │   ├── confusion_matrix.py       # CM visualization
│   │   ├── roc_curves.py             # ROC/AUC analysis
│   │   └── statistical_tests.py      # Significance testing
│   │
│   └── utils/
│       ├── config.py                 # Configuration management
│       ├── logger.py                 # Logging utilities
│       ├── visualization.py          # Plotting functions
│       └── data_loader.py            # Data loading utilities
│
├── notebooks/
│   ├── 01_data_exploration.ipynb     # EDA
│   ├── 02_fft_analysis.ipynb         # FFT visualization
│   ├── 03_feature_engineering.ipynb  # Feature extraction
│   ├── 04_model_training.ipynb       # Model training
│   └── 05_results_analysis.ipynb     # Results & viz
│
├── experiments/
│   ├── baseline_models/              # Baseline experiments
│   ├── fft_experiments/              # FFT-based experiments
│   └── ensemble_experiments/         # Ensemble experiments
│
├── models/                           # Saved trained models
├── results/                          # Experiment results
│   ├── metrics/                      # Performance metrics
│   ├── visualizations/               # Plots & figures
│   └── predictions/                  # Model predictions
│
├── docs/
│   ├── tez_raporu.pdf               # Final thesis report
│   ├── sunum.pptx                    # Presentation
│   └── literature_review.md          # Literature review
│
├── requirements.txt                  # Python dependencies
├── environment.yml                   # Conda environment
├── setup.py                          # Package setup
└── README.md                         # This file
```

### 🛠️ Teknolojiler

#### Core Libraries
- **Python 3.10+**
- **PyTorch / TensorFlow**: Deep learning framework
- **NumPy & SciPy**: FFT ve matematiksel işlemler
- **OpenCV**: Görüntü işleme
- **SimpleITK / PyDICOM**: DICOM handling

#### Feature Extraction
- **PyRadiomics**: Radiomics feature extraction
- **Scikit-image**: Image processing & texture analysis
- **Pywavelets**: Wavelet transformations

#### Machine Learning
- **XGBoost**: Gradient boosting classifier
- **Scikit-learn**: Classical ML algorithms
- **Optuna**: Hyperparameter optimization

#### Explainability (XAI)
- **SHAP**: SHapley Additive exPlanations
- **LIME**: Local Interpretable Model-agnostic Explanations
- **Grad-CAM**: Gradient-weighted Class Activation Mapping

#### Visualization
- **Matplotlib & Seaborn**: Plotting
- **Plotly**: Interactive visualizations
- **TensorBoard**: Training monitoring

### 🔑 Temel Özellikler

#### 1. **NASH Tespiti için Özel Özellikler**
- **Steatosis Detection**: Yağlanma oranının tespiti (HU < 40)
- **Hepatomegaly**: Karaciğer büyümesi analizi
- **Spleen Analysis**: Dalak boyutu ve tekstürü
- **Vascular Changes**: Portal venöz basınç göstergeleri

#### 2. **FFT Tabanlı Fibroz Karakterizasyonu**
- **Low-Frequency Components**: Genel doku yapısı
- **High-Frequency Components**: İnce doku detayları ve fibrotik bantlar
- **Frequency Ratio**: Düşük/yüksek frekans oranı
- **Directional Spectrum**: Yönlü doku paternleri

#### 3. **Multi-Phase CT Analizi**
- **Arterial Phase**: Erken enhancement patterns
- **Portal Venous Phase**: Optimal karaciğer-lezyon kontrast
- **Delayed Phase**: Geç fibroz enhancement

### 📈 Beklenen Sonuçlar

- **NASH Tespiti Doğruluğu**: > 90%
- **Fibroz Evreleme AUC**: > 0.85 (F0 vs F1-F4, F0-F2 vs F3-F4)
- **Önemli Fibroz (≥F2) Tespiti**: Sensitivity > 85%, Specificity > 80%
- **İleri Fibroz (≥F3) Tespiti**: Sensitivity > 80%, Specificity > 85%

### 📚 Referans Literatür

1. **FFT in Medical Imaging**: "Texture Analysis of Liver Fibrosis Using FFT"
2. **NASH Detection**: "AI-based NASH Diagnosis from CT Imaging"
3. **Radiomics**: "Radiomics-based Liver Fibrosis Staging"
4. **TCIA Dataset**: TCGA-LIHC collection documentation

### 👨‍💻 Geliştirici

**Bülent Tuğrul**  
Ankara Üniversitesi  
Bilgisayar Mühendisliği Bölümü  
Öğrenci No: 22290673

### 📝 Lisans

Bu proje akademik amaçlı geliştirilmiştir.

---

## 🚀 Hızlı Başlangıç

```bash
# Environment kurulumu
conda env create -f environment.yml
conda activate liver-fibrosis

# Veri indirme (TCIA)
python scripts/download_tcia_data.py

# Preprocessing
python src/data_processing/preprocess_pipeline.py

# Feature extraction
python src/feature_extraction/extract_all_features.py

# Model training
python src/models/train_xgboost.py

# Evaluation
python src/evaluation/evaluate_models.py
```

## 📊 Sonuç

Bu proje, **2D FFT tabanlı frekans domain analizi** ile **klasik uzamsal domain radiomics**'i birleştirerek NASH hastalarında karaciğer fibrozunun non-invaziv evrelendirilmesi için yenilikçi bir yaklaşım sunmaktadır.
