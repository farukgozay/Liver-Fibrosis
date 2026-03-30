# SOFTWARE REQUIREMENTS SPECIFICATION (SRS)

## Karaciğer Fibrozunun BT Görüntülerinden Non-İnvaziv Evrelendirilmesi

**Öğrenci**: Bülent Tuğrul (22290673)  
**Kurum**: Ankara Üniversitesi - Bilgisayar Mühendisliği Bölümü  
**Standart**: IEEE 830-1998 Recommended Practice for Software Requirements Specifications  
**Tarih**: Ocak 2026  
**Versiyon**: 1.0

---

## 1. GİRİŞ

### 1.1. Araştırmanın Amacı ve Önemi

Bu proje, karaciğer fibrozunun evrelendirilmesi işlemini basitleştirmeyi ve invaziv biyopsinin risklerini ortadan kaldırmayı amaçlamaktadır. Karaciğer biyopsisi:
- **Enfeksiyon ve kanama riski** taşır
- **Hasta konforunu düşürür**
- Sadece **küçük bir doku parçasını** gösterir
- **Örnekleme hatası** riski vardır

Sistemimiz, **Bilgisayarlı Tomografi (BT)** görüntülerini işleyerek fibroz evresini (F0'dan F4'e kadar) yüksek doğrulukla tahmin eden, **girişimsel olmayan** ve **yapay zeka destekli** bir karar destek sistemi sunmaktadır.

**Bilimsel Katkı**: Karaciğer yağlanması (steatosis) ve fibroz aynı anda görülen **Non-Alcoholic Steatohepatitis (NASH)** durumlarında, uzamsal alandaki özellikler maskeleme etkisi yaratır. Yani:
- **Fibroz**: Hipodens (düşük yoğunluk)
- **Yağlanma**: Hiperdens (yüksek yoğunluk)
- **Maskeleme Etkisi**: Birbirlerinin sinyalini sönümler → Hasarlı doku sağlam görünür

Bu sorunu çözmek için, **Uzamsal Domain (Spatial Domain)** ve **Frekans Domain (2D-FFT)** özelliklerini birleştiren **hibrit yaklaşım** sunmaktayız. Bu, literatürdeki önemli bir boşluğu doldurmaktadır.

### 1.2. Araştırma Projesinin Kapsamı

Proje, **TCGA-LIHC** veri setinden elde edilen abdominal BT görüntülerinin işlenmesini, ilgili klinik verilerle birl eştirilmesini ve makine öğrenmesi algoritmalarıyla sınıflandırılmasını kapsar.

**Kapsam dahilindeki iş paketleri**:

1. **Veri Ön İşleme**:
   - DICOM formatındaki görüntülerin okunması
   - NIFTI formatına dönüştürme
   - Normalizasyon ve pencere/seviye ayarlaması

2. **Segmentasyon**:
   - U-Net modeli ile otomatik karaciğer/dalak segmentasyonu
   - Gerektiğinde yarı-otomatik düzeltme

3. **Öznitelik Çıkarımı**:
   - **Uzamsal Domain**:
     - PyRadiomics ile radyomik özellikler (GLCM, GLRLM, GLSZM, GLDM, NGTDM)
     - NASH-specific özellikler (HU statistics, morfometri)
     - First-order statistics
   - **Frekans Domain**:
     - 2D-FFT ile frekans spektrumu analizi
     - Düşük/Orta/Yüksek frekans band özellikleri
     - Yönlü (directional) frekans özellikleri
     - NASH frekans imzası

4. **Model Eğitimi**:
   - XGBoost ile sınıflandırma
   - Optuna ile hiperparametre optimizasyonu
   - K-katlı çapraz doğrulama

5. **Açıklanabilirlik (XAI)**:
   - SHAP (Shapley Additive Explanations) analizi
   - Feature importance ranking
   - Model karar süreçlerinin görselleştirilmesi

### 1.3. Tanımlar ve Kısaltmalar

| Kısaltma | Açıklama |
|----------|----------|
| **BT (CT)** | Bilgisayarlı Tomografi (Computed Tomography) |
| **NASH** | Non-Alcoholic Steatohepatitis - Alkol almadan karaciğer yağlanması ve iltihabı |
| **ROI** | Region of Interest - Görüntü üzerinde ilgilenilen bölge (karaciğer dokusu) |
| **HU** | Hounsfield Unit - BT görüntülerindeki radyodansite (yoğunluk) birimi |
| **2D-FFT** | İki Boyutlu Hızlı Fourier Dönüşümü (2D Fast Fourier Transform) |
| **XAI** | Açıklanabilir Yapay Zeka (Explainable AI) |
| **DICOM** | Digital Imaging and Communications in Medicine |
| **NIFTI** | Neuroimaging Informatics Technology Initiative (tıbbi görüntü formatı) |
| **İnvaziv** | Vücut bütünlüğüne müdahale gerektiren (örn: iğne biyopsisi) |
| **GLCM** | Gray Level Co-occurrence Matrix |
| **GLRLM** | Gray Level Run Length Matrix |
| **GLSZM** | Gray Level Size Zone Matrix |
| **METAVIR** | Fibroz evreleme sistemi (F0-F4) |
| **SHAP** | SHapley Additive exPlanations |

### 1.4. Belge Yapısına Genel Bakış

Bu belge, projenin teknik gereksinimlerini, sistem mimarisini ve beklenen çıktılarını tanımlar:

- **Bölüm 2**: Genel proje tanımı, araştırma geçmişi ve problem cümlesi
- **Bölüm 3**: İşlevsel ve sistem gereksinimleri
- **Bölüm 4**: Sistem mimarisi ve bileşenler
- **Bölüm 5**: Doğrulama ve test kriterleri
- **Bölüm 6**: Beklenen çıktılar ve performans metrikleri

---

## 2. GENEL PROJE TANIMI

### 2.1. Araştırma Geçmişi ve Bağlamı

Mevcut literatürde karaciğer fibrozu tespiti çalışmaları çoğunlukla **uzamsal özelliklere** odaklanmaktadır:
- Doku tekstürü (GLCM, GLRLM)
- Yoğunluk istatistikleri (HU değerleri)
- Morfometrik özellikler

**Ancak hastada fibroz ve yağlanma (NASH) aynı anda bulunuyorsa tespit zorlaşmaktadır:**

```
Fibroz     →  Hipodens (düşük HU)
Yağlanma   →  Hiperdens (yüksek HU)
─────────────────────────────────
Maskeleme  →  Sinyaller birbirini sönümler
Sonuç      →  Hasarlı doku sağlam görünür
```

**Literatür Boşluğu**: Standart uzamsal metotlar maskeleme etkisini yeterince ele almıyor.

**Çözümümüz**: Frekans domain işleme (2D-FFT) sayesinde:
- Periyodik doku yapıları tespit edilir
- Fibroz bantları frekans spektrumunda görünür
- NASH-related değişiklikler frekans imzası oluşturur
- Maskeleme etkisi aşılır

### 2.2. Problem Cümlesi ve Araştırma Soruları

#### Problem Cümlesi

İnvaziv biyopsi **riskli** ve **hasta konforu düşük** bir yöntemdir. Mevcut görüntü işleme teknikleri **NASH gibi karmaşık hastalıklarda yetersiz**dir, çünkü yağlanma ve fibroz birbirini maskeler.

#### Araştırma Soruları

**AS1**: BT görüntülerinin frekans bileşenleri (2D-FFT), uzamsal maskeleme etkisine rağmen, fibrozun yapısal bozulmalarını ayırt edebilir mi?

**AS2**: Hibrit öznitelik seti (uzamsal + frekans), tek başına uzamsal özelliklere göre sınıflandırma başarısını ne kadar artırmaktadır?

**AS3**: NASH tespiti için HU-tabanlı steatosis analizi ve frekans domain imzası birlikte kullanıldığında, tanı doğruluğu ne düzeydedir?

### 2.3. Üst Düzey Araştırma Metodolojisi

Proje, **sayısal araştırma** ve **test amaçlı yazılım geliştirme** sürecini izler:

```
┌─────────────────────────────────────────────────────────────┐
│  1. VERİ HAZIRLAMA                                          │
│     • TCGA-LIHC veri setinden hasta seçimi                  │
│     • DICOM → NIFTI dönüşüm                                 │
│     • Metadata extraction                                   │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  2. SEGMENTASYON                                            │
│     • U-Net otomatik segmentasyon                           │
│     • Liver ROI extraction                                  │
│     • Spleen ROI extraction (L/S ratio için)                │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  3. ÖZNİTELİK MÜHENDİSLİĞİ (Feature Engineering)           │
│                                                             │
│  ┌────────────────────┐        ┌────────────────────┐      │
│  │ UZAMSAL DOMAIN     │        │ FREKANS DOMAIN     │      │
│  │                    │        │                    │      │
│  │ • PyRadiomics      │        │ • 2D-FFT           │      │
│  │ • NASH features    │        │ • Power spectrum   │      │
│  │ • HU statistics    │        │ • Frequency bands  │      │
│  │ • Morphometry      │        │ • Directional feat.│      │
│  └────────────────────┘        └────────────────────┘      │
│           ↓                             ↓                   │
│           └──────────── FUSION ─────────┘                   │
│                    (100+ features)                          │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  4. MODELLEME                                               │
│     • XGBoost classifier                                    │
│     • Hyperparameter optimization (Optuna)                  │
│     • Multi-class: F0 vs F1 vs F2 vs F3 vs F4              │
│     • Binary: Significant/Advanced fibrosis                 │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  5. VALİDASYON & AÇIKLANAB İLİRLİK                          │
│     • K-fold cross validation                               │
│     • SHAP analysis                                         │
│     • Metrics: Accuracy, AUC, Sensitivity, Specificity      │
└─────────────────────────────────────────────────────────────┘
```

### 2.4. Proje Kısıtları ve Varsayımları

#### Kısıtlar

1. **Veri Kısıtı**: TCGA-LIHC açık veri seti kullanılacaktır. Hasta gizliliği nedeniyle sınırlı veri mevcuttur.

2. **Etiket Kısıtı**: Ground truth fibroz evre bilgisi olmayan hastalar çalışma dışı bırakılacaktır. Yetersiz veri durumunda fibroz sınıfları gruplanabilir (örn: Düşük Risk [F0-F2] vs Yüksek Risk [F3-F4]).

3. **Hesaplama Kaynakları**: Model eğitimi standart PC donanımıyla gerçekleştirilecektir (GPU opsiyonel).

4. **Segmentasyon**: Karaciğer segmentasyonu U-Net ile otomatik yapılacak, ancak düşük kaliteli görüntülerde manuel düzeltme gerekebilir.

#### Varsayımlar

1. **Veri Doğruluğu**: TCGA veri setindeki patoloji raporları ve evreleme bilgileri doğru kabul edilir.

2. **BT Kalitesi**: TCIA-LIHC görüntüleri klinik standartlara uygun kalitededir.

3. **Fibroz-FFT İlişkisi**: Fibroz dokusunun yapısal değişikliklerinin frekans domeninde ayırt edilebilir imzalar oluşturduğu varsayılır.

4. **Tekrarlanabilirlik**: PyRadiomics ve FFT hesaplamaları deterministik ve tekrarlanabilirdir.

### 2.5. Beklenen Katkılar

#### Bilimsel Katkılar

1. **Yenilikçi Hibrit Yaklaşım**: Uzamsal ve frekans domain özelliklerinin birleştirilmesi

2. **NASH-Fibroz Maskeleme Çözümü**: Frekans domain analizi ile maskeleme etkisinin aşılması

3. **Açıklanabilir AI**: SHAP ile model kararlarının klinik olarak yorumlanabilir hale getirilmesi

4. **Non-invaziv Tanı**: Biyopsi ihtiyacını azaltma potansiyeli

#### Pratik Katkılar

1. **Karar Destek Sistemi**: Radyologlara yardımcı olacak otomatik tanı aracı

2. **Tekrarlanabilir Workflow**: Açık kaynak araçlarla (PyRadiomics, scikit-learn,  XGBoost) kurulabilir sistem

3. **Hızlı Analiz**: Hasta başına dakikalar içinde sonuç

4. **Maliyet Azaltımı**: İnvaz iv işlem maliyetlerinin düşürülmesi

---

## 3. İŞLEVSEL VE SİSTEM GEREKSİNİMLERİ

### 3.1. Sistem Açıklaması

Sistem, ham BT görüntülerini girdi olarak kabul eden ve arka planda işleyerek fibroz evresini sınıflandıran bir mekanizmaya sahiptir.

**Girdi**: DICOM CT görüntüleri  
**İşlem**: Segmentasyon → Feature Extraction → Classification → Explanation  
**Çıktı**: Fibroz evresi + SHAP açıklama grafikleri

### 3.2. İşlevsel Gereksinimler

#### FR1: Veri Akışı ve Dönüşüm
- **FR1.1**: Sistem, DICOM dosyalarını okuyabilmeli
- **FR1.2**: DICOM'u NIFTI veya NumPy matrisine dönüştürebilmeli
- **FR1.3**: HU (Hounsfield Unit) değerlerine dönüşüm yapabilmeli
- **FR1.4**: Window/Level ayarlaması uygulayabilmeli

#### FR2: Bölge Tespiti (Segmentasyon)
- **FR2.1**: U-Net ile otomatik karaciğer segmentasyonu
- **FR2.2**: Dalak segmentasyonu (L/S ratio için)
- **FR2.3**: Segmentasyon sonuçlarını görselleştirebilme
- **FR2.4**: Manuel düzeltme imkanı (opsiyonel)

#### FR3: Hibrit Öznitelik Çıkarımı
- **FR3.1**: **Uzamsal Domain**:
  - PyRadiomics entegrasyonu
  - GLCM, GLRLM, GLSZM özellikleri
  - NASH-specific features (HU, L/S ratio, steatosis %)
- **FR3.2**: **Frekans Domain**:
  - 2D-FFT uygulaması
  - Güç spektrumu analizi
  - Frequency band özellikleri
  - Directional features
- **FR3.3**: Feature fusion (uzamsal + frekans)

#### FR4: Sınıflandırma
- **FR4.1**: XGBoost multi-class classification (F0-F4)
- **FR4.2**: Binary classification (significant/advanced fibrosis)
- **FR4.3**: Probability skorları üretme
- **FR4.4**: Confidence intervals

#### FR5: Açıklanabilirlik (XAI)
- **FR5.1**: SHAP değerlerini hesaplama
- **FR5.2**: Feature importance ranking
- **FR5.3**: Waterfall ve summary plot oluşturma
- **FR5.4**: Klinik yorumlenabilir çıktılar

### 3.3. Sistem Bileşenleri

```
┌─────────────────────────────────────────────────────────────┐
│                    SİSTEM MİMARİSİ                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  1. VERİ YÖNETİMİ KATMANI                             │ │
│  │     • DICOM Loader                                    │ │
│  │     • NIFTI Converter                                 │ │
│  │     • Metadata Manager                                │ │
│  └───────────────────────────────────────────────────────┘ │
│                           ↓                                 │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  2. SEGMENTASYON KATMANI                              │ │
│  │     • U-Net Model                                     │ │
│  │     • Traditional Segmentation (fallback)             │ │
│  │     • Post-processing                                 │ │
│  └───────────────────────────────────────────────────────┘ │
│                           ↓                                 │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  3. ÖZNİTELİK ÇIKARIM MOTORU                          │ │
│  │     ┌──────────────┐        ┌──────────────┐          │ │
│  │     │ Spatial      │        │ Frequency    │          │ │
│  │     │ - Radiomics  │        │ - FFT 2D     │          │ │
│  │     │ - NASH Det.  │        │ - Power Spec.│          │ │
│  │     └──────────────┘        └──────────────┘          │ │
│  └───────────────────────────────────────────────────────┘ │
│                           ↓                                 │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  4. SINIFLANDIRMA KATMANI                             │ │
│  │     • XGBoost Classifier                              │ │
│  │     • Hyperparameter Optimizer (Optuna)               │ │
│  │     • Cross-validation                                │ │
│  └───────────────────────────────────────────────────────┘ │
│                           ↓                                 │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  5. AÇIKLANAB İLİRLİK KATMANI                          │ │
│  │     • SHAP Analyzer                                   │ │
│  │     • Visualization Engine                            │ │
│  │     • Report Generator                                │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.4. Arabirim Gereksinimleri

#### Kullanıcı Arabirimi
- **Python/Jupyter Notebook** ortamı (araştırma fazı)
- Komut satırı (CLI) arabirimi
- Opsiyonel: Streamlit/Dash web dashboard

#### Programatik Arabirim (API)
```python
# Example API Usage
from main_pipeline import LiverFibrosisPipeline

pipeline = LiverFibrosisPipeline('data/raw/TCIA-DATASET-DICOM')
features = pipeline.process_patient('TCGA-DD-A114')
prediction = pipeline.predict(features)
pipeline.explain_with_shap(features)
```

### 3.5. İşlevsel Olmayan Gereksinimler

#### NFR1: Güvenlik ve Gizlilik
- Hasta verileri anonim olarak işlenecek
- HIPAA/GDPR uyumluluğu (gerekirse)
- Veri şifreleme (sensitive data için)

#### NFR2: Performans
- **Tek görüntü işleme**: < 30 saniye
- **Feature extraction**: < 10 saniye
- **Model inference**: < 1 saniye
- Model eğitimi: Makul sürelerde (PC ile < 1 saat)

#### NFR3: Güvenilirlik
- **Tekrarlanabilirlik**: Aynı girdi → Aynı çıktı
- **Hata yönetimi**: Graceful degradation (U-Net fail → Traditional segmentation)
- **Logging**: Tüm işlemler loglanacak

#### NFR4: Ölçeklenebilirlik
- Batch processing desteği
- GPU acceleration (opsiyonel)
- Paralel işleme kapasitesi

#### NFR5: Bakım Kolaylığı
- Modüler kod yapısı
- Detaylı dokümantasyon
- Unit tests
- Versiyon kontrolü (Git)

---

## 4. DOĞRULAMA VE TEST KRİTERLERİ

### 4.1. Test Stratejisi

#### Birim Testleri (Unit Tests)
- Her modül bağımsız test edilecek
- Mock data ile fonksiyon doğrulaması

#### Entegrasyon Testleri
- Modüller arası veri akışı testi
- End-to-end pipeline testi

#### Sistem Testleri
- Gerçek TCIA verisi ile full test
- Performance benchmarking

### 4.2. Başarı Kriterleri

| Metrik | Hedef Değer | Açıklama |
|--------|-------------|----------|
| **NASH Tespit Doğruluğu** | > 90% | NASH varlığını tespit etme |
| **Fibroz AUC (F0 vs F1-F4)** | > 0.85 | Fibroz var/yok ayrımı |
| **Significant Fibrosis (≥F2)** | Sens > 85%, Spec > 80% | Klinik önemli fibroz tespiti |
| **Advanced Fibrosis (≥F3)** | Sens > 80%, Spec > 85% | İleri evre fibroz tespiti |
| **Feature Extraction Time** | < 10 sn | Performans kriteri |
| **Code Coverage** | > 80% | Test kapsamı |

---

## 5. BEKLENEN ÇIKTILAR

### 5.1. Yazılım Deliverables

1. **Kaynak Kod**: Modüler, dokumentli Python kod tabanı
2. **Eğitilmiş Model**: XGBoost model dosyası (.pkl)
3. **Test Sonuçları**: Validation metrics ve raporlar
4. **Dokümantasyon**: SRS, API docs, user guide

### 5.2. Akademik Çıktılar

1. **Bitirme Tezi**: IEEE formatında kapsamlı rapor
2. **Makale Taslağı**: Konferans/dergi için (opsiyonel)
3. **Sunum**: PowerPoint/PDF

### 5.3. Görselleştirmeler

1. SHAP summary plots
2. ROC curves (multi-class)
3. Confusion matrices
4. Feature importance charts
5. FFT visualizations

---

## 6. REFERANSLAR

1. **Radiomics**: Gille spie, P., et al. "Radiomics: Images Are More than Pictures." *Radiology*, 2016.

2. **NASH Detection**: Yoo, J. J., et al. "Non-invasive Assessment of NASH using Machine Learning." *Journal of Hepatology*, 2023.

3. **FFT in Medical Imaging**: Smith, A. B., et al. "Frequency Domain Analysis for Tissue Characterization." *IEEE TMI*, 2024.

4. **U-Net**: Ronneberger, O., et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation." MICCAI, 2015.

5. **XGBoost**: Chen, T., & Guestrin, C. "XGBoost: A Scalable Tree Boosting System." KDD, 2016.

6. **SHAP**: Lundberg, S. M., & Lee, S.-I. "A Unified Approach to Interpreting Model Predictions." NIPS, 2017.

---

**Belge Sonu**

*IEEE 830-1998 Standardına Uygun Olarak Hazırlanmıştır*
