# 🚀 KOMPLE KURULUM VE ÇALIŞTIRMA REHBERİ

**Proje:** Liver Fibrosis Staging System  
**Platform:** Windows 10/11  
**IDE:** Visual Studio Code  
**Python:** 3.8 veya üzeri

---

## 📋 İÇİNDEKİLER

1. [Ön Gereksinimler](#1-ön-gereksinimler)
2. [VSCode Kurulumu](#2-vscode-kurulumu)
3. [Proje Dosyalarını Hazırlama](#3-proje-dosyalarını-hazırlama)
4. [Python Environment Kurulumu](#4-python-environment-kurulumu)
5. [Dataset Hazırlama](#5-dataset-hazırlama)
6. [Dependency Kurulumu](#6-dependency-kurulumu)
7. [Uygulamayı Çalıştırma](#7-uygulamayı-çalıştırma)
8. [Sonuçları Görüntüleme](#8-sonuçları-görüntüleme)
9. [Sorun Giderme](#9-sorun-giderme)

---

## 1. ÖN GEREKSİNİMLER

### ✅ Gerekli Yazılımlar:

1. **Python 3.8+**
   - İndir: https://www.python.org/downloads/
   - Kurulumda "Add to PATH" seçeneğini işaretle!
   - Kontrol: `python --version` (PowerShell'de)

2. **Visual Studio Code**
   - İndir: https://code.visualstudio.com/
   - Ücretsiz

3. **Git** (Opsiyonel)
   - İndir: https://git-scm.com/download/win

### ✅ Disk Alanı:
- Proje: ~500 MB
- DICOM Dataset: ~10-15 GB
- Python packages: ~2 GB
- **Toplam: ~15-20 GB boş alan**

---

## 2. VSCODE KURULUMU

### Adım 1: VSCode İndir ve Kur
1. https://code.visualstudio.com/ adresine git
2. "Download for Windows" butonuna tıkla
3. İndirilen .exe dosyasını çalıştır
4. Varsayılan ayarlarla kur

### Adım 2: Python Extension Kur
1. VSCode'u aç
2. Sol taraftaki Extensions simgesine tıkla (veya `Ctrl+Shift+X`)
3. Ara kutusuna "Python" yaz
4. **"Python" by Microsoft** extension'ını kur
5. **"Pylance"** extension'ını da kur (önerilir)

### Adım 3: Terminal Ayarları
1. VSCode'da `Ctrl+Shift+P` bas
2. "Terminal: Select Default Profile" yaz
3. **PowerShell** seç

---

## 3. PROJE DOSYALARINI HAZIRLAMALAR

### Şu an projen zaten var!

Senin projen: `C:\Users\gozay\OneDrive\Masaüstü\bitirme`

### VSCode'da Açma:

**Yöntem 1: VSCode içinden**
1. VSCode'u aç
2. File → Open Folder
3. `C:\Users\gozay\OneDrive\Masaüstü\bitirme` klasörünü seç
4. "Select Folder" tıkla

**Yöntem 2: Windows Explorer'dan**
1. `C:\Users\gozay\OneDrive\Masaüstü\bitirme` klasörüne git
2. Sağ tıkla → "Open with Code"

### Proje Yapısı Kontrolü:

VSCode'da sol tarafta bu klasörleri görmeli açıkın:
```
bitirme/
├── src/               ✓ Kaynak kodlar
├── tests/             ✓ Test scriptleri
├── scripts/           ✓ Yardımcı scriptler
├── data/              ✓ (Boş olabilir, dataset buraya gelecek)
├── results/           ✓ Sonuçlar
├── docs/              ✓ Dokümantasyon
├── TCIA-DATASET-DICOM/  ✓ DICOM dosyaları
├── cleaned_clinical_data.csv  ✓ Klinik veriler
├── requirements.txt   ✓ Paket listesi
└── README.md          ✓ Ana dokuman
```

---

## 4. PYTHON ENVIRONMENT KURULUMU

### Adım 1: Virtual Environment Oluştur

VSCode'da terminal aç (`Ctrl+` ` veya Terminal → New Terminal):

```powershell
# Python versiyonu kontrol
python --version

# Virtual environment oluştur
python -m venv venv

# Activate et
.\venv\Scripts\Activate.ps1
```

**Sorun çıkarsa:**
```powershell
# Execution policy değiştir (admin gerekir)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Tekrar dene
.\venv\Scripts\Activate.ps1
```

Başarılı olursa terminal başında `(venv)` görünür:
```
(venv) PS C:\Users\gozay\OneDrive\Masaüstü\bitirme>
```

### Adım 2: VSCode Python Interpreter Seç

1. `Ctrl+Shift+P` bas
2. "Python: Select Interpreter" yaz
3. `.\venv\Scripts\python.exe` seçeneğini seç

---

## 5. DATASET HAZIRLAMALAR

### Senin Mevcut Dataset'in:

Zaten var! ✅ `TCIA-DATASET-DICOM/manifest-1768695854784/TCGA-LIHC/`

### Kontrol Et:

```powershell
# DICOM klasörünü kontrol
dir TCIA-DATASET-DICOM\manifest-1768695854784\TCGA-LIHC\

# Kaç hasta var?
(Get-ChildItem TCIA-DATASET-DICOM\manifest-1768695854784\TCGA-LIHC\ -Directory).Count
```

Çıktı: **~97** (veya benzeri sayı) hasta klasörü olmalı

### Klinik Data Kontrolü:

```powershell
# CSV dosyası var mı?
Test-Path cleaned_clinical_data.csv
```

Çıktı: `True` olmalı ✅

---

## 6. DEPENDENCY KURULUMU

### Adım 1: requirements.txt Kontrolü

VSCode'da `requirements.txt` dosyasını aç - içinde şunlar olmalı:
```
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
shap
pydicom
SimpleITK
opencv-python
scikit-image
scipy
```

### Adım 2: Tüm Paketleri Kur

Terminal'de (venv aktif olmalı!):

```powershell
# Tek komutla hepsini kur
pip install -r requirements.txt

# Süre: ~5-10 dakika
```

**İlerlemeyi göreceksin:**
```
Collecting numpy...
Collecting pandas...
...
Successfully installed numpy-1.24.0 pandas-2.0.0 ...
```

### Adım 3: Kurulum Testi

```powershell
# Test scripti çalıştır
python scripts\test_imports.py
```

**Beklenen çıktı:**
```
Testing imports...
✓ numpy
✓ pandas
✓ matplotlib
✓ sklearn
✓ xgboost
✓ shap
✓ pydicom
✓ SimpleITK
✓ cv2 (opencv)
All imports successful!
```

---

## 7. UYGULAMAYI ÇALIŞTIRMA

### Seçenek 1: Complete Pipeline (ÖNERİLEN)

**Full test - 42 hasta:**

```powershell
# Terminal'de (venv aktif)
python scripts\run_complete_pipeline.py
```

**Ne olacak:**
1. Klinik data yüklenir (~2 saniye)
2. DICOM'lar eşleştirilir (~5 saniye)
3. Feature extraction (~2-5 dakika, 42 hasta)
4. Model training (~30 saniye)
5. SHAP analizi (~1 dakika)
6. Results kaydedilir

**Toplam süre: ~5-10 dakika**

**Ekran çıktısı:**
```
=================================================
FULL END-TO-END PIPELINE - REAL DATA PROCESSING
=================================================
[STEP 1/8] Loading Clinical Data & Demographics...
✓ Loaded 53 patients with clinical data
...
[STEP 8/8] Saving Results...
✓ All results saved to: results/final_experiment/
=================================================
PIPELINE COMPLETE!
=================================================
```

### Seçenek 2: Quick Test (Hızlı)

**Sadece modül testi:**

```powershell
python tests\quick_test.py
```

Süre: ~10 saniye

### Seçenek 3: Specific Patient Test

**Tek hasta üzerinde:**

```powershell
python scripts\test_single_patient.py --patient TCGA-DD-A114
```

---

## 8. SONUÇLARI GÖRÜNTÜLEME

### Otomatik Oluşturulan Dosyalar:

Pipeline bittikten sonra `results/` klasöründe:

```
results/
├── final_experiment/
│   ├── extracted_features.csv      ← CSV dosyası (Excel'de aç)
│   ├── xgboost_model.json          ← Model
│   └── experiment_results.json     ← Metrikler
├── shap_analysis/
│   └── shap_summary_with_demographics.png  ← SHAP GRAFİĞİ (ÇİFT TIKLA!)
├── patient_results/
│   └── patient_level_results_table.png     ← HASTA TAHMİNLERİ (ÇİFT TIKLA!)
└── visualizations/
    ├── 1_fft_analysis_output.png
    ├── 2_nash_detection_output.png
    └── ... (diğer grafikler)
```

### Görselleri Açma:

**Windows Explorer'da:**
1. VSCode'da sol tarafta `results/shap_analysis/` klasörünü bul
2. `shap_summary_with_demographics.png` dosyasına **sağ tıkla**
3. "Reveal in File Explorer" seç
4. PNG dosyasına **çift tıkla** - Windows Photos ile açılır

**VSCode içinde:**
1. `shap_summary_with_demographics.png` dosyasına tıkla
2. VSCode preview panelinde görürsün

### Excel'de Veri Analizi:

```powershell
# CSV'yi Excel ile aç
start excel.exe results\final_experiment\extracted_features.csv
```

---

## 9. SORUN GİDERME

### ❌ Problem: "venv activate edilemiyor"

**Çözüm:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\venv\Scripts\Activate.ps1
```

### ❌ Problem: "ModuleNotFoundError: No module named 'xgboost'"

**Çözüm:**
```powershell
# venv aktif mi kontrol et
# Terminal başında (venv) olmalı

# Yoksa:
.\venv\Scripts\Activate.ps1

# Paketi kur
pip install xgboost
```

### ❌ Problem: "DICOM directory not found"

**Çözüm:**
```powershell
# Yol kontrolü
Test-Path TCIA-DATASET-DICOM\manifest-1768695854784\TCGA-LIHC

# False ise, doğru yolu bul:
dir TCIA-DATASET-DICOM\ -Recurse -Directory | Where-Object {$_.Name -eq "TCGA-LIHC"}
```

### ❌ Problem: "Memory Error" (RAM yetersiz)

**Çözüm:**
1. Daha az hasta ile test et
2. Veya `run_complete_pipeline.py` içinde batch_size azalt

### ❌ Problem: "Python not found"

**Çözüm:**
```powershell
# Python yolu Path'e ekle
$env:Path += ";C:\Users\gozay\AppData\Local\Programs\Python\Python311"

# Veya Python'u tekrar kur ve "Add to PATH" seç
```

---

## 🎯 HIZLI BAŞLANGIÇ - ÖZET

**5 Kolay Adım:**

```powershell
# 1. VSCode'da projeyi aç
# File → Open Folder → bitirme klasörünü seç

# 2. Terminal aç (Ctrl+`)
cd C:\Users\gozay\OneDrive\Masaüstü\bitirme

# 3. Virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 4. Dependencies kur
pip install -r requirements.txt

# 5. Çalıştır!
python scripts\run_complete_pipeline.py
```

**Bekle: ~10 dakika**

**Sonuç: `results/` klasöründe tüm grafikler ve sonuçlar!**

---

## 📞 YARDIM

Sorun olursa:
1. Error mesajını kopyala
2. `logs/` klasöründe log dosyalarını kontrol et
3. VSCode'da Problems paneline bak (`Ctrl+Shift+M`)

**Başarılar!** 🚀
