# Project Summary: Multimodal Phishing Detection — Group 15

## Konteks
Research paper untuk mata kuliah di Binus University.
**Judul:** "A Multimodal Approach to Phishing Detection Integrating Message Context, URL Structure, and HTML Content"

Tujuan sesi ini: menjalankan eksperimen ML untuk bagian **Result & Discussion** paper.

---

## Arsitektur Sistem (3-Stage Latency-Aware Pipeline)

```
Input: Pesan WA/SMS (Bahasa Indonesia)
         │
         ▼
┌─────────────────────────────────┐
│ STAGE 1: Message Classification │
│ Model : TF-IDF + Logistic Reg.  │
│ Data  : sms_spam_indo.csv        │
└─────────────┬───────────────────┘
              │ Jika Ambiguous (confidence < 70%)
              ▼
┌─────────────────────────────────┐
│ STAGE 2: URL Classification     │
│ Model : XGBoost                 │
│ Data  : PhiUSIIL (URL features) │
└─────────────┬───────────────────┘
              │ Jika Ambiguous
              ▼
┌─────────────────────────────────┐
│ STAGE 3: HTML Classification    │
│ Model : Random Forest           │
│ Data  : PhiUSIIL (HTML features)│
└─────────────────────────────────┘
```

---

## Struktur Folder (path lokal)

```
D:\BINUS\Semester 4\RM\phising-detection\
├── data\
│   ├── sms_spam_indo.csv                  ← 1143 baris, kolom: 'Kategori', 'Pesan'
│   └── PhiUSIIL_Phishing_URL_Dataset.csv  ← 235k baris, 55 kolom, label: 0=phishing 1=legit
├── notebooks\
│   ├── 01_message_classification.ipynb    ← SUDAH DIFIX, siap dijalankan
│   ├── 02_url_classification.ipynb        ← Dibuat baru, siap dijalankan
│   └── 03_html_classification.ipynb       ← Dibuat baru, siap dijalankan
├── result\
│   ├── figures\                           ← 9 PNG sudah tersimpan
│   └── metrics\                           ← CSV + model .pkl sudah tersimpan
├── run_stage1.py                          ← Script runner Stage 1 (sudah dijalankan)
├── run_stage2.py                          ← Script runner Stage 2 (sudah dijalankan)
├── run_stage3.py                          ← Script runner Stage 3 (sudah dijalankan)
└── CLAUDE_CODE_SUMMARY.md                 ← file ini
```

---

## Dataset

| Stage | File | Kolom Penting |
|-------|------|---------------|
| Stage 1 | `sms_spam_indo.csv` | `Pesan` (teks), `Kategori` (ham=0 / spam=1) |
| Stage 2 | `PhiUSIIL_...csv` | 22 URL structural features + `label` |
| Stage 3 | `PhiUSIIL_...csv` | 28 HTML-derived features + `label` |

**Penting:** Label di PhiUSIIL: `0 = phishing`, `1 = legitimate` (kebalikan dari Stage 1).

---

## Hasil Eksperimen (sudah dijalankan, 2026-05-11)

| Stage | Model | Accuracy | Precision | Recall | F1-Score |
|-------|-------|----------|-----------|--------|----------|
| Stage 1 — Message | TF-IDF + Logistic Regression | **98.25%** | 0.9827 | 0.9825 | 0.9825 |
| Stage 2 — URL | XGBoost | **99.99%** | 0.9999 | 0.9999 | 0.9999 |
| Stage 3 — HTML | Random Forest | **99.74%** | 0.9974 | 0.9974 | 0.9974 |

File lengkap: `result/metrics/all_stages_comparison.csv`

### Detail Stage 1 (Confusion Matrix, test set 229 baris):
- TN=111, FP=3, FN=1, TP=114
- 15.7% pesan masuk "ambiguous" → dilanjut ke Stage 2

---

## Status Pekerjaan

- [x] Semua dependency terinstall (scikit-learn, xgboost, pandas, matplotlib, seaborn, joblib)
- [x] Notebook 01 difix (Colab → local, kolom CSV diperbaiki)
- [x] Notebook 02 dibuat baru
- [x] Notebook 03 dibuat baru
- [x] Ketiga script runner dijalankan → hasil tersimpan di `result/`
- [ ] **Notebook 01 dijalankan manual oleh user di VS Code** ← sedang dikerjakan (ada error Cell 4 yang sudah difix)
- [ ] Notebook 02 dijalankan manual
- [ ] Notebook 03 dijalankan manual
- [ ] Result & Discussion ditulis di paper

---

## Bug yang Sudah Difix di Notebook 01

| Cell | Bug | Fix |
|------|-----|-----|
| Cell 1 | `!pip install` (Colab syntax) + `from google.colab import drive` | Diganti import biasa + path setup lokal |
| Cell 2 | `drive.mount('/content/drive')` + path Google Drive | Diganti `pd.read_csv(os.path.join(BASE_DIR, 'data', 'sms_spam_indo.csv'))` |
| Cell 3 | `TEXT_COL='text'`, `LABEL_COL='label'` (kolom tidak ada) + `matplotlib.use('Agg')` konflik | Rename `Pesan`→`text`, `Kategori`→`label_str`, map ham/spam → 0/1, hapus `use('Agg')` |
| Cell 12 | Save model ke current dir (bukan result/) | Save ke `METRIC_DIR` = `result/metrics/` |

### Root Cause Utama Cell 4 Error:
Cell 3 lama memanggil `matplotlib.use('Agg')` **setelah** `plt` sudah diimport di Cell 1. Ini menyebabkan Python crash di Cell 3, sehingga `df['text']` tidak pernah dibuat, lalu Cell 4 gagal dengan error berikutnya.

---

## Hal Teknis Penting untuk Session Baru

1. **Notebook 01 Cell 1** sekarang set semua path:
   ```python
   BASE_DIR   = os.path.abspath(os.path.join(os.getcwd(), '..'))
   FIG_DIR    = os.path.join(BASE_DIR, 'result', 'figures')
   METRIC_DIR = os.path.join(BASE_DIR, 'result', 'metrics')
   ```
   Asumsi: notebook dijalankan dari folder `notebooks/` (default VS Code behavior).

2. **Windows terminal:** Jalankan script Python dengan `PYTHONIOENCODING=utf-8` kalau ada emoji di print statement.

3. **Stage 2 & 3** pakai sample 50k baris dari dataset 235k untuk kecepatan training.

4. **Folder output** di project ini bernama `result/` (bukan `results/`).

---

## Output yang Sudah Ada di result/

### result/figures/
- `stage1_label_distribution.png`
- `stage1_confusion_matrix.png`
- `stage1_top_keywords.png`
- `stage2_label_distribution.png`
- `stage2_confusion_matrix.png`
- `stage2_feature_importance.png`
- `stage3_confusion_matrix.png`
- `stage3_feature_importance.png`
- `all_stages_comparison.png`

### result/metrics/
- `stage1_metrics.csv`, `stage1_model_lr.pkl`, `stage1_tfidf_vectorizer.pkl`
- `stage2_metrics.csv`, `stage2_model_xgb.pkl`
- `stage3_metrics.csv`, `stage3_model_rf.pkl`
- `all_stages_comparison.csv`

---

## Referensi Paper

1. Opara et al. (2024) — URL + HTML hybrid detection
2. Haq et al. (2024) — Deep learning URL detection
3. Liaqat (2024) — BERT untuk phishing text classification
4. Abutair et al. (2020) — PhishHaven real-time framework (basis latency-aware design)
5. Zaman et al. (2025) — BERT + XLM-RoBERTa semantic URL analysis
