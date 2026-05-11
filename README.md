# 🛡️ Multimodal Phishing Detection — Group 15

> A latency-aware, three-stage phishing detection framework integrating Indonesian message context, URL structure, and HTML content analysis.

**Binus University — Computer Science**
Alexandro Julio Soplantila · Kyoshiro Kaynelie · Claudius Cezar Panangian Manurung · Geoffrey Leslie

Supervisors: Muhammad Fikri Hasani, S.Kom., M.T. · Andien Dwi Novika, S.Kom., M.Kom.

---

## 📌 Overview

Phishing attacks in Indonesia increasingly combine technical URL manipulation with social engineering in Bahasa Indonesia — using trigger words like *"SELAMAT!"*, *"HADIAH"*, or *"Klik sekarang!"* to lure victims before they even reach a malicious link.

This project proposes a **three-stage, latency-aware detection pipeline**:

```
[Stage 1] Message Text ──► Phishing? ──► STOP & FLAG
               │ Ambiguous?
               ▼
[Stage 2]  URL Analysis ──► Phishing? ──► STOP & FLAG
               │ Ambiguous?
               ▼
[Stage 3] HTML Analysis ──► Final Classification
```

Each stage only triggers if the previous one is inconclusive — keeping the system fast for obvious cases while thorough for ambiguous ones.

---

## 📁 Repository Structure

```
phishing-detection-group15/
├── data/
│   ├── README.md                  ← instructions to download datasets
│   └── sample/                    ← small sample data for testing
├── notebooks/
│   ├── 01_message_classification.ipynb   ← Stage 1: Indonesian text (TF-IDF + LR)
│   ├── 02_url_classification.ipynb       ← Stage 2: URL features (XGBoost)
│   └── 03_html_classification.ipynb      ← Stage 3: HTML features (Random Forest)
├── results/
│   ├── metrics/                   ← accuracy, precision, recall, F1 per stage
│   └── figures/                   ← confusion matrices, feature importance plots
├── paper/
│   └── Research_Paper_Group15.pdf
├── requirements.txt
└── README.md
```

---

## 📦 Datasets

| Stage | Dataset | Source | Size |
|-------|---------|--------|------|
| Stage 1 — Message | Indonesian WhatsApp Spam/Phishing | Kaggle (2025 paper) | ~2,584 messages |
| Stage 2 & 3 — URL + HTML | PhiUSIIL Phishing URL Dataset | [UCI ML Repository](https://archive.ics.uci.edu/dataset/967) | 235,795 URLs, 54 features |

> **Note:** Datasets are NOT included in this repo due to size. See `data/README.md` for download instructions.

---

## 🚀 Getting Started

### Option A: Google Colab (Recommended)

1. Clone or open any notebook directly in Colab:
   ```
   https://colab.research.google.com/github/[your-username]/phishing-detection-group15/blob/main/notebooks/01_message_classification.ipynb
   ```
2. Run the first cell to install dependencies automatically.

### Option B: Local Machine

```bash
git clone https://github.com/[your-username]/phishing-detection-group15.git
cd phishing-detection-group15
pip install -r requirements.txt
jupyter notebook
```

---

## 🧪 Models Used

| Stage | Model | Rationale |
|-------|-------|-----------|
| Stage 1 — Message Text | TF-IDF + Logistic Regression | Fast, interpretable; effective for keyword-based Indonesian phishing triggers |
| Stage 2 — URL Analysis | XGBoost | Handles mixed feature types; strong performance on tabular URL features |
| Stage 3 — HTML Analysis | Random Forest | Robust to noisy HTML structural features; good feature importance reporting |

---

## 📊 Evaluation Metrics

All stages are evaluated using:
- **Accuracy** — overall correct predictions
- **Precision** — how often a phishing flag is correct (minimize false alarms)
- **Recall** — how many real threats are caught (minimize missed attacks)
- **F1-Score** — harmonic mean of Precision & Recall (primary metric for imbalanced data)

---

## 🗂️ Task Division

| Member | Role |
|--------|------|
| Alexandro Julio Soplantila | Conceptualization, Research Lead |
| Claudius Cezar Panangian Manurung | Data Collection |
| Geoffrey Leslie | Modelization |
| Kyoshiro Kaynelie | Methodology |

---

## 📚 References

1. Opara, C., Chen, Y., & Wei, B. (2024). *Look before you leap: Detecting phishing web pages by exploiting raw URL and HTML characteristics.* https://doi.org/10.1016/j.eswa.2023.121183
2. Haq, Q. E. ul, Faheem, M. H., & Ahmad, I. (2024). *Detecting phishing URLs based on a deep learning approach.* https://doi.org/10.3390/app142210086
3. Liaqat, M. S. (2024). *Exploring phishing attacks in the AI age.* https://www.jcbi.org/index.php/Main/article/view/567/534
4. Abutair, L., Belghith, A., & Al-Ahmadi, S. (2020). *PhishHaven — An Efficient Real-Time AI Phishing URLs Detection System.* IEEE Access.
5. Zaman, M. et al. (2025). *Transformer-based Phishing Detection: Leveraging BERT and XLM-RoBERTa for Semantic URL Analysis.*

---

## 📄 License

This project is developed for academic purposes at Binus University. Not for commercial use.
