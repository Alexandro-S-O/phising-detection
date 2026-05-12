"""Rewrite all code cells in notebook 01 so they run cleanly in VS Code / Jupyter."""
import nbformat

with open('notebooks/01_message_classification.ipynb', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

new_code = [
    # Code Cell 1 — imports + path setup
    """\
# Cell 1: Import Libraries & Path Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, re, joblib, warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
warnings.filterwarnings('ignore')

# Notebook ada di /notebooks/, data & result ada di parent folder
BASE_DIR   = os.path.abspath(os.path.join(os.getcwd(), '..'))
FIG_DIR    = os.path.join(BASE_DIR, 'result', 'figures')
METRIC_DIR = os.path.join(BASE_DIR, 'result', 'metrics')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(METRIC_DIR, exist_ok=True)

print('Libraries imported.')
print(f'BASE_DIR   : {BASE_DIR}')
print(f'FIG_DIR    : {FIG_DIR}')
print(f'METRIC_DIR : {METRIC_DIR}')
""",

    # Code Cell 2 — load dataset
    """\
# Cell 2: Load Dataset
DATASET_PATH = os.path.join(BASE_DIR, 'data', 'sms_spam_indo.csv')

df = pd.read_csv(DATASET_PATH)
print(f'Dataset loaded: {len(df)} rows')
print(f'Kolom: {list(df.columns)}')
df.head(10)
""",

    # Code Cell 3 — EDA + fix columns
    """\
# Cell 3: EDA & Fix Columns
# Kolom aktual CSV: 'Pesan' (teks) dan 'Kategori' (ham / spam)
df = df.rename(columns={'Pesan': 'text', 'Kategori': 'label_str'})
df['label'] = df['label_str'].map({'ham': 0, 'spam': 1})
df = df.dropna(subset=['label']).reset_index(drop=True)
df['label'] = df['label'].astype(int)

print('Distribusi Label:')
print(df['label'].value_counts().rename({0: 'Aman (ham)', 1: 'Phishing (spam)'}))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df['label'].value_counts().plot(kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c'])
axes[0].set_title('Distribusi Label')
axes[0].set_xticklabels(['Aman (0)', 'Phishing (1)'], rotation=0)
axes[0].set_ylabel('Jumlah')
df['label'].value_counts().plot(kind='pie', ax=axes[1],
    labels=['Aman', 'Phishing'], colors=['#2ecc71', '#e74c3c'], autopct='%1.1f%%')
axes[1].set_title('Proporsi Label')
axes[1].set_ylabel('')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'stage1_label_distribution.png'), dpi=150, bbox_inches='tight')
plt.show()
print('stage1_label_distribution.png saved.')
""",

    # Code Cell 4 — preprocessing
    """\
# Cell 4: Preprocessing Text
def preprocess_text(text):
    if pd.isna(text):
        return ''
    text = str(text).lower()
    text = re.sub(r'http\\S+', 'URL', text)
    text = re.sub(r'\\d+', 'NUM', text)
    text = re.sub(r'[^\\w\\s]', ' ', text)
    text = re.sub(r'\\s+', ' ', text).strip()
    return text

df['text_clean'] = df['text'].apply(preprocess_text)
df = df[df['text_clean'].str.len() > 0].reset_index(drop=True)

print(f'Missing values: {df[["text_clean","label"]].isnull().sum().to_dict()}')
print(f'Total data bersih: {len(df)} baris')
df[['text', 'text_clean', 'label']].head(5)
""",

    # Code Cell 5 — train/test split
    """\
# Cell 5: Train/Test Split
X = df['text_clean']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f'Training set : {len(X_train)} baris')
print(f'Testing set  : {len(X_test)} baris')
print(f'Label train  : {y_train.value_counts().to_dict()}')
print(f'Label test   : {y_test.value_counts().to_dict()}')
""",

    # Code Cell 6 — TF-IDF
    """\
# Cell 6: TF-IDF Vectorization
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

print(f'TF-IDF shape train : {X_train_tfidf.shape}')
print(f'TF-IDF shape test  : {X_test_tfidf.shape}')
print(f'Sample features    : {tfidf.get_feature_names_out()[:20]}')
""",

    # Code Cell 7 — train model
    """\
# Cell 7: Train Logistic Regression
model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
model.fit(X_train_tfidf, y_train)
print('Model trained.')
""",

    # Code Cell 8 — evaluate
    """\
# Cell 8: Evaluate
y_pred = model.predict(X_test_tfidf)
y_prob = model.predict_proba(X_test_tfidf)

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall    = recall_score(y_test, y_pred, average='weighted')
f1        = f1_score(y_test, y_pred, average='weighted')

print('=' * 50)
print('HASIL EVALUASI -- STAGE 1: MESSAGE CLASSIFIER')
print('=' * 50)
print(f'Accuracy  : {accuracy:.4f} ({accuracy*100:.2f}%)')
print(f'Precision : {precision:.4f}')
print(f'Recall    : {recall:.4f}')
print(f'F1-Score  : {f1:.4f}')
print(classification_report(y_test, y_pred, target_names=['Aman (0)', 'Phishing (1)']))

stage1_metrics = {
    'Stage': 'Stage 1 - Message', 'Model': 'TF-IDF + Logistic Regression',
    'Accuracy': round(accuracy, 4), 'Precision': round(precision, 4),
    'Recall': round(recall, 4), 'F1-Score': round(f1, 4)
}
""",

    # Code Cell 9 — confusion matrix
    """\
# Cell 9: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['Aman (0)', 'Phishing (1)'],
    yticklabels=['Aman (0)', 'Phishing (1)'])
plt.title('Confusion Matrix -- Stage 1: Indonesian Message Classifier\\n(TF-IDF + Logistic Regression)', fontsize=13)
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'stage1_confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.show()

tn, fp, fn, tp = cm.ravel()
print(f'TN={tn}  FP={fp}  FN={fn}  TP={tp}')
print('stage1_confusion_matrix.png saved.')
""",

    # Code Cell 10 — top keywords
    """\
# Cell 10: Top Phishing Keywords
feature_names = tfidf.get_feature_names_out()
coef = model.coef_[0]
top_phishing_words = [(feature_names[i], coef[i]) for i in coef.argsort()[-20:][::-1]]
top_safe_words     = [(feature_names[i], coef[i]) for i in coef.argsort()[:20]]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
words_p, scores_p = zip(*top_phishing_words)
axes[0].barh(range(20), scores_p, color='#e74c3c')
axes[0].set_yticks(range(20)); axes[0].set_yticklabels(words_p)
axes[0].set_title('Top 20 Kata Indikator PHISHING', fontsize=13)
axes[0].set_xlabel('Koefisien TF-IDF'); axes[0].invert_yaxis()

words_s, scores_s = zip(*top_safe_words)
axes[1].barh(range(20), scores_s, color='#2ecc71')
axes[1].set_yticks(range(20)); axes[1].set_yticklabels(words_s)
axes[1].set_title('Top 20 Kata Indikator AMAN', fontsize=13)
axes[1].set_xlabel('Koefisien TF-IDF'); axes[1].invert_yaxis()

plt.suptitle('Feature Importance -- Stage 1: Indonesian Message Classifier', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'stage1_top_keywords.png'), dpi=150, bbox_inches='tight')
plt.show()
print('stage1_top_keywords.png saved.')
""",

    # Code Cell 11 — latency-aware simulation
    """\
# Cell 11: Latency-Aware Simulation
THRESHOLD = 0.70
max_prob = y_prob.max(axis=1)
clear_phishing = ((y_pred == 1) & (max_prob >= THRESHOLD)).sum()
clear_safe     = ((y_pred == 0) & (max_prob >= THRESHOLD)).sum()
ambiguous      = (max_prob < THRESHOLD).sum()
total          = len(y_pred)

print(f'Confidence threshold : {THRESHOLD*100:.0f}%')
print(f'Total test           : {total}')
print(f'Langsung AMAN        : {clear_safe} ({clear_safe/total*100:.1f}%)')
print(f'Langsung PHISHING    : {clear_phishing} ({clear_phishing/total*100:.1f}%)')
print(f'AMBIGUOUS -> Stage 2 : {ambiguous} ({ambiguous/total*100:.1f}%)')
stage1_metrics['Ambiguous_pct'] = round(ambiguous / total, 4)
""",

    # Code Cell 12 — save
    """\
# Cell 12: Save Model & Results
joblib.dump(model, os.path.join(METRIC_DIR, 'stage1_model_lr.pkl'))
joblib.dump(tfidf, os.path.join(METRIC_DIR, 'stage1_tfidf_vectorizer.pkl'))
pd.DataFrame([stage1_metrics]).to_csv(os.path.join(METRIC_DIR, 'stage1_metrics.csv'), index=False)

print('Saved to result/metrics/:')
print('  stage1_model_lr.pkl')
print('  stage1_tfidf_vectorizer.pkl')
print('  stage1_metrics.csv')
print()
print('STAGE 1 SELESAI!')
print(f'  Accuracy  : {stage1_metrics["Accuracy"]*100:.2f}%')
print(f'  F1-Score  : {stage1_metrics["F1-Score"]*100:.2f}%')
print('Lanjut ke: 02_url_classification.ipynb')
""",
]

# Replace code cells in order
code_idx = 0
for cell in nb.cells:
    if cell.cell_type == 'code':
        if code_idx < len(new_code):
            cell['source'] = new_code[code_idx]
        code_idx += 1

with open('notebooks/01_message_classification.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print(f'Done. Rewrote {len(new_code)} code cells.')
