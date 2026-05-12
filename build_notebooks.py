"""Build/patch .ipynb notebooks for local execution."""

import json, os
import nbformat as nbf

BASE = os.path.dirname(os.path.abspath(__file__))
NB_DIR = os.path.join(BASE, 'notebooks')

def md(text): return nbf.v4.new_markdown_cell(text)
def code(src): return nbf.v4.new_code_cell(src)

# ─────────────────────────────────────────────────────────────────────────────
# PATCH Notebook 01
# ─────────────────────────────────────────────────────────────────────────────
nb01_path = os.path.join(NB_DIR, '01_message_classification.ipynb')
with open(nb01_path, encoding='utf-8') as f:
    nb01 = nbf.read(f, as_version=4)

# Find and patch Cell 1 (install) — remove colab-only pip install banner
# Find Cell 2 (drive mount) — replace with local load
# Find Cell 3 (column names) — fix TEXT_COL and LABEL_COL

for cell in nb01.cells:
    src = cell.get('source', '')
    if 'drive.mount' in src or 'google.colab' in src:
        cell['source'] = """import os

BASE = os.path.dirname(os.path.abspath('__file__')) if '__file__' in dir() else os.getcwd()
# Sesuaikan path ke lokasi project Anda
DATASET_PATH = os.path.join(BASE, '..', 'data', 'sms_spam_indo.csv')

df = pd.read_csv(DATASET_PATH)
print(f'Dataset berhasil di-load!')
print(f'Total baris: {len(df)}')
print(f'Kolom: {list(df.columns)}')
df.head(10)"""

    if "TEXT_COL = 'text'" in src and "LABEL_COL = 'label'" in src:
        cell['source'] = src.replace(
            "TEXT_COL = 'text'    # kolom berisi teks pesan",
            "TEXT_COL = 'Pesan'  # kolom aktual di sms_spam_indo.csv"
        ).replace(
            "LABEL_COL = 'label'  # kolom berisi label (0=aman, 1=phishing/spam)",
            "LABEL_COL = 'Kategori'  # kolom aktual: nilai 'ham' (aman) dan 'spam' (phishing)"
        ).replace(
            "df = df.rename(columns={TEXT_COL: 'text', LABEL_COL: 'label'})",
            "df = df.rename(columns={TEXT_COL: 'text', LABEL_COL: 'label_str'})\n"
            "df['label'] = df['label_str'].map({'ham': 0, 'spam': 1})\n"
            "df = df.dropna(subset=['label']).reset_index(drop=True)\n"
            "df['label'] = df['label'].astype(int)"
        )

    # Fix save paths for figures
    if "plt.savefig('stage1_" in src:
        cell['source'] = src.replace(
            "plt.savefig('stage1_",
            "plt.savefig(os.path.join(BASE, '..', 'result', 'figures', 'stage1_"
        ).replace(".png', dpi=150", ".png'), dpi=150")

with open(nb01_path, 'w', encoding='utf-8') as f:
    nbf.write(nb01, f)
print(f'Notebook 01 patched: {nb01_path}')

# ─────────────────────────────────────────────────────────────────────────────
# CREATE Notebook 02 — Stage 2: URL Classification (XGBoost)
# ─────────────────────────────────────────────────────────────────────────────
nb02 = nbf.v4.new_notebook()
nb02.cells = [
    md("# Stage 2: URL Feature Classification\n### Multimodal Phishing Detection — Group 15, Binus University\n\n**Model:** XGBoost  \n**Data:** PhiUSIIL_Phishing_URL_Dataset.csv (URL structural features)"),
    md("## Cell 1: Import Libraries"),
    code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, joblib, warnings
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
warnings.filterwarnings('ignore')

BASE       = os.path.abspath('..')
FIG_DIR    = os.path.join(BASE, 'result', 'figures')
METRIC_DIR = os.path.join(BASE, 'result', 'metrics')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(METRIC_DIR, exist_ok=True)
print('Libraries imported.')"""),

    md("## Cell 2: Load Dataset"),
    code("""DATA_PATH = os.path.join(BASE, 'data', 'PhiUSIIL_Phishing_URL_Dataset.csv')

URL_FEATURES = [
    'URLLength', 'DomainLength', 'IsDomainIP', 'URLSimilarityIndex',
    'CharContinuationRate', 'TLDLegitimateProb', 'URLCharProb', 'TLDLength',
    'NoOfSubDomain', 'HasObfuscation', 'NoOfObfuscatedChar', 'ObfuscationRatio',
    'NoOfLettersInURL', 'LetterRatioInURL', 'NoOfDegitsInURL', 'DegitRatioInURL',
    'NoOfEqualsInURL', 'NoOfQMarkInURL', 'NoOfAmpersandInURL',
    'NoOfOtherSpecialCharsInURL', 'SpacialCharRatioInURL', 'IsHTTPS'
]
LABEL_COL = 'label'  # 0=phishing, 1=legitimate

df = pd.read_csv(DATA_PATH, usecols=URL_FEATURES + [LABEL_COL], nrows=50000)
df = df.dropna().reset_index(drop=True)
print(f'Loaded: {len(df)} rows, {len(URL_FEATURES)} URL features')
print(df[LABEL_COL].value_counts().rename({0: 'Phishing', 1: 'Legitimate'}))
df.head()"""),

    md("## Cell 3: EDA — Label Distribution"),
    code("""fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df[LABEL_COL].value_counts().plot(kind='bar', ax=axes[0], color=['#e74c3c', '#2ecc71'])
axes[0].set_title('Distribusi Label — URL Dataset')
axes[0].set_xticklabels(['Phishing (0)', 'Legitimate (1)'], rotation=0)
axes[0].set_ylabel('Jumlah')
df[LABEL_COL].value_counts().plot(kind='pie', ax=axes[1],
    labels=['Phishing', 'Legitimate'], colors=['#e74c3c', '#2ecc71'], autopct='%1.1f%%')
axes[1].set_title('Proporsi Label')
axes[1].set_ylabel('')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'stage2_label_distribution.png'), dpi=150, bbox_inches='tight')
plt.show()"""),

    md("## Cell 4: Train/Test Split"),
    code("""X = df[URL_FEATURES]
y = df[LABEL_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f'Train: {len(X_train)} | Test: {len(X_test)}')"""),

    md("## Cell 5: Train XGBoost Model"),
    code("""model = XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    random_state=42, eval_metric='logloss', verbosity=0
)
model.fit(X_train, y_train)
print('Model trained.')"""),

    md("## Cell 6: Evaluate Model"),
    code("""y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall    = recall_score(y_test, y_pred, average='weighted')
f1        = f1_score(y_test, y_pred, average='weighted')

print('='*50)
print('HASIL EVALUASI -- STAGE 2: URL CLASSIFIER')
print('='*50)
print(f'Accuracy  : {accuracy:.4f} ({accuracy*100:.2f}%)')
print(f'Precision : {precision:.4f}')
print(f'Recall    : {recall:.4f}')
print(f'F1-Score  : {f1:.4f}')
print(classification_report(y_test, y_pred, target_names=['Phishing (0)', 'Legitimate (1)']))

stage2_metrics = {
    'Stage': 'Stage 2 - URL', 'Model': 'XGBoost',
    'Accuracy': round(accuracy, 4), 'Precision': round(precision, 4),
    'Recall': round(recall, 4), 'F1-Score': round(f1, 4)
}"""),

    md("## Cell 7: Confusion Matrix"),
    code("""cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
    xticklabels=['Phishing (0)', 'Legitimate (1)'],
    yticklabels=['Phishing (0)', 'Legitimate (1)'])
plt.title('Confusion Matrix -- Stage 2: URL Classifier (XGBoost)', fontsize=13)
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'stage2_confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.show()
tn, fp, fn, tp = cm.ravel()
print(f'TN={tn}  FP={fp}  FN={fn}  TP={tp}')"""),

    md("## Cell 8: Feature Importance"),
    code("""importances = pd.Series(model.feature_importances_, index=URL_FEATURES).sort_values(ascending=False)
plt.figure(figsize=(10, 8))
importances.head(15).plot(kind='barh', color='#e67e22')
plt.gca().invert_yaxis()
plt.title('Top 15 URL Feature Importances (XGBoost)', fontsize=13)
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'stage2_feature_importance.png'), dpi=150, bbox_inches='tight')
plt.show()"""),

    md("## Cell 9: Latency-Aware Simulation"),
    code("""THRESHOLD = 0.70
max_prob = y_prob.max(axis=1)
clear_phishing = ((y_pred == 0) & (max_prob >= THRESHOLD)).sum()
clear_legit    = ((y_pred == 1) & (max_prob >= THRESHOLD)).sum()
ambiguous      = (max_prob < THRESHOLD).sum()
total          = len(y_pred)

print(f'Confidence threshold : {THRESHOLD*100:.0f}%')
print(f'Langsung PHISHING    : {clear_phishing} ({clear_phishing/total*100:.1f}%)')
print(f'Langsung LEGITIMATE  : {clear_legit} ({clear_legit/total*100:.1f}%)')
print(f'AMBIGUOUS -> Stage 3 : {ambiguous} ({ambiguous/total*100:.1f}%)')
stage2_metrics['Ambiguous_pct'] = round(ambiguous / total, 4)"""),

    md("## Cell 10: Save Model & Metrics"),
    code("""joblib.dump(model, os.path.join(METRIC_DIR, 'stage2_model_xgb.pkl'))
pd.DataFrame([stage2_metrics]).to_csv(os.path.join(METRIC_DIR, 'stage2_metrics.csv'), index=False)
print('stage2_model_xgb.pkl saved')
print('stage2_metrics.csv saved')
print('STAGE 2 SELESAI! Lanjut ke 03_html_classification.ipynb')"""),
]

nb02_path = os.path.join(NB_DIR, '02_url_classification.ipynb')
with open(nb02_path, 'w', encoding='utf-8') as f:
    nbf.write(nb02, f)
print(f'Notebook 02 created: {nb02_path}')

# ─────────────────────────────────────────────────────────────────────────────
# CREATE Notebook 03 — Stage 3: HTML Classification (Random Forest)
# ─────────────────────────────────────────────────────────────────────────────
nb03 = nbf.v4.new_notebook()
nb03.cells = [
    md("# Stage 3: HTML Feature Classification\n### Multimodal Phishing Detection — Group 15, Binus University\n\n**Model:** Random Forest  \n**Data:** PhiUSIIL_Phishing_URL_Dataset.csv (HTML-derived features)"),
    md("## Cell 1: Import Libraries"),
    code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, joblib, warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
warnings.filterwarnings('ignore')

BASE       = os.path.abspath('..')
FIG_DIR    = os.path.join(BASE, 'result', 'figures')
METRIC_DIR = os.path.join(BASE, 'result', 'metrics')
print('Libraries imported.')"""),

    md("## Cell 2: Load Dataset"),
    code("""DATA_PATH = os.path.join(BASE, 'data', 'PhiUSIIL_Phishing_URL_Dataset.csv')

HTML_FEATURES = [
    'LineOfCode', 'LargestLineLength', 'HasTitle', 'DomainTitleMatchScore',
    'URLTitleMatchScore', 'HasFavicon', 'Robots', 'IsResponsive',
    'NoOfURLRedirect', 'NoOfSelfRedirect', 'HasDescription', 'NoOfPopup',
    'NoOfiFrame', 'HasExternalFormSubmit', 'HasSocialNet', 'HasSubmitButton',
    'HasHiddenFields', 'HasPasswordField', 'Bank', 'Pay', 'Crypto',
    'HasCopyrightInfo', 'NoOfImage', 'NoOfCSS', 'NoOfJS',
    'NoOfSelfRef', 'NoOfEmptyRef', 'NoOfExternalRef'
]
LABEL_COL = 'label'  # 0=phishing, 1=legitimate

df = pd.read_csv(DATA_PATH, usecols=HTML_FEATURES + [LABEL_COL], nrows=50000)
df = df.dropna().reset_index(drop=True)
print(f'Loaded: {len(df)} rows, {len(HTML_FEATURES)} HTML features')
print(df[LABEL_COL].value_counts().rename({0: 'Phishing', 1: 'Legitimate'}))
df.head()"""),

    md("## Cell 3: Train/Test Split"),
    code("""X = df[HTML_FEATURES]
y = df[LABEL_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f'Train: {len(X_train)} | Test: {len(X_test)}')"""),

    md("## Cell 4: Train Random Forest"),
    code("""model = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print('Model trained.')"""),

    md("## Cell 5: Evaluate Model"),
    code("""y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall    = recall_score(y_test, y_pred, average='weighted')
f1        = f1_score(y_test, y_pred, average='weighted')

print('='*50)
print('HASIL EVALUASI -- STAGE 3: HTML CLASSIFIER')
print('='*50)
print(f'Accuracy  : {accuracy:.4f} ({accuracy*100:.2f}%)')
print(f'Precision : {precision:.4f}')
print(f'Recall    : {recall:.4f}')
print(f'F1-Score  : {f1:.4f}')
print(classification_report(y_test, y_pred, target_names=['Phishing (0)', 'Legitimate (1)']))

stage3_metrics = {
    'Stage': 'Stage 3 - HTML', 'Model': 'Random Forest',
    'Accuracy': round(accuracy, 4), 'Precision': round(precision, 4),
    'Recall': round(recall, 4), 'F1-Score': round(f1, 4)
}"""),

    md("## Cell 6: Confusion Matrix"),
    code("""cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
    xticklabels=['Phishing (0)', 'Legitimate (1)'],
    yticklabels=['Phishing (0)', 'Legitimate (1)'])
plt.title('Confusion Matrix -- Stage 3: HTML Classifier (Random Forest)', fontsize=13)
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'stage3_confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.show()
tn, fp, fn, tp = cm.ravel()
print(f'TN={tn}  FP={fp}  FN={fn}  TP={tp}')"""),

    md("## Cell 7: Feature Importance"),
    code("""importances = pd.Series(model.feature_importances_, index=HTML_FEATURES).sort_values(ascending=False)
plt.figure(figsize=(10, 8))
importances.head(15).plot(kind='barh', color='#8e44ad')
plt.gca().invert_yaxis()
plt.title('Top 15 HTML Feature Importances (Random Forest)', fontsize=13)
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'stage3_feature_importance.png'), dpi=150, bbox_inches='tight')
plt.show()"""),

    md("## Cell 8: Perbandingan Semua Stage"),
    code("""stage1 = pd.read_csv(os.path.join(METRIC_DIR, 'stage1_metrics.csv'))
stage2 = pd.read_csv(os.path.join(METRIC_DIR, 'stage2_metrics.csv'))

cols = ['Stage', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
comparison = pd.concat([
    stage1[cols],
    stage2[cols],
    pd.DataFrame([stage3_metrics])[cols]
], ignore_index=True)
comparison.to_csv(os.path.join(METRIC_DIR, 'all_stages_comparison.csv'), index=False)
print(comparison.to_string(index=False))"""),

    md("## Cell 9: Comparison Chart"),
    code("""fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(comparison))
width = 0.2
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
for i, (metric, color) in enumerate(zip(metrics_to_plot, colors)):
    ax.bar(x + i*width, comparison[metric], width, label=metric, color=color, alpha=0.85)
ax.set_xlabel('Stage')
ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison — All 3 Stages', fontsize=13)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(['Stage 1\\n(TF-IDF + LR)', 'Stage 2\\n(XGBoost)', 'Stage 3\\n(Random Forest)'])
ax.set_ylim(0.9, 1.01)
ax.legend()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.2f}'))
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'all_stages_comparison.png'), dpi=150, bbox_inches='tight')
plt.show()"""),

    md("## Cell 10: Save Model & Metrics"),
    code("""joblib.dump(model, os.path.join(METRIC_DIR, 'stage3_model_rf.pkl'))
pd.DataFrame([stage3_metrics]).to_csv(os.path.join(METRIC_DIR, 'stage3_metrics.csv'), index=False)
print('stage3_model_rf.pkl saved')
print('stage3_metrics.csv saved')
print('all_stages_comparison.csv saved')
print('SEMUA STAGE SELESAI!')"""),
]

nb03_path = os.path.join(NB_DIR, '03_html_classification.ipynb')
with open(nb03_path, 'w', encoding='utf-8') as f:
    nbf.write(nb03, f)
print(f'Notebook 03 created: {nb03_path}')
print('\nDONE. All notebooks ready.')
