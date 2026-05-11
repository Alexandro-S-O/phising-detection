"""Stage 3: HTML Feature Classification — local runner (Random Forest)"""

import pandas as pd
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

BASE       = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE, 'data', 'PhiUSIIL_Phishing_URL_Dataset.csv')
FIG_DIR    = os.path.join(BASE, 'result', 'figures')
METRIC_DIR = os.path.join(BASE, 'result', 'metrics')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(METRIC_DIR, exist_ok=True)

# HTML-derived features (exclude URL structural and string columns)
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

print("Loading dataset (sample 50k rows)...")
df = pd.read_csv(DATA_PATH, usecols=HTML_FEATURES + [LABEL_COL], nrows=50000)
print(f"Loaded: {len(df)} rows, {len(HTML_FEATURES)} HTML features")
print(f"Label distribution:\n{df[LABEL_COL].value_counts().rename({0:'Phishing',1:'Legitimate'})}")

df = df.dropna().reset_index(drop=True)
X = df[HTML_FEATURES]
y = df[LABEL_COL]

# ── Split ─────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ── Train Random Forest ───────────────────────────────────────────────────────
print("\nTraining Random Forest...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("Model trained.")

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall    = recall_score(y_test, y_pred, average='weighted')
f1        = f1_score(y_test, y_pred, average='weighted')

print('\n' + '='*50)
print('HASIL EVALUASI -- STAGE 3: HTML CLASSIFIER')
print('='*50)
print(f'Accuracy  : {accuracy:.4f} ({accuracy*100:.2f}%)')
print(f'Precision : {precision:.4f}')
print(f'Recall    : {recall:.4f}')
print(f'F1-Score  : {f1:.4f}')
print('='*50)
print(classification_report(y_test, y_pred, target_names=['Phishing (0)', 'Legitimate (1)']))

stage3_metrics = {
    'Stage': 'Stage 3 - HTML',
    'Model': 'Random Forest',
    'Accuracy':  round(accuracy, 4),
    'Precision': round(precision, 4),
    'Recall':    round(recall, 4),
    'F1-Score':  round(f1, 4)
}

# ── Confusion matrix ──────────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
    xticklabels=['Phishing (0)', 'Legitimate (1)'],
    yticklabels=['Phishing (0)', 'Legitimate (1)'])
plt.title('Confusion Matrix -- Stage 3: HTML Classifier\n(Random Forest)', fontsize=13)
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'stage3_confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.close()
tn, fp, fn, tp = cm.ravel()
print(f'TN={tn}  FP={fp}  FN={fn}  TP={tp}')

# ── Feature importance ────────────────────────────────────────────────────────
importances = pd.Series(model.feature_importances_, index=HTML_FEATURES).sort_values(ascending=False)
plt.figure(figsize=(10, 8))
importances.head(15).plot(kind='barh', color='#8e44ad')
plt.gca().invert_yaxis()
plt.title('Top 15 HTML Feature Importances\n(Random Forest)', fontsize=13)
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'stage3_feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()
print("stage3_feature_importance.png saved")

# ── Comparison table: all 3 stages ───────────────────────────────────────────
stage1 = pd.read_csv(os.path.join(METRIC_DIR, 'stage1_metrics.csv'))
stage2 = pd.read_csv(os.path.join(METRIC_DIR, 'stage2_metrics.csv'))
stage3_row = pd.DataFrame([stage3_metrics])

cols = ['Stage', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
comparison = pd.concat([stage1[cols], stage2[cols], stage3_row[cols]], ignore_index=True)
comparison.to_csv(os.path.join(METRIC_DIR, 'all_stages_comparison.csv'), index=False)

print('\n' + '='*65)
print('PERBANDINGAN KETIGA STAGE')
print('='*65)
print(comparison.to_string(index=False))

# ── Comparison bar chart ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
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
ax.set_xticklabels([
    'Stage 1\n(TF-IDF + LR)',
    'Stage 2\n(XGBoost)',
    'Stage 3\n(Random Forest)'
])
ax.set_ylim(0.9, 1.01)
ax.legend()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.2f}'))
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'all_stages_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("all_stages_comparison.png saved")

# ── Save ──────────────────────────────────────────────────────────────────────
joblib.dump(model, os.path.join(METRIC_DIR, 'stage3_model_rf.pkl'))
metrics_df = pd.DataFrame([stage3_metrics])
metrics_df.to_csv(os.path.join(METRIC_DIR, 'stage3_metrics.csv'), index=False)

print('\nFiles saved:')
print('  result/metrics/stage3_model_rf.pkl')
print('  result/metrics/stage3_metrics.csv')
print('  result/metrics/all_stages_comparison.csv')
print('  result/figures/stage3_confusion_matrix.png')
print('  result/figures/stage3_feature_importance.png')
print('  result/figures/all_stages_comparison.png')
print('\n' + '='*55)
print('STAGE 3 SELESAI! SEMUA PIPELINE DONE.')
print('='*55)
