"""Stage 2: URL Feature Classification — local runner (XGBoost)"""

import pandas as pd
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

BASE       = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE, 'data', 'PhiUSIIL_Phishing_URL_Dataset.csv')
FIG_DIR    = os.path.join(BASE, 'result', 'figures')
METRIC_DIR = os.path.join(BASE, 'result', 'metrics')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(METRIC_DIR, exist_ok=True)

# URL structural features — only features computable from URL string alone.
# Excluded (4 features):
#   URLSimilarityIndex  → data leakage: all legitimate URLs = 100.0 in PhiUSIIL
#   CharContinuationRate → definition unclear, cannot reliably reproduce in extractor
#   TLDLegitimateProb   → requires external TLD probability lookup table
#   URLCharProb         → requires character frequency reference distribution
URL_FEATURES = [
    'URLLength', 'DomainLength', 'IsDomainIP', 'TLDLength',
    'NoOfSubDomain', 'HasObfuscation', 'NoOfObfuscatedChar', 'ObfuscationRatio',
    'NoOfLettersInURL', 'LetterRatioInURL', 'NoOfDegitsInURL', 'DegitRatioInURL',
    'NoOfEqualsInURL', 'NoOfQMarkInURL', 'NoOfAmpersandInURL',
    'NoOfOtherSpecialCharsInURL', 'SpacialCharRatioInURL', 'IsHTTPS'
]
LABEL_COL = 'label'  # 0=phishing, 1=legitimate

print("Loading dataset (sample 50k rows)...")
df = pd.read_csv(DATA_PATH, usecols=URL_FEATURES + [LABEL_COL], nrows=50000)
print(f"Loaded: {len(df)} rows, {len(URL_FEATURES)} URL features")
print(f"Label distribution:\n{df[LABEL_COL].value_counts().rename({0:'Phishing',1:'Legitimate'})}")

# Drop any rows with nulls
df = df.dropna().reset_index(drop=True)
X = df[URL_FEATURES]
y = df[LABEL_COL]

# ── Split ────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ── Label distribution plot ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
y.value_counts().plot(kind='bar', ax=axes[0], color=['#e74c3c', '#2ecc71'])
axes[0].set_title('Distribusi Label — URL Dataset')
axes[0].set_xticklabels(['Phishing (0)', 'Legitimate (1)'], rotation=0)
axes[0].set_ylabel('Jumlah')
y.value_counts().plot(kind='pie', ax=axes[1], labels=['Phishing', 'Legitimate'],
    colors=['#e74c3c', '#2ecc71'], autopct='%1.1f%%')
axes[1].set_title('Proporsi Label')
axes[1].set_ylabel('')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'stage2_label_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()

# ── Train XGBoost ────────────────────────────────────────────────────────────
print("\nTraining XGBoost...")
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss',
    verbosity=0
)
model.fit(X_train, y_train)
print("Model trained.")

# ── Evaluate ─────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall    = recall_score(y_test, y_pred, average='weighted')
f1        = f1_score(y_test, y_pred, average='weighted')

print('\n' + '='*50)
print('HASIL EVALUASI -- STAGE 2: URL CLASSIFIER')
print('='*50)
print(f'Accuracy  : {accuracy:.4f} ({accuracy*100:.2f}%)')
print(f'Precision : {precision:.4f}')
print(f'Recall    : {recall:.4f}')
print(f'F1-Score  : {f1:.4f}')
print('='*50)
print(classification_report(y_test, y_pred, target_names=['Phishing (0)', 'Legitimate (1)']))

stage2_metrics = {
    'Stage': 'Stage 2 - URL',
    'Model': 'XGBoost',
    'Accuracy':  round(accuracy, 4),
    'Precision': round(precision, 4),
    'Recall':    round(recall, 4),
    'F1-Score':  round(f1, 4)
}

# ── Confusion matrix ─────────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
    xticklabels=['Phishing (0)', 'Legitimate (1)'],
    yticklabels=['Phishing (0)', 'Legitimate (1)'])
plt.title('Confusion Matrix -- Stage 2: URL Classifier\n(XGBoost)', fontsize=13)
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'stage2_confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.close()
tn, fp, fn, tp = cm.ravel()
print(f'TN={tn}  FP={fp}  FN={fn}  TP={tp}')

# ── Feature importance ────────────────────────────────────────────────────────
importances = pd.Series(model.feature_importances_, index=URL_FEATURES).sort_values(ascending=False)
plt.figure(figsize=(10, 8))
importances.head(15).plot(kind='barh', color='#e67e22')
plt.gca().invert_yaxis()
plt.title('Top 15 URL Feature Importances\n(XGBoost)', fontsize=13)
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'stage2_feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()
print("stage2_feature_importance.png saved")

# ── Latency-aware simulation ─────────────────────────────────────────────────
THRESHOLD = 0.70
max_prob = y_prob.max(axis=1)
clear_phishing = ((y_pred == 0) & (max_prob >= THRESHOLD)).sum()
clear_legit    = ((y_pred == 1) & (max_prob >= THRESHOLD)).sum()
ambiguous      = (max_prob < THRESHOLD).sum()
total          = len(y_pred)

print('\n' + '='*55)
print('SIMULASI LATENCY-AWARE -- STAGE 2')
print('='*55)
print(f'Confidence threshold : {THRESHOLD*100:.0f}%')
print(f'Total URL test       : {total}')
print(f'Langsung PHISHING    : {clear_phishing} ({clear_phishing/total*100:.1f}%)')
print(f'Langsung LEGITIMATE  : {clear_legit} ({clear_legit/total*100:.1f}%)')
print(f'AMBIGUOUS -> Stage 3 : {ambiguous} ({ambiguous/total*100:.1f}%)')

stage2_metrics['Ambiguous_pct'] = round(ambiguous / total, 4)

# ── Save ──────────────────────────────────────────────────────────────────────
joblib.dump(model, os.path.join(METRIC_DIR, 'stage2_model_xgb.pkl'))
metrics_df = pd.DataFrame([stage2_metrics])
metrics_df.to_csv(os.path.join(METRIC_DIR, 'stage2_metrics.csv'), index=False)

print('\nFiles saved:')
print('  result/metrics/stage2_model_xgb.pkl')
print('  result/metrics/stage2_metrics.csv')
print('  result/figures/stage2_confusion_matrix.png')
print('  result/figures/stage2_feature_importance.png')
print('  result/figures/stage2_label_distribution.png')
print('\n' + '='*55)
print('STAGE 2 SELESAI!')
print('='*55)
