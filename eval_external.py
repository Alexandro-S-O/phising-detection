"""
External Dataset Evaluation — 250 URL batch
  100 Phishing  (URLPhishing.txt)
  100 Benign    (URLBenign.txt)
   50 Defacement(URLDefacement.txt)

Runs silently through the 3-stage pipeline and prints
precision / recall / F1 / accuracy + per-stage routing table.
"""

import os, sys, re, urllib.parse
import joblib, numpy as np, pandas as pd
import requests
from feature_extractor import extract_urls_from_text, extract_url_features, extract_html_features

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.abspath(__file__))
METRIC_DIR = os.path.join(BASE, 'result', 'metrics')
DATA_DIR   = r"D:\BINUS\Semester 4\RM\other URL phising datset"

PHISHING_FILE    = os.path.join(DATA_DIR, 'URLPhishing.txt')
BENIGN_FILE      = os.path.join(DATA_DIR, 'URLBenign.txt')
DEFACEMENT_FILE  = os.path.join(DATA_DIR, 'URLDefacement.txt')

# ── Config ────────────────────────────────────────────────────────────────────
THRESHOLD = 0.70
TRUSTED_SUFFIXES = {'.ac.id', '.go.id', '.sch.id', '.mil.id', '.or.id'}

URL_FEATURES = [
    'URLLength', 'DomainLength', 'IsDomainIP', 'TLDLength',
    'NoOfSubDomain', 'HasObfuscation', 'NoOfObfuscatedChar', 'ObfuscationRatio',
    'NoOfLettersInURL', 'LetterRatioInURL', 'NoOfDegitsInURL', 'DegitRatioInURL',
    'NoOfEqualsInURL', 'NoOfQMarkInURL', 'NoOfAmpersandInURL',
    'NoOfOtherSpecialCharsInURL', 'SpacialCharRatioInURL', 'IsHTTPS'
]
HTML_FEATURES = [
    'LineOfCode', 'LargestLineLength', 'HasTitle', 'DomainTitleMatchScore',
    'URLTitleMatchScore', 'HasFavicon', 'Robots', 'IsResponsive',
    'NoOfURLRedirect', 'NoOfSelfRedirect', 'HasDescription', 'NoOfPopup',
    'NoOfiFrame', 'HasExternalFormSubmit', 'HasSocialNet', 'HasSubmitButton',
    'HasHiddenFields', 'HasPasswordField', 'Bank', 'Pay', 'Crypto',
    'HasCopyrightInfo', 'NoOfImage', 'NoOfCSS', 'NoOfJS',
    'NoOfSelfRef', 'NoOfEmptyRef', 'NoOfExternalRef'
]

# ── Load models ───────────────────────────────────────────────────────────────
print("Loading models...")
models = {k: joblib.load(os.path.join(METRIC_DIR, f))
          for k, f in [
              ('vectorizer', 'stage1_tfidf_vectorizer.pkl'),
              ('s1',         'stage1_model_lr.pkl'),
              ('s2',         'stage2_model_xgb.pkl'),
              ('s3',         'stage3_model_rf.pkl'),
          ]}
print("Models loaded.\n")

# ── Silent classify ───────────────────────────────────────────────────────────
def classify(sms_text: str) -> dict:
    """
    Returns dict with keys:
      pred  : 'PHISHING' | 'LEGITIMATE'
      stage : 1 | 2 | 3 | 'trusted_bypass' | 'no_url_fallback'
    """
    # Stage 1
    vec     = models['vectorizer'].transform([sms_text])
    s1_prob = models['s1'].predict_proba(vec)[0]
    s1_pred = int(s1_prob.argmax())
    s1_conf = float(s1_prob.max())
    s1_norm = 1 - s1_pred      # flip ham/spam → PhiUSIIL convention

    if s1_conf >= THRESHOLD:
        return {'pred': 'PHISHING' if s1_norm == 0 else 'LEGITIMATE', 'stage': 1}

    # URL extraction
    urls = extract_urls_from_text(sms_text)
    if not urls:
        return {'pred': 'PHISHING' if s1_norm == 0 else 'LEGITIMATE', 'stage': 'no_url_fallback'}

    url    = urls[0]
    parsed = urllib.parse.urlparse(url)
    domain = parsed.netloc.split(':')[0]

    # Trusted bypass
    if any(domain.endswith(sfx) for sfx in TRUSTED_SUFFIXES):
        return {'pred': 'LEGITIMATE', 'stage': 'trusted_bypass'}

    # Stage 2
    url_df  = pd.DataFrame([extract_url_features(url)])[URL_FEATURES]
    s2_prob = models['s2'].predict_proba(url_df)[0]
    s2_pred = int(s2_prob.argmax())
    s2_conf = float(s2_prob.max())

    if s2_conf >= THRESHOLD:
        return {'pred': 'PHISHING' if s2_pred == 0 else 'LEGITIMATE', 'stage': 2}

    # Stage 3
    try:
        resp = requests.get(url, timeout=5, headers={'User-Agent': 'Mozilla/5.0'})
        resp.raise_for_status()
        html_df  = pd.DataFrame([extract_html_features(resp.text, domain)])[HTML_FEATURES]
        s3_prob  = models['s3'].predict_proba(html_df)[0]
        s3_pred  = int(s3_prob.argmax())
        return {'pred': 'PHISHING' if s3_pred == 0 else 'LEGITIMATE', 'stage': 3}
    except Exception:
        return {'pred': 'PHISHING' if s2_pred == 0 else 'LEGITIMATE', 'stage': 2}


# ── Run evaluation ────────────────────────────────────────────────────────────
def run_file(path: str, true_label: str) -> pd.DataFrame:
    rows = []
    with open(path, encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]
    total = len(lines)
    for i, sms in enumerate(lines, 1):
        print(f"  [{i:3d}/{total}] {sms[:70]}{'...' if len(sms)>70 else ''}", flush=True)
        result = classify(sms)
        rows.append({
            'true_label': true_label,
            'pred':       result['pred'],
            'stage':      result['stage'],
            'message':    sms[:80],
        })
    return pd.DataFrame(rows)

print("=" * 65)
print("  EVALUATING — 100 PHISHING")
print("=" * 65)
df_phish = run_file(PHISHING_FILE, 'PHISHING')

print("\n" + "=" * 65)
print("  EVALUATING — 100 BENIGN")
print("=" * 65)
df_benign = run_file(BENIGN_FILE, 'BENIGN')

print("\n" + "=" * 65)
print("  EVALUATING — 50 DEFACEMENT")
print("=" * 65)
df_def = run_file(DEFACEMENT_FILE, 'DEFACEMENT')

df_all = pd.concat([df_phish, df_benign, df_def], ignore_index=True)

# ── Metrics ───────────────────────────────────────────────────────────────────
def print_section(title):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)

# --- Primary: Phishing vs Benign ---
print_section("PRIMARY METRICS — Phishing Detection (100P + 100B)")

df_primary = df_all[df_all['true_label'].isin(['PHISHING', 'BENIGN'])].copy()
tp = ((df_primary['true_label'] == 'PHISHING') & (df_primary['pred'] == 'PHISHING')).sum()
tn = ((df_primary['true_label'] == 'BENIGN')   & (df_primary['pred'] == 'LEGITIMATE')).sum()
fp = ((df_primary['true_label'] == 'BENIGN')   & (df_primary['pred'] == 'PHISHING')).sum()
fn = ((df_primary['true_label'] == 'PHISHING') & (df_primary['pred'] == 'LEGITIMATE')).sum()

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
accuracy  = (tp + tn) / len(df_primary)
fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0   # false positive rate

print(f"  TP (phishing caught)       : {tp}")
print(f"  TN (benign correct)        : {tn}")
print(f"  FP (benign → phishing)     : {fp}   ← false alarm")
print(f"  FN (phishing → legitimate) : {fn}   ← missed phishing")
print(f"")
print(f"  Accuracy    : {accuracy*100:.1f}%")
print(f"  Precision   : {precision*100:.1f}%")
print(f"  Recall      : {recall*100:.1f}%   (phishing catch rate)")
print(f"  F1-Score    : {f1*100:.1f}%")
print(f"  FP Rate     : {fpr*100:.1f}%   (false alarm rate on benign)")

# --- Stage routing breakdown ---
print_section("STAGE ROUTING — How Each Category Was Decided")
for cat in ['PHISHING', 'BENIGN', 'DEFACEMENT']:
    sub = df_all[df_all['true_label'] == cat]
    n = len(sub)
    if n == 0: continue
    routing = sub['stage'].value_counts()
    print(f"\n  [{cat}] n={n}")
    for stage, cnt in routing.items():
        print(f"    Stage {str(stage):15s}: {cnt:3d}  ({cnt/n*100:.1f}%)")

# --- Defacement breakdown ---
print_section("DEFACEMENT ANALYSIS — 50 URLs (out-of-scope category)")
def_phish = (df_def['pred'] == 'PHISHING').sum()
def_legit = (df_def['pred'] == 'LEGITIMATE').sum()
print(f"  Classified as PHISHING   : {def_phish}  ({def_phish/50*100:.1f}%)")
print(f"  Classified as LEGITIMATE : {def_legit}  ({def_legit/50*100:.1f}%)")
print(f"  (No ground-truth label for defacement in this pipeline)")

# --- False negatives (missed phishing) ---
missed = df_phish[df_phish['pred'] == 'LEGITIMATE']
if len(missed) > 0:
    print_section(f"MISSED PHISHING (False Negatives) — {len(missed)} cases")
    for _, row in missed.iterrows():
        print(f"  Stage {row['stage']} | {row['message']}")

# --- False positives (benign flagged) ---
false_alarms = df_benign[df_benign['pred'] == 'PHISHING']
if len(false_alarms) > 0:
    print_section(f"FALSE ALARMS (Benign → Phishing) — {len(false_alarms)} cases")
    for _, row in false_alarms.iterrows():
        print(f"  Stage {row['stage']} | {row['message']}")

# --- Save results ---
out_path = os.path.join(BASE, 'result', 'metrics', 'external_eval_results.csv')
df_all.to_csv(out_path, index=False)
print(f"\n  Results saved to: result/metrics/external_eval_results.csv")
print("=" * 65)
