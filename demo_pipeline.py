"""
End-to-end phishing detection demo — Latency-Aware 3-Stage Pipeline
Group 15 — Binus University

Usage:
    python demo_pipeline.py
    python demo_pipeline.py "Pesan SMS kamu di sini https://link.xyz"

Requirements:
    pip install requests beautifulsoup4
    All 3 models must be trained first (run run_stage1/2/3.py).
"""

import os
import sys
import urllib.parse

import joblib
import numpy as np
import pandas as pd
import requests

from feature_extractor import (
    extract_urls_from_text,
    extract_url_features,
    extract_html_features,
)

# ─── Config ───────────────────────────────────────────────────────────────────

BASE       = os.path.dirname(os.path.abspath(__file__))
METRIC_DIR = os.path.join(BASE, 'result', 'metrics')
THRESHOLD  = 0.70   # confidence below this → escalate to next stage

# Indonesian institutional TLD suffixes that are out-of-distribution for
# PhiUSIIL (Western-centric dataset). Stage 2/3 models exhibit systematic
# false positives for these domains, so we apply a trusted-suffix bypass.
TRUSTED_SUFFIXES = {'.ac.id', '.go.id', '.sch.id', '.mil.id', '.or.id'}

# Feature lists must match training scripts exactly
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

# ─── Load models ──────────────────────────────────────────────────────────────

def load_models():
    paths = {
        'vectorizer': os.path.join(METRIC_DIR, 'stage1_tfidf_vectorizer.pkl'),
        's1':         os.path.join(METRIC_DIR, 'stage1_model_lr.pkl'),
        's2':         os.path.join(METRIC_DIR, 'stage2_model_xgb.pkl'),
        's3':         os.path.join(METRIC_DIR, 'stage3_model_rf.pkl'),
    }
    missing = [k for k, p in paths.items() if not os.path.exists(p)]
    if missing:
        print(f"[ERROR] Model file(s) not found: {missing}")
        print("  Run run_stage1.py, run_stage2.py, run_stage3.py first.")
        sys.exit(1)
    return {k: joblib.load(p) for k, p in paths.items()}


# ─── Helper ───────────────────────────────────────────────────────────────────

def _label(pred: int) -> str:
    return 'PHISHING' if pred == 0 else 'LEGITIMATE'

def _bar(conf: float) -> str:
    filled = int(conf * 20)
    return '[' + '█' * filled + '░' * (20 - filled) + f'] {conf:.1%}'


# ─── Pipeline ────────────────────────────────────────────────────────────────

def detect(sms_text: str, models: dict) -> int:
    print('\n' + '═' * 62)
    print('  INPUT MESSAGE')
    print('═' * 62)
    print(f'  {sms_text}')
    print('═' * 62)

    # ── STAGE 1: Text Classification ─────────────────────────────────────────
    print('\n┌─ STAGE 1 — Text Classification (TF-IDF + Logistic Regression)')
    vec      = models['vectorizer'].transform([sms_text])
    s1_prob  = models['s1'].predict_proba(vec)[0]
    s1_pred  = int(s1_prob.argmax())
    s1_conf  = float(s1_prob.max())

    # Stage 1 uses ham=0/spam=1; normalize to PhiUSIIL convention (0=phishing, 1=legit)
    s1_pred_norm = 1 - s1_pred

    print(f'│  Prediction  : {_label(s1_pred_norm)}')
    print(f'│  Confidence  : {_bar(s1_conf)}')

    if s1_conf >= THRESHOLD:
        print(f'└─ FINAL DECISION: {_label(s1_pred_norm)}  (Stage 1 confident)\n')
        return s1_pred_norm

    print(f'│  Confidence < {THRESHOLD:.0%} → AMBIGUOUS')
    print('└─ Escalating to Stage 2...')

    # ── URL Extraction ────────────────────────────────────────────────────────
    urls = extract_urls_from_text(sms_text)
    if not urls:
        print('\n  [!] No URL found in message.')
        print(f'  → FINAL DECISION (fallback): {_label(s1_pred_norm)}\n')
        return s1_pred_norm

    url    = urls[0]
    parsed = urllib.parse.urlparse(url)
    domain = parsed.netloc.split(':')[0]
    print(f'\n  URL extracted : {url}')

    # Trusted Indonesian institutional TLDs are out-of-distribution for
    # the PhiUSIIL-trained model (Western-centric dataset), causing false
    # positives. Bypass Stage 2/3 and return LEGITIMATE (PhiUSIIL label=1).
    if any(domain.endswith(sfx) for sfx in TRUSTED_SUFFIXES):
        print(f'  [i] Domain ends with trusted Indonesian institutional TLD')
        print(f'  → FINAL DECISION: LEGITIMATE  (trusted domain bypass)\n')
        return 1

    # ── STAGE 2: URL Classification ───────────────────────────────────────────
    print('\n┌─ STAGE 2 — URL Classification (XGBoost, 18 structural features)')
    url_feats = extract_url_features(url)
    url_df    = pd.DataFrame([url_feats])[URL_FEATURES]
    s2_prob   = models['s2'].predict_proba(url_df)[0]
    s2_pred   = int(s2_prob.argmax())
    s2_conf   = float(s2_prob.max())

    print(f'│  Prediction  : {_label(s2_pred)}')
    print(f'│  Confidence  : {_bar(s2_conf)}')

    if s2_conf >= THRESHOLD:
        print(f'└─ FINAL DECISION: {_label(s2_pred)}  (Stage 2 confident)\n')
        return s2_pred

    print(f'│  Confidence < {THRESHOLD:.0%} → AMBIGUOUS')
    print('└─ Escalating to Stage 3...')

    # ── STAGE 3: HTML Classification ──────────────────────────────────────────
    print('\n┌─ STAGE 3 — HTML Classification (Random Forest, 28 content features)')
    try:
        response  = requests.get(url, timeout=5,
                                 headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        html_feats = extract_html_features(response.text, domain)
        html_df    = pd.DataFrame([html_feats])[HTML_FEATURES]
        s3_prob    = models['s3'].predict_proba(html_df)[0]
        s3_pred    = int(s3_prob.argmax())
        s3_conf    = float(s3_prob.max())

        print(f'│  Prediction  : {_label(s3_pred)}')
        print(f'│  Confidence  : {_bar(s3_conf)}')
        print(f'└─ FINAL DECISION: {_label(s3_pred)}  (Stage 3)\n')
        return s3_pred

    except Exception as exc:
        print(f'│  [!] HTML fetch failed: {exc}')
        print(f'└─ FINAL DECISION (fallback): {_label(s2_pred)}  (Stage 2 result)\n')
        return s2_pred


# ─── Main ─────────────────────────────────────────────────────────────────────

DEFAULT_TEST_CASES = [
    # Obvious phishing SMS (Indonesia)
    "Selamat! Akun BCA Anda berhasil memenangkan undian Rp 50.000.000. "
    "Klik sekarang: http://bca-hadiah-resmi.xyz/klaim?id=9912",

    # Legitimate-looking message
    "Halo, jadwal ujian tengah semester telah diperbarui. "
    "Silakan cek di https://student.binus.ac.id/schedule",

    # Phishing tanpa URL (hanya teks)
    "URGENT: Kartu kredit Anda telah diblokir karena aktivitas mencurigakan. "
    "Hubungi 0800-1234-BANK segera untuk verifikasi.",

    # Mixed — URL ada tapi domain mencurigakan
    "Promo spesial! Belanja di Tokopedia dan dapatkan cashback 90%. "
    "Daftar di http://tokopedia-promo.tk/cashback hari ini saja!",
]

if __name__ == '__main__':
    print("Loading models...")
    models = load_models()
    print("All models loaded.\n")

    test_cases = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_TEST_CASES

    results = []
    for sms in test_cases:
        pred = detect(sms, models)
        results.append({'message': sms[:60] + '...', 'verdict': _label(pred)})

    print('\n' + '═' * 62)
    print('  SUMMARY')
    print('═' * 62)
    for r in results:
        icon = '🚨' if r['verdict'] == 'PHISHING' else '✅'
        print(f"  {icon} {r['verdict']:12s} | {r['message']}")
    print('═' * 62)
