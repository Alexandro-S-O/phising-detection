"""Check what feature is driving Stage 2 FP for benign URLs."""
import pandas as pd
from feature_extractor import extract_urls_from_text, extract_url_features

# Read original benign file
with open(r"D:\BINUS\Semester 4\RM\other URL phising datset\URLBenign.txt", encoding='utf-8') as f:
    benign_msgs = [l.strip() for l in f if l.strip()]

results = pd.read_csv('result/metrics/external_eval_results.csv')
benign_res = results[results['true_label']=='BENIGN'].reset_index(drop=True)

# Sample 5 FP and 5 TN — print their features
fp_msgs = [benign_msgs[i] for i in range(len(benign_msgs))
           if benign_res.loc[i, 'pred'] == 'PHISHING'][:5]
tn_msgs = [benign_msgs[i] for i in range(len(benign_msgs))
           if benign_res.loc[i, 'pred'] == 'LEGITIMATE'][:5]

def show_features(msgs, label):
    print(f'\n=== {label} ===')
    for msg in msgs:
        urls = extract_urls_from_text(msg)
        if not urls:
            print(f'  NO URL: {msg[:60]}')
            continue
        url = urls[0]
        f = extract_url_features(url)
        print(f'  URL      : {url[:60]}')
        print(f'  IsHTTPS  : {f["IsHTTPS"]}  URLLength: {f["URLLength"]}  '
              f'NoOfSubDomain: {f["NoOfSubDomain"]}  '
              f'DomainLength: {f["DomainLength"]}')
        print(f'  NoOfQMark: {f["NoOfQMarkInURL"]}  NoOfEquals: {f["NoOfEqualsInURL"]}  '
              f'NoOfDigits: {f["NoOfDegitsInURL"]}  TLDLength: {f["TLDLength"]}')
        print()

show_features(fp_msgs, 'BENIGN FP (wrongly flagged PHISHING)')
show_features(tn_msgs, 'BENIGN TN (correctly LEGITIMATE)')
