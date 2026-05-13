import pandas as pd
from feature_extractor import extract_urls_from_text

df = pd.read_csv('result/metrics/external_eval_results.csv')
benign = df[df['true_label']=='BENIGN'].copy()

def get_url_len(msg):
    urls = extract_urls_from_text(msg)
    if not urls: return 0
    return len(urls[0])

benign['url_len'] = benign['message'].apply(get_url_len)

print('=== URL LENGTH: Benign FP vs TN ===')
print('FP (salah ditandai phishing):')
print(benign[benign['pred']=='PHISHING']['url_len'].describe().round(1))
print()
print('TN (benar dikenali legitimate):')
print(benign[benign['pred']=='LEGITIMATE']['url_len'].describe().round(1))
print()
print('PhiUSIIL legit: mean=26.3, std=4.8, max=49')
print()

fp = benign[benign['pred']=='PHISHING']
over = (fp['url_len'] > 49).sum()
under = (fp['url_len'] <= 49).sum()
print(f'Benign FP URLLength >  49 (out-of-distribution PhiUSIIL): {over}/{len(fp)}')
print(f'Benign FP URLLength <= 49 (in-distribution PhiUSIIL):     {under}/{len(fp)}')
