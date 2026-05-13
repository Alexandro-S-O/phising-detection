"""
Feature extraction utilities for the phishing detection pipeline.

Provides three extractors:
  - extract_urls_from_text()  : pull URLs from raw SMS/WhatsApp text
  - extract_url_features()    : compute 18 URL structural features (Stage 2 input)
  - extract_html_features()   : compute 28 HTML content features  (Stage 3 input)

All feature names match the PhiUSIIL dataset column names so the saved
.pkl models can be used directly without re-mapping.
"""

import re
import urllib.parse
from typing import Dict, List, Any

# ─── URL Extractor ────────────────────────────────────────────────────────────

# Pattern A: explicit protocol (http/https)
# Pattern B: www. prefix
# Pattern C: bare domain — word boundary + domain labels + known TLD + optional path
#   Covers URLs like: alexpay2.beget.tech  muhanovabeauty.ru  espn.go.com/nba/...
_KNOWN_TLDS = (
    r'com|net|org|edu|gov|info|biz|io|co|ai|ru|de|fr|br|id|ph|sg|'
    r'vn|my|th|tw|hk|at|it|nl|pl|us|ca|jp|cn|kr|es|se|no|fi|dk|be|'
    r'ch|hu|ro|bg|ua|il|sa|ae|mx|in|nz|ar|cl|za|ng|ma|uk|au|gr|'
    r'tech|xyz|online|site|store|shop|live|news|me|tv|cc|tk|ml|ga|cf|gq|'
    r'sch|ac|or|go|uni|beget|[a-z]{2,4}'
)

_URL_PATTERN = re.compile(
    r'(?:'
    r'https?://[^\s<>"\'()]+'                          # [A] http/https
    r'|www\.[a-zA-Z0-9][^\s<>"\'()]+'                 # [B] www.
    r'|(?<!\w)'                                        # [C] bare domain — not mid-word
      r'(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?\.){1,5}'
      r'(?:' + _KNOWN_TLDS + r')'
      r'(?:/[^\s<>"\'()\[\]]*)?'
      r'(?!\w)'                                        # not followed by word char
    r')',
    re.IGNORECASE
)

def extract_urls_from_text(text: str) -> List[str]:
    """Return all URLs found in an SMS/WhatsApp message string.

    Detects three URL forms:
      - Explicit protocol: https://example.com/path
      - www prefix:        www.example.com
      - Bare domain:       example.com  or  sub.domain.co.id/path
    """
    urls = _URL_PATTERN.findall(text)
    result = []
    for url in urls:
        url = url.rstrip('.,;!)')
        # Normalise to absolute URL so urllib.parse works downstream
        low = url.lower()
        if not low.startswith(('http://', 'https://')):
            url = ('https://' if low.startswith('www.') else 'http://') + url
        result.append(url)
    return result


# ─── URL Feature Extractor ────────────────────────────────────────────────────

_IP_PATTERN   = re.compile(r'^\d{1,3}(\.\d{1,3}){3}$')
_BASIC_CHARS  = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
                    '0123456789-._~:/?#[]@!$&\'()*+,;=%')

def extract_url_features(url: str) -> Dict[str, Any]:
    """
    Compute 18 URL structural features from a raw URL string.

    Feature list matches run_stage2.py URL_FEATURES exactly so the
    trained XGBoost model can be called without any column remapping.

    Excluded from PhiUSIIL original 22:
      URLSimilarityIndex  (data leakage)
      CharContinuationRate (unclear definition)
      TLDLegitimateProb   (needs external lookup table)
      URLCharProb         (needs character frequency reference)
    """
    parsed     = urllib.parse.urlparse(url)
    netloc     = parsed.netloc or ''
    domain     = netloc.split(':')[0]          # strip port
    parts      = domain.split('.')
    tld        = parts[-1] if len(parts) > 1 else ''
    # subdomain count: parts beyond "domain.tld" pair
    n_subdomain = max(0, len(parts) - 2)

    url_len = len(url)
    letters = sum(c.isalpha()  for c in url)
    digits  = sum(c.isdigit()  for c in url)
    special = sum(c not in _BASIC_CHARS for c in url)
    obf     = url.count('%')                   # percent-encoded chars

    return {
        'URLLength':                url_len,
        'DomainLength':             len(domain),
        'IsDomainIP':               1 if _IP_PATTERN.match(domain) else 0,
        'TLDLength':                len(tld),
        'NoOfSubDomain':            n_subdomain,
        'IsHTTPS':                  1 if url.lower().startswith('https') else 0,
        'HasObfuscation':           1 if obf > 0 else 0,
        'NoOfObfuscatedChar':       obf,
        'ObfuscationRatio':         round(obf / url_len, 6) if url_len else 0,
        'NoOfLettersInURL':         letters,
        'LetterRatioInURL':         round(letters / url_len, 6) if url_len else 0,
        'NoOfDegitsInURL':          digits,   # PhiUSIIL typo preserved
        'DegitRatioInURL':          round(digits / url_len, 6) if url_len else 0,
        'NoOfEqualsInURL':          url.count('='),
        'NoOfQMarkInURL':           url.count('?'),
        'NoOfAmpersandInURL':       url.count('&'),
        'NoOfOtherSpecialCharsInURL': special,
        'SpacialCharRatioInURL':    round(special / url_len, 6) if url_len else 0,
    }


# ─── HTML Feature Extractor ───────────────────────────────────────────────────

_SOCIAL_DOMAINS = {
    'facebook.com', 'twitter.com', 'x.com', 'instagram.com',
    'linkedin.com', 'youtube.com', 'tiktok.com', 'whatsapp.com',
    't.me', 'telegram.me'
}

def extract_html_features(html: str, domain: str = '') -> Dict[str, Any]:
    """
    Compute 28 HTML content features from a raw HTML string.

    Feature list matches run_stage3.py HTML_FEATURES exactly.
    Requires: pip install beautifulsoup4
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError("beautifulsoup4 is required: pip install beautifulsoup4")

    soup  = BeautifulSoup(html, 'html.parser')
    lines = html.splitlines()
    text  = soup.get_text(separator=' ').lower()

    # ── Metadata ──────────────────────────────────────────────────────────────
    title_tag  = soup.find('title')
    title_text = title_tag.get_text().strip().lower() if title_tag else ''
    has_title  = 1 if title_text else 0

    desc_tag      = soup.find('meta', attrs={'name': re.compile(r'^description$', re.I)})
    has_desc      = 1 if (desc_tag and desc_tag.get('content', '').strip()) else 0

    favicon       = soup.find('link', rel=lambda r: r and 'icon' in ' '.join(r).lower())
    has_favicon   = 1 if favicon else 0

    robots_tag    = soup.find('meta', attrs={'name': re.compile(r'^robots$', re.I)})
    robots        = 1 if robots_tag else 0

    viewport      = soup.find('meta', attrs={'name': re.compile(r'^viewport$', re.I)})
    is_responsive = 1 if viewport else 0

    # ── Title match scores (0 or 100 — simple substring check) ───────────────
    domain_root          = domain.split('.')[0].lower() if domain else ''
    domain_title_match   = 100 if (domain_root and domain_root in title_text) else 0
    url_title_match      = domain_title_match   # same signal without full URL

    # ── Content signals ───────────────────────────────────────────────────────
    has_copyright = 1 if ('©' in text or 'copyright' in text) else 0
    has_bank      = 1 if any(w in text for w in ['bank', 'banking', 'rekening', 'atm']) else 0
    has_pay       = 1 if any(w in text for w in ['pay', 'payment', 'bayar', 'transfer', 'pulsa']) else 0
    has_crypto    = 1 if any(w in text for w in ['crypto', 'bitcoin', 'ethereum', 'btc', 'wallet']) else 0

    # ── Links ─────────────────────────────────────────────────────────────────
    all_hrefs  = [a.get('href', '') for a in soup.find_all('a', href=True)]
    has_social = 1 if any(sd in h for sd in _SOCIAL_DOMAINS for h in all_hrefs) else 0
    n_self_ref = sum(1 for h in all_hrefs
                     if domain and (domain in h or h.startswith('/') or h == '#'))
    n_empty    = sum(1 for h in all_hrefs if not h.strip() or h.strip() == '#')
    n_ext_ref  = sum(1 for h in all_hrefs
                     if h.startswith('http') and domain and domain not in h)

    # ── Form / input signals ──────────────────────────────────────────────────
    has_submit   = 1 if (
        soup.find('input',  {'type': re.compile(r'^submit$',   re.I)}) or
        soup.find('button', {'type': re.compile(r'^submit$',   re.I)}) or
        soup.find('button', string=re.compile(r'submit|login|sign in', re.I))
    ) else 0
    has_hidden   = 1 if soup.find('input', {'type': re.compile(r'^hidden$',   re.I)}) else 0
    has_password = 1 if soup.find('input', {'type': re.compile(r'^password$', re.I)}) else 0

    forms       = soup.find_all('form')
    has_ext_form = 0
    for form in forms:
        action = form.get('action', '')
        if action and action.startswith('http') and domain and domain not in action:
            has_ext_form = 1
            break

    # ── Counts ────────────────────────────────────────────────────────────────
    n_image  = len(soup.find_all('img'))
    n_css    = len(soup.find_all('link',
                   rel=lambda r: r and 'stylesheet' in ' '.join(r).lower()))
    n_js     = len(soup.find_all('script'))
    n_iframe = len(soup.find_all('iframe'))
    n_popup  = len(soup.find_all(onclick=re.compile(r'window\.open', re.I)))

    # Redirect: meta refresh counts as a URL redirect
    meta_refresh  = soup.find('meta', {'http-equiv': re.compile(r'^refresh$', re.I)})
    n_url_redirect  = 1 if meta_refresh else 0
    n_self_redirect = 0   # JS-based self-redirects not detectable from static HTML

    return {
        'LineOfCode':           len(lines),
        'LargestLineLength':    max((len(l) for l in lines), default=0),
        'HasTitle':             has_title,
        'DomainTitleMatchScore': domain_title_match,
        'URLTitleMatchScore':   url_title_match,
        'HasFavicon':           has_favicon,
        'Robots':               robots,
        'IsResponsive':         is_responsive,
        'NoOfURLRedirect':      n_url_redirect,
        'NoOfSelfRedirect':     n_self_redirect,
        'HasDescription':       has_desc,
        'NoOfPopup':            n_popup,
        'NoOfiFrame':           n_iframe,
        'HasExternalFormSubmit': has_ext_form,
        'HasSocialNet':         has_social,
        'HasSubmitButton':      has_submit,
        'HasHiddenFields':      has_hidden,
        'HasPasswordField':     has_password,
        'Bank':                 has_bank,
        'Pay':                  has_pay,
        'Crypto':               has_crypto,
        'HasCopyrightInfo':     has_copyright,
        'NoOfImage':            n_image,
        'NoOfCSS':              n_css,
        'NoOfJS':               n_js,
        'NoOfSelfRef':          n_self_ref,
        'NoOfEmptyRef':         n_empty,
        'NoOfExternalRef':      n_ext_ref,
    }
