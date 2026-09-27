"""
Feature engineering for the entity resolution matcher.

featurize_pair(s1_record, cand_record) -> dict of numeric/categorical features
This is the function the model trains on and runs inference through.

Libraries: rapidfuzz (fast Levenshtein/token ratios), jellyfish (phonetics),
scikit-learn (TF-IDF cosine). All pip-installable, MIT/BSD licensed.
"""

import re
import string

import jellyfish
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

_LEGAL_SUFFIXES = [
    "corporation", "corp", "incorporated", "inc", "limited", "ltd",
    "private", "pvt", "llc", "llp", "co", "company", "group", "holdings",
]
_LEGAL_SUFFIX_RE = re.compile(
    r"\b(" + "|".join(_LEGAL_SUFFIXES) + r")\b\.?", re.IGNORECASE
)

_ADDR_ABBREVIATIONS = {
    r"\brd\b": "road", r"\bst\b": "street", r"\bave\b": "avenue",
    r"\bblvd\b": "boulevard", r"\bdr\b": "drive", r"\bln\b": "lane",
    r"\bapt\b": "apartment", r"\bste\b": "suite", r"\bunit\b": "unit",
}

_PUNCT_TABLE = str.maketrans("", "", string.punctuation.replace("&", ""))


def normalize_name(name):
    """Lowercase, expand '&', strip legal suffixes and punctuation."""
    if not isinstance(name, str):
        return ""
    s = name.lower()
    s = s.replace("&", " and ")
    s = _LEGAL_SUFFIX_RE.sub("", s)
    s = s.translate(_PUNCT_TABLE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_address(addr):
    """Lowercase, standardize common street abbreviations, strip punctuation."""
    if not isinstance(addr, str):
        return ""
    s = addr.lower()
    for pat, full in _ADDR_ABBREVIATIONS.items():
        s = re.sub(pat, full, s)
    s = s.translate(_PUNCT_TABLE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_numeric_tokens(addr_norm):
    """Pull out digit-only tokens (street numbers, PIN/ZIP codes)."""
    return re.findall(r"\b\d+\b", addr_norm)


# ---------------------------------------------------------------------------
# Similarity primitives
# ---------------------------------------------------------------------------

def jaccard(tokens_a, tokens_b):
    a, b = set(tokens_a), set(tokens_b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def token_overlap_ratio(tokens_a, tokens_b):
    """Fraction of the smaller token set covered by the larger one."""
    a, b = set(tokens_a), set(tokens_b)
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def phonetic_match(name_a, name_b):
    """1 if the first significant tokens sound alike (Soundex), else 0."""
    def first_token(s):
        toks = s.split()
        return toks[0] if toks else ""

    ta, tb = first_token(name_a), first_token(name_b)
    if not ta or not tb:
        return 0
    try:
        return int(jellyfish.soundex(ta) == jellyfish.soundex(tb))
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# TF-IDF cosine (fit once on the full name corpus, reuse for every pair)
# ---------------------------------------------------------------------------

class NameTfidf:
    """
    Fit once on every normalized business_name across all sources, then
    call .cosine(name_a, name_b) per pair. Fitting once and reusing avoids
    refitting a vectorizer per pair, which would be far too slow at scale.
    """

    def __init__(self, ngram_range=(2, 4)):
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram_range)
        self._fitted = False

    def fit(self, all_normalized_names):
        self.vectorizer.fit(all_normalized_names)
        self._fitted = True

    def cosine(self, name_a, name_b):
        if not self._fitted or not name_a or not name_b:
            return 0.0
        vecs = self.vectorizer.transform([name_a, name_b])
        return float(cosine_similarity(vecs[0], vecs[1])[0, 0])


# ---------------------------------------------------------------------------
# Main pairwise featurizer
# ---------------------------------------------------------------------------

def featurize_pair(s1_record, cand_record, tfidf: NameTfidf = None):
    """
    s1_record, cand_record: dicts (or pandas Series) with keys
        'business_name', 'business_address', 'country'
    Returns a flat dict of features. Country is passed through as a raw
    categorical value (encode it downstream, e.g. via one-hot or target
    encoding) -- never hard-code the set of countries, since the test
    set has France in addition to train's US/India.
    """
    name_a_raw = s1_record.get("business_name", "")
    name_b_raw = cand_record.get("business_name", "")
    addr_a_raw = s1_record.get("business_address", "")
    addr_b_raw = cand_record.get("business_address", "")

    name_a = normalize_name(name_a_raw)
    name_b = normalize_name(name_b_raw)
    addr_a = normalize_address(addr_a_raw)
    addr_b = normalize_address(addr_b_raw)

    name_tokens_a = name_a.split()
    name_tokens_b = name_b.split()
    addr_tokens_a = addr_a.split()
    addr_tokens_b = addr_b.split()

    num_a = extract_numeric_tokens(addr_a)
    num_b = extract_numeric_tokens(addr_b)

    country_a = s1_record.get("country", "")
    country_b = cand_record.get("country", "")

    features = {
        # --- name features ---
        "name_exact_match": int(name_a == name_b and name_a != ""),
        "name_jaccard": jaccard(name_tokens_a, name_tokens_b),
        "name_token_sort_ratio": fuzz.token_sort_ratio(name_a, name_b) / 100.0,
        "name_token_set_ratio": fuzz.token_set_ratio(name_a, name_b) / 100.0,
        "name_levenshtein_ratio": fuzz.ratio(name_a, name_b) / 100.0,
        "name_partial_ratio": fuzz.partial_ratio(name_a, name_b) / 100.0,
        "name_phonetic_match": phonetic_match(name_a, name_b),
        "name_len_ratio": _len_ratio(name_a, name_b),
        "name_tfidf_cosine": tfidf.cosine(name_a, name_b) if tfidf else 0.0,

        # --- address features ---
        "addr_token_overlap": token_overlap_ratio(addr_tokens_a, addr_tokens_b),
        "addr_jaccard": jaccard(addr_tokens_a, addr_tokens_b),
        "addr_levenshtein_ratio": fuzz.ratio(addr_a, addr_b) / 100.0,
        "addr_numeric_match": int(bool(set(num_a) & set(num_b))),
        "addr_a_has_numeric": int(bool(num_a)),
        "addr_b_has_numeric": int(bool(num_b)),
        "addr_len_ratio": _len_ratio(addr_a, addr_b),

        # --- structural / metadata features ---
        "country_match": int(country_a == country_b and country_a != ""),
        "country_a": country_a,   # keep raw; encode downstream (open-set)
        "country_b": country_b,
        "source_prefix_b": cand_record.get("entity_id", "")[:2],  # 'S2' or 'S3'
    }
    return features


def _len_ratio(a, b):
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 1.0
    return min(la, lb) / max(la, lb) if max(la, lb) > 0 else 0.0
