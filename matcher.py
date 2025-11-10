import json
import os
import re
import unicodedata
from typing import List, Dict, Any, Tuple

# Optional ML model support
try:
    import joblib  # type: ignore
except Exception:  # pragma: no cover
    joblib = None

# Generic, high-level domain tags we treat specially
GENERIC_TAGS = {
    'auth', 'login', 'courses', 'notifications', 'navigation', 'account', 'admin', 'tech', 'feed', 'general'
}


def _intent_model_path() -> str:
    return os.path.join(os.path.dirname(__file__), 'models', 'intent.joblib')


_INTENT_MODEL = None  # lazy-loaded model dict with vectorizer/classifier/label_binarizer

essential_model_keys = ('vectorizer', 'classifier', 'label_binarizer')


def _load_intent_model():
    global _INTENT_MODEL
    if _INTENT_MODEL is not None:
        return _INTENT_MODEL
    if joblib is None:
        return None
    path = _intent_model_path()
    if not os.path.exists(path):
        return None
    try:
        model = joblib.load(path)
        # Basic sanity: keys present
        if not all(k in model for k in essential_model_keys):
            return None
        _INTENT_MODEL = model
        return _INTENT_MODEL
    except Exception:
        return None


def predict_tags_ml(message: str, threshold: float = 0.4, top_k: int = 2) -> List[str]:
    model = _load_intent_model()
    if not model:
        return []
    vec = model.get('vectorizer')
    clf = model.get('classifier')
    mlb = model.get('label_binarizer')
    if not vec or not clf or not mlb:
        return []
    Xv = vec.transform([message or ''])
    tags: List[str] = []
    # Suppress overly generic domains unless probability is higher
    generic_threshold = max(threshold + 0.15, 0.5)
    try:
        probs = clf.predict_proba(Xv)[0]
        indices = probs.argsort()[::-1]
        for idx in indices[:max(top_k, 1)]:
            cls = str(mlb.classes_[idx])
            p = float(probs[idx])
            if cls in GENERIC_TAGS:
                if p >= generic_threshold:
                    tags.append(cls)
            else:
                if p >= threshold:
                    tags.append(cls)
        if not tags and len(indices) > 0:
            tags.append(str(mlb.classes_[indices[0]]))
    except Exception:
        try:
            scores = clf.decision_function(Xv)
            sc = scores[0]
            indices = sc.argsort()[::-1]
            for idx in indices[:max(top_k, 1)]:
                cls = str(mlb.classes_[idx])
                s = float(sc[idx])
                if cls in GENERIC_TAGS:
                    if s >= 0.5:
                        tags.append(cls)
                else:
                    if s >= 0:
                        tags.append(cls)
            if not tags and len(indices) > 0:
                tags.append(str(mlb.classes_[indices[0]]))
        except Exception:
            pass
    return tags


def load_faq(path: str = None) -> List[Dict[str, Any]]:
    faq_path = path or os.path.join(os.path.dirname(__file__), 'faq.json')
    with open(faq_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError('faq.json must be an array of entries')
    return data


def _strip_punct_and_symbols(s: str) -> str:
    out = []
    for ch in s:
        cat = unicodedata.category(ch)
        # Remove punctuation (P*) and symbols (S*). Keep spaces.
        if cat.startswith('P') or cat.startswith('S'):
            out.append(' ')
        else:
            out.append(ch)
    return ''.join(out)


def normalize(s: str) -> str:
    if s is None:
        return ''
    s = unicodedata.normalize('NFKC', str(s)).lower()
    s = _strip_punct_and_symbols(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize(s: str) -> List[str]:
    return [t for t in normalize(s).split(' ') if t]


# Lexicon rules: regex -> tags (data‑driven; AI classifier augments)
LEXICON: List[Tuple[re.Pattern, List[str]]] = [
    # Auth & login
    (re.compile(r"(?:can\'?t|cant)\s*log\s*in|\blog\s*in\b|\bsign\s*in\b|\blogin\b", re.I), ['auth', 'login']),
    (re.compile(r"google\s*(login|sign[-\s]*in)|\bgoogle\b|\bfedcm\b", re.I), ['auth', 'google-login', 'fedcm']),
    (re.compile(r"\bpassword\b", re.I), ['auth', 'password']),
    (re.compile(r"\breset\b|\bforgot\b", re.I), ['auth', 'reset']),
    (re.compile(r"\bsession\b|\btimeout\b|\bexpiring\b", re.I), ['auth', 'session', 'timeout']),

    # Notifications
    (re.compile(r"\bnotifications?\b|\bbell\b", re.I), ['notifications', 'bell']),
    (re.compile(r"\bclear\b|mark\s+all\s+as\s+read", re.I), ['notifications', 'clear', 'mark-read']),

    # Navigation / courses
    (re.compile(r"\bcourses?\b|\bcourse\b", re.I), ['courses']),
    (re.compile(r"\bsearch\b", re.I), ['search']),
    (re.compile(r"\bfilter\b", re.I), ['filter']),
    (re.compile(r"\blanguage\b", re.I), ['language']),
    (re.compile(r"\bpreview\b", re.I), ['preview']),
    (re.compile(r"\benrol?l\b", re.I), ['enroll']),

    # Admin / account
    (re.compile(r"\badmin\b|\bdashboard\b", re.I), ['admin']),
    (re.compile(r"\brole\b|\bfarmer\b", re.I), ['role', 'farmer']),

    # Profile/edit name/picture
    (re.compile(r"profile|avatar|photo|picture|pfp|display\s*name", re.I), ['account', 'profile']),
    (re.compile(r"(edit|change|update)\s+(my\s+)?(profile\s+)?(picture|photo|avatar|pfp)\b", re.I), ['profile-picture']),
    (re.compile(r"display\s*name|change\s+name|edit\s+name|username", re.I), ['display-name']),

    # Certificates
    (re.compile(r"certificate|certificat|certification", re.I), ['certificates', 'certificate']),

    # Error/support
    (re.compile(r"\berror\b|\bissue\b|\bproblem\b|\bbug\b|\bfailed\b|\bfailure\b|\bcrash\b", re.I), ['tech', 'support', 'error']),

    # Billing
    (re.compile(r"refund|billing|subscription|payment|premium", re.I), ['billing', 'subscription', 'payment', 'premium']),

    # Small talk / acks / off-topic
    (re.compile(r"^(hi|hello|hey)\b|\bthanks\b|\bthank you\b|\bgoodbye\b|\bsorry\b", re.I), ['greeting']),
    (re.compile(r"\bok\b|\byes\b", re.I), ['acknowledgement']),
    (re.compile(r"joke\b|tell\s+me\s+a\s+joke", re.I), ['off-topic']),
    # About/Introduce/YEFFA
    (re.compile(r"\bintroduce\s+(yourself|you)\b|\bwho\s+are\s+you\b|\babout\s+(you|yourself|yeffa|us)\b", re.I), ['about', 'introduce']),
    (re.compile(r"\bwhat\s+is\s+yeffa\b|\bwhat\s+does\s+yeffa\s+stand\s+for\b|\bye\s*ffa\b", re.I), ['about', 'yeffa', 'what-is', 'acronym']),
]


def infer_intent_tags(message: str) -> List[str]:
    tags: set[str] = set()
    for pattern, mapped in LEXICON:
        if pattern.search(message or ''):
            tags.update(mapped)
    return list(tags)


def jaccard(a_tokens: List[str], b_tokens: List[str]) -> float:
    a = set(a_tokens)
    b = set(b_tokens)
    inter = len(a & b)
    union = len(a | b)
    return 0.0 if union == 0 else inter / union


def score_entry(entry: Dict[str, Any], intent_tags: List[str], msg_tokens: List[str]) -> Dict[str, float]:
    entry_tags = [str(t).lower() for t in entry.get('tags', [])]
    generic = GENERIC_TAGS

    generic_matches = 0
    specific_matches = 0
    for t in intent_tags:
        if t in entry_tags:
            if t in generic:
                generic_matches += 1
            else:
                specific_matches += 1

    q_tokens = tokenize(entry.get('question', '') or '')
    tag_tokens = entry_tags
    overlap_q = jaccard(msg_tokens, q_tokens)
    overlap_t = jaccard(msg_tokens, tag_tokens)

    bonus = 0.0
    input_has_google = any(t == 'google' or 'google' in t for t in msg_tokens) or re.search(r"google", ' '.join(msg_tokens), re.I)
    if input_has_google:
        if 'google' in (entry.get('question', '') or '').lower():
            bonus += 0.6
        if 'fedcm' in entry_tags:
            bonus += 0.3

    score = specific_matches * 2.2 + generic_matches * 0.8 + overlap_q * 1.2 + overlap_t * 0.5 + bonus
    return {
        'score': score,
        'tagMatches': float(specific_matches + generic_matches),
        'overlapQ': overlap_q,
        'overlapT': overlap_t,
    }


def find_best_match(faq: List[Dict[str, Any]], user_message: str) -> Dict[str, Any]:
    # Start with rule-based tags
    intent_tags = infer_intent_tags(user_message)
    # Merge ML predictions, de-duplicated; avoid adding generic ML tags if we already have a specific tag
    ml_tags = predict_tags_ml(user_message)
    has_specific = any(t not in GENERIC_TAGS for t in intent_tags)
    for t in ml_tags:
        if has_specific and t in GENERIC_TAGS:
            continue
        if t not in intent_tags:
            intent_tags.append(t)

    msg_tokens = tokenize(user_message)
    best = None
    best_score = None
    best_detail = None
    for e in faq:
        s = score_entry(e, intent_tags, msg_tokens)
        if best is None or s['score'] > best_score:
            best = e
            best_score = s['score']
            best_detail = s

    high_conf = ((best_detail.get('tagMatches', 0) >= 1 and best_detail['score'] >= 2.0) or best_detail['score'] >= 2.5)
    med_conf = (best_detail['score'] >= 1.3)
    return {
        'best': best,
        'intentTags': intent_tags,
        'highConfidence': bool(high_conf),
        'mediumConfidence': bool(med_conf),
        'detail': best_detail,
    }


def _select_fallback(faq: List[Dict[str, Any]], intent_tags: List[str]) -> Dict[str, Any]:
    if 'acknowledgement' in intent_tags:
        for e in faq:
            if str(e.get('id', '')).lower() == 'chat-007':
                return e
    for e in faq:
        if str(e.get('id', '')).lower() == 'chat-001':
            return e
    for e in faq:
        if 'off-topic' in [str(t) for t in e.get('tags', [])]:
            return e
    return faq[0]


def _ensure_log_dir() -> str:
    d = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(d, exist_ok=True)
    return d


def _log_unhandled(message: str, details: Dict[str, Any]) -> None:
    try:
        d = _ensure_log_dir()
        path = os.path.join(d, 'unhandled.jsonl')
        rec = {
            'ts': __import__('datetime').datetime.utcnow().isoformat() + 'Z',
            'message': message,
            'details': details,
        }
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    except Exception:
        pass


def _needs_support_line(message: str, matched_tags: List[str], confidence: str) -> bool:
    msg = (message or '').lower()
    asks_support = any(w in msg for w in ['support', 'contact', 'help'])
    has_support_tag = any(t in ('support',) for t in matched_tags)
    if confidence == 'low':
        return True
    return bool(asks_support or has_support_tag)


def respond(user_message: str, faq_data: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    faq = faq_data or load_faq()
    res = find_best_match(faq, user_message)
    best = res['best']
    intent_tags = res['intentTags']
    detail = res['detail']
    if res['highConfidence'] or res['mediumConfidence']:
        confidence = 'high' if res['highConfidence'] else 'medium'
        answer = best.get('answer') or ''
        if _needs_support_line(user_message, intent_tags, confidence):
            answer = answer.rstrip() + "\nFor additional help, use the contact form at /contact."
        return {
            'id': best.get('id'),
            'answer': answer,
            'confidence': confidence,
            'matchedTags': intent_tags,
            'debug': detail,
        }

    fb = _select_fallback(faq, intent_tags)
    _log_unhandled(user_message, {'intentTags': intent_tags, 'best': {'id': best.get('id'), 'score': detail['score']}})
    return {
        'id': fb.get('id'),
        'answer': (fb.get('answer') or '').rstrip() + "\nFor additional help, use the contact form at /contact.",
        'confidence': 'low',
        'matchedTags': intent_tags,
        'debug': {'fallbackFrom': best.get('id'), 'score': detail['score']},
    }
