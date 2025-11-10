import json
import os
import hashlib
from typing import List, Dict, Any

import numpy as np

try:
    import joblib  # type: ignore
except Exception:  # pragma: no cover
    joblib = None

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover
    SentenceTransformer = None


def load_faq(path: str = None) -> List[Dict[str, Any]]:
    faq_path = path or os.path.join(os.path.dirname(__file__), 'faq.json')
    with open(faq_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError('faq.json must be an array of entries')
    return data


def _intent_model_path() -> str:
    return os.path.join(os.path.dirname(__file__), 'models', 'intent.joblib')


_INTENT_MODEL = None


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
        _INTENT_MODEL = joblib.load(path)
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
    tuned = model.get('thresholds') or {}
    if not vec or not clf or not mlb:
        return []
    Xv = vec.transform([message or ''])
    tags: List[str] = []
    generic = {'auth', 'login', 'courses', 'notifications', 'navigation', 'account', 'admin', 'tech', 'feed', 'general'}
    generic_threshold = max(threshold + 0.15, 0.5)
    try:
        probs = clf.predict_proba(Xv)[0]
        indices = probs.argsort()[::-1]
        for idx in indices[:max(top_k, 1)]:
            cls = str(mlb.classes_[idx])
            p = probs[idx]
            # Use tuned per-label threshold if available; else fall back to generic/default
            t = float(tuned.get(cls, None)) if cls in tuned else None
            if t is None:
                t = generic_threshold if cls in generic else threshold
            if p >= t:
                tags.append(cls)
        if not tags and len(indices) > 0:
            tags.append(str(mlb.classes_[indices[0]]))
    except Exception:
        pass
    return tags


_EMBEDDER = None
_EMB_INDEX = None  # {'sig': str, 'ids': List[str], 'matrix': np.ndarray}


def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER
    if SentenceTransformer is None:
        return None
    _EMBEDDER = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return _EMBEDDER


def _faq_signature(faq: List[Dict[str, Any]]) -> str:
    payload = json.dumps([(e.get('id'), e.get('question'), e.get('tags')) for e in faq], ensure_ascii=False, separators=(',', ':'))
    return hashlib.md5(payload.encode('utf-8', errors='ignore')).hexdigest()


def _embeddings_path() -> str:
    return os.path.join(os.path.dirname(__file__), 'models', 'faq_embeddings.joblib')


def _load_embeddings():
    global _EMB_INDEX
    if _EMB_INDEX is not None:
        return _EMB_INDEX
    if joblib is None:
        return None
    path = _embeddings_path()
    if not os.path.exists(path):
        return None
    try:
        _EMB_INDEX = joblib.load(path)
        return _EMB_INDEX
    except Exception:
        return None


def _save_embeddings(obj: Dict[str, Any]):
    if joblib is None:
        return
    os.makedirs(os.path.join(os.path.dirname(__file__), 'models'), exist_ok=True)
    joblib.dump(obj, _embeddings_path())


def _build_embeddings(faq: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    model = _get_embedder()
    if model is None:
        return None
    ids = []
    texts = []
    for e in faq:
        q = (e.get('question') or '').strip()
        tg = ' '.join([str(t) for t in e.get('tags', [])])
        texts.append((q + ' ' + tg).strip())
        ids.append(e.get('id'))
    matrix = model.encode(texts, normalize_embeddings=True)
    return {'sig': _faq_signature(faq), 'ids': ids, 'matrix': np.asarray(matrix, dtype=np.float32)}


def _ensure_embeddings_for_faq(faq: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    idx = _load_embeddings()
    sig = _faq_signature(faq)
    if idx and idx.get('sig') == sig:
        return idx
    built = _build_embeddings(faq)
    if built is not None:
        _save_embeddings(built)
    return built


def _needs_support_line(message: str, matched_tags: List[str], confidence: str) -> bool:
    msg = (message or '').lower()
    asks_support = any(w in msg for w in ['support', 'contact', 'help'])
    has_support_tag = any(t in ('support',) for t in matched_tags)
    if confidence == 'low':
        return True
    return bool(asks_support or has_support_tag)


def respond(user_message: str, faq_data: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    faq = faq_data or load_faq()
    intent_tags = predict_tags_ml(user_message)

    idx = _ensure_embeddings_for_faq(faq)
    best = None
    best_score = -1.0
    best_detail = None
    sem_sim = 0.0
    min_conf_threshold = 0.60  # minimum semantic confidence for direct answers
    # Collect top candidates for debugging/analysis
    top_candidates: List[Dict[str, Any]] = []
    if idx and _get_embedder() is not None:
        qv = _get_embedder().encode([user_message], normalize_embeddings=True)
        mat = idx['matrix']
        sims = (mat @ np.asarray(qv[0], dtype=np.float32))
        top_k = int(min(10, len(sims)))
        top_idx = np.argsort(-sims)[:top_k]
        # Pre-materialize entries by id for speed
        by_id = {str(x.get('id')): x for x in faq}
        for i in top_idx:
            entry_id = idx['ids'][i]
            e = by_id.get(str(entry_id))
            if not e:
                continue
            entry_tags = [str(t).lower() for t in e.get('tags', [])]
            tag_matches = sum(1 for t in intent_tags if t in entry_tags)
            # Combined score: semantic similarity primary, tag matches secondary
            combined = float(0.8 * float(sims[i]) + 0.2 * (tag_matches / max(1, len(entry_tags))))
            top_candidates.append({
                'id': e.get('id'),
                'semantic': float(sims[i]),
                'tagMatches': float(tag_matches),
                'combined': combined,
            })
            if combined > best_score:
                best = e
                best_score = combined
                best_detail = {
                    'score': combined,
                    'semantic': float(sims[i]),
                    'tagMatches': float(tag_matches),
                }
        sem_sim = float(best_detail['semantic']) if best_detail else 0.0
    else:
        # If embeddings unavailable, fall back to tags only
        for e in faq:
            entry_tags = [str(t).lower() for t in e.get('tags', [])]
            tag_matches = sum(1 for t in intent_tags if t in entry_tags)
            score = float(tag_matches)
            if score > best_score:
                best = e
                best_score = score
                best_detail = {'score': score, 'semantic': 0.0, 'tagMatches': float(tag_matches)}

    high_conf = (sem_sim >= min_conf_threshold) or (best_detail.get('tagMatches', 0) >= 1 and best_detail['score'] >= 0.5)

    if high_conf:
        confidence = 'high'
        answer = best.get('answer') or ''
        if _needs_support_line(user_message, intent_tags, confidence):
            # Avoid redundant guidance if the selected answer already mentions contact/support
            ans_l = answer.lower()
            if ('/contact' not in ans_l) and (' contact ' not in f' {ans_l} ') and (' support ' not in f' {ans_l} '):
                answer = answer.rstrip() + "\nFor additional help, use the contact form at /contact."
        return {
            'id': best.get('id'),
            'answer': answer,
            'confidence': confidence,
            'confidenceScore': float(sem_sim),
            'matchedTags': intent_tags,
            'intent': str(best.get('id')),
            'debug': {
                **(best_detail or {}),
                'alternatives': sorted(top_candidates, key=lambda x: -x.get('combined', 0.0))[:3] if top_candidates else [],
            },
        }

    # Low confidence: polite fallback, no arbitrary closest intent answer
    # Log low-confidence queries for future improvement
    try:
        log_dir = os.path.join(os.path.dirname(__file__), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, 'unhandled.jsonl'), 'a', encoding='utf-8') as f:
            rec = {
                'ts': __import__('datetime').datetime.utcnow().isoformat() + 'Z',
                'message': user_message,
                'intentTags': intent_tags,
                'bestScore': float(best_detail['score'] if best_detail else 0.0),
                'bestSemantic': float(best_detail['semantic'] if best_detail else 0.0),
                'topCandidates': sorted(top_candidates, key=lambda x: -x.get('combined', 0.0))[:5] if top_candidates else [],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass

    return {
        'id': 'fallback',
        'answer': "I'm not sure I understand that yet. Could you rephrase or use the menu?",
        'confidence': 'low',
        'confidenceScore': float(sem_sim),
        'matchedTags': intent_tags,
        'intent': 'fallback',
        'debug': {
            'score': best_detail['score'] if best_detail else 0.0,
            'alternatives': sorted(top_candidates, key=lambda x: -x.get('combined', 0.0))[:3] if top_candidates else [],
        },
    }
