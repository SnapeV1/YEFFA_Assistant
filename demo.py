from matcher import respond, load_faq
import sys


def _safe(s: str) -> str:
    enc = sys.stdout.encoding or 'utf-8'
    try:
        return s.encode(enc, errors='replace').decode(enc, errors='replace')
    except Exception:
        return s


def run(query: str) -> None:
    res = respond(query, load_faq())
    print("Q:", _safe(query))
    print("->", res['id'], f"({res['confidence']})")
    print(_safe(res['answer']))
    print('tags:', ','.join(res.get('matchedTags', [])))
    print('debug:', res.get('debug'))
    print('---')


tests = [
    "Can’t log in with Google",
    "Where are my notifications?",
    "Hi",
    "👍",
    # A clearly off-topic/unknown query to exercise fallback + logging
    "What’s the refund policy for premium plans?",
]

for t in tests:
    run(t)
