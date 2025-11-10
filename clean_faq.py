import json
import re
from pathlib import Path

SRC = Path(__file__).parent / 'faq.json'


CONTACT_PATTERNS = [
    r"\bIf you need help,? use the contact form at /contact\.?",
    r"\bUse the contact form at /contact\b.*?\.?",
    r"\bFor support,? use the contact form at /contact\.?",
]

ENC_FIXES = [
    # Common Windows-1252/UTF-8 mojibake fixes
    ("â€™", "’"),  # right single quote
    ("â€˜", "‘"),  # left single quote
    ("â€“", "–"),  # en dash
    ("â€”", "—"),  # em dash
    ("â€œ", "“"),  # left double quote
    ("â€�", "”"),  # right double quote
    ("â€¦", "…"),  # ellipsis
    ("Ã¢â‚¬â„¢", "’"),  # right single quote
    ("Ã¢â‚¬Ëœ", "‘"),  # left single quote
    ("Ã¢â‚¬â€œ", "–"),  # en dash
    ("Ã¢â‚¬â€", "—"),  # em dash
    ("Ã¢â€˜", "‘"),
    ("Ã¢â€›", "›"),
    ("Ã¢â€š", "‚"),
    ("Ã¢â€žÂ¢", "™"),
    ("Ã¢â€œ", "“"),  # left double quote
    ("Ã¢â€", "”"),  # right double quote
    ("Ã‚Â", ""),    # stray non-breaking space marker
    ("Â", ""),      # another stray
    ("Ã©", "é"),
    ("Ã", ""),      # generic stray if isolated
    # Earlier artifacts observed
    ("weÃ¢â‚¬â„¢ll", "we’ll"),
    ("we�?Tll", "we’ll"),
    ("don�?Tt", "don’t"),
    ("Don�?Tt", "Don’t"),
    ("Sign�?`In", "Sign‑In"),
    ("sign�?`in", "sign‑in"),
    ("third�?`party", "third‑party"),
    ("third�?`", "third‑"),
]


def _remove_control_chars(s: str) -> str:
    # Remove C0/C1 control characters except standard whitespace
    return re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", s)


def _try_unmangle(s: str) -> str:
    """Attempt to repair mojibake by re-encoding as bytes and decoding as UTF-8.
    Tries common mis-decoding paths (latin1, cp1252). Returns original on failure.
    """
    orig = s
    for enc in ("cp1252", "latin1"):
        try:
            b = s.encode(enc, errors="strict")
            fixed = b.decode("utf-8", errors="strict")
            if fixed != orig:
                s = fixed
        except Exception:
            continue
    return s


def clean_answer(text: str) -> str:
    if not text:
        return text
    out = _remove_control_chars(text)
    # Try unmangling first if suspicious sequences present
    if any(t in out for t in ("â€", "Ã", "Â")):
        out = _try_unmangle(out)
    # Remove any contact sentence(s)
    for pat in CONTACT_PATTERNS:
        out = re.sub(pat, "", out, flags=re.IGNORECASE)
    # Normalize whitespace and stray punctuation
    out = re.sub(r"\s+", " ", out).strip()
    # Encoding fixes
    for bad, good in ENC_FIXES:
        out = out.replace(bad, good)
    # Fix paired replacement chars used as quotes: �text� -> "text"
    out = re.sub(r"�\s*([^�]+?)\s*�", r'"\1"', out)
    # Replace backtick-quoted phrases with straight quotes: `text` -> "text"
    out = re.sub(r"`([^`]+)`", r'"\1"', out)
    # Replace mixed backtick+apostrophe quotes: `text' -> "text"
    out = re.sub(r"`([^']+)'", r'"\1"', out)
    # Common word-level fixes with replacement chars
    out = re.sub(r"(?i)sign�+in", "sign‑in", out)
    out = re.sub(r"(?i)third�+party", "third‑party", out)
    out = out.replace("don�t", "don’t").replace("doesn�t", "doesn’t").replace("can�t", "can’t").replace("won�t", "won’t").replace("it�s", "it’s")
    # Cleanup doubled spaces before punctuation
    out = re.sub(r"\s+([.,;:!?])", r"\1", out)
    return out


def main():
    # Read and strip UTF-8 BOM if present to avoid JSONDecodeError
    raw = SRC.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    data = json.loads(raw.decode('utf-8'))
    changed = 0
    for e in data:
        # Clean both question and answer fields
        q = e.get('question') or ''
        a = e.get('answer') or ''
        new_q = clean_answer(q)
        new_a = clean_answer(a)
        if new_q != q:
            e['question'] = new_q
            changed += 1
        if new_a != a:
            e['answer'] = new_a
            changed += 1
    # Write back as UTF-8 without BOM
    SRC.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding='utf-8')
    print(f"Cleaned {changed} fields in {SRC}")


if __name__ == '__main__':
    main()
