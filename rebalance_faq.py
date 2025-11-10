import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import Counter
import argparse


DEFAULT_MERGE: Dict[str, str] = {
    'about-us': 'about',
    'who-we-are': 'about',
}


def load_faq(path: Path) -> List[Dict[str, Any]]:
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    data = json.loads(raw.decode('utf-8'))
    if not isinstance(data, list):
        raise ValueError('FAQ must be an array of entries')
    return data


def save_faq(path: Path, data: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding='utf-8')


def normalize_tags(tags: List[str], merge: Dict[str, str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for t in tags or []:
        tt = str(t).strip().lower()
        tt = merge.get(tt, tt)
        if tt and tt not in seen:
            out.append(tt)
            seen.add(tt)
    return out


def rebalance(
    data: List[Dict[str, Any]],
    min_count: int,
    drop_ubiquitous: bool,
    merge: Dict[str, str],
    fallback_tag: str = 'general',
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    # Merge/normalize first
    for e in data:
        e['tags'] = normalize_tags(e.get('tags') or [], merge)
    # Counts before
    before = Counter(t for e in data for t in e.get('tags') or [])
    n = len(data)
    ubiquitous = {t for t, c in before.items() if drop_ubiquitous and c == n}
    low = {t for t, c in before.items() if c < min_count}

    changed = 0
    removed_per_id: Dict[str, List[str]] = {}
    for e in data:
        old = list(e.get('tags') or [])
        new = [t for t in old if t not in ubiquitous and t not in low]
        if not new:
            new = [fallback_tag]
        if new != old:
            e['tags'] = new
            changed += 1
            removed = [t for t in old if t not in new]
            if removed:
                removed_per_id[str(e.get('id'))] = removed
    after = Counter(t for e in data for t in e.get('tags') or [])
    return data, {
        'entries': n,
        'ubiquitous_removed': sorted(list(ubiquitous)),
        'low_removed': {t: before[t] for t in sorted(low)},
        'changed_entries': changed,
        'before_counts': dict(before),
        'after_counts': dict(after),
        'removed_per_entry': removed_per_id,
    }


def write_report(path: Path, info: Dict[str, Any]):
    lines: List[str] = []
    lines.append(f"Entries: {info.get('entries')}")
    if info.get('ubiquitous_removed'):
        lines.append('Ubiquitous removed: ' + ', '.join(info['ubiquitous_removed']))
    if info.get('low_removed'):
        low = ', '.join([f"{k}:{v}" for k, v in info['low_removed'].items()])
        lines.append('Low-frequency removed (<min): ' + low)
    # Top after
    after_counts = info.get('after_counts', {})
    top_after = sorted(after_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:20]
    lines.append('Top labels after: ' + ', '.join([f"{k}:{v}" for k, v in top_after]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def append_jsonl(path: Path, obj: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False) + '\n')


def parse_merge_args(pairs: List[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for p in pairs or []:
        if '=' in p:
            a, b = p.split('=', 1)
            a = a.strip().lower()
            b = b.strip().lower()
            if a and b:
                mapping[a] = b
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', default='faq.json')
    ap.add_argument('--out', dest='out', default='faq.cleaned.json')
    ap.add_argument('--min_count', type=int, default=5)
    ap.add_argument('--keep_ubiquitous', action='store_true')
    ap.add_argument('--merge', action='append', help='Pairs like from=to; can be repeated')
    ap.add_argument('--report', default=str(Path('models') / 'faq_clean_report.txt'))
    ap.add_argument('--log', default=str(Path('logs') / 'cleaning.jsonl'))
    args = ap.parse_args()

    inp = Path(args.inp)
    outp = Path(args.out)
    report = Path(args.report)
    log = Path(args.log)

    merge_map = dict(DEFAULT_MERGE)
    merge_map.update(parse_merge_args(args.merge))

    data = load_faq(inp)
    cleaned, info = rebalance(
        data,
        min_count=args.min_count,
        drop_ubiquitous=not args.keep_ubiquitous,
        merge=merge_map,
    )
    save_faq(outp, cleaned)
    write_report(report, info)
    append_jsonl(log, {'action': 'rebalance', 'info': info, 'in': str(inp), 'out': str(outp), 'min_count': args.min_count})
    print(f"Rebalanced dataset written to {outp}. Changed entries: {info['changed_entries']}")


if __name__ == '__main__':
    main()

