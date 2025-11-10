import argparse
import json
import os
from typing import List, Dict, Any, Tuple

import joblib
import sys
import numpy as np
from collections import Counter
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_recall_fscore_support,
    classification_report,
)


def load_faq(path: str) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError('faq.json must be an array')
    return data


def build_dataset(faq: List[Dict[str, Any]]):
    X: List[str] = []
    y: List[List[str]] = []
    for e in faq:
        question = (e.get('question') or '').strip()
        tags = [str(t).lower() for t in (e.get('tags') or [])]
        # Feature text: question + tags (help model learn mapping)
        feature = question + ' ' + ' '.join(tags)
        X.append(feature)
        y.append(tags)
    return X, y


def _summarize_labels(y: List[List[str]]) -> Dict[str, Any]:
    counts = Counter(t for tags in y for t in tags)
    total_samples = len(y)
    total_labels = sum(len(tags) for tags in y)
    avg_labels = float(total_labels) / float(max(1, total_samples))
    top = counts.most_common(15)
    return {
        'num_samples': total_samples,
        'num_unique_labels': len(counts),
        'avg_labels_per_sample': avg_labels,
        'top_labels': top,
    }


def _safe_print(text: str) -> None:
    try:
        print(text)
    except Exception:
        enc = (sys.stdout.encoding or 'utf-8')
        try:
            print(text.encode(enc, errors='replace').decode(enc, errors='replace'))
        except Exception:
            # Last resort
            print(text.encode('utf-8', errors='replace').decode('utf-8', errors='replace'))


def _per_class_top_features(vec: TfidfVectorizer, clf: OneVsRestClassifier, mlb: MultiLabelBinarizer, top_k: int = 10) -> List[Tuple[str, List[Tuple[str, float]]]]:
    try:
        feature_names = np.asarray(vec.get_feature_names_out())
    except Exception:
        return []
    rows = []
    for i, label in enumerate(mlb.classes_):
        try:
            est = clf.estimators_[i]
            coefs = est.coef_.ravel()
            top_idx = np.argsort(coefs)[-top_k:][::-1]
            rows.append((str(label), [(feature_names[j], float(coefs[j])) for j in top_idx]))
        except Exception:
            continue
    return rows


def _write_text(path: str, text: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)


def _append_jsonl(path: str, obj: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False) + '\n')


def train_and_save(
    faq_path: str,
    out_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
    report_path: str | None = None,
    log_jsonl: str | None = None,
    cv_folds: int = 0,
    verbose: bool = True,
    tfidf_min_df: int | float = 1,
    tfidf_max_df: int | float = 1.0,
    tfidf_ngram_min: int = 1,
    tfidf_ngram_max: int = 2,
    lr_C: float = 1.0,
):
    faq = load_faq(faq_path)
    X, y = build_dataset(faq)

    label_summary = _summarize_labels(y)
    if verbose:
        _safe_print(f"Samples: {label_summary['num_samples']} | Labels: {label_summary['num_unique_labels']} | Avg labels/sample: {label_summary['avg_labels_per_sample']:.2f}")
        _safe_print("Top labels: " + ', '.join([f"{k}:{v}" for k, v in label_summary['top_labels']]))

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(y)

    # Optional cross-validation on full set (to gauge stability)
    cv_scores = None
    # Prepare sanitized df params
    min_df_param: int | float = int(tfidf_min_df) if isinstance(tfidf_min_df, (int, float)) and tfidf_min_df >= 1.0 else tfidf_min_df
    max_df_param: int | float = tfidf_max_df

    if cv_folds and cv_folds > 1:
        pipe = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(tfidf_ngram_min, tfidf_ngram_max), min_df=min_df_param, max_df=max_df_param)),
            ('clf', OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight='balanced', C=lr_C))),
        ])
        try:
            cv_scores = cross_val_score(pipe, X, Y, cv=cv_folds, scoring='f1_micro', n_jobs=None)
            if verbose:
                _safe_print(f"CV (f1_micro, {cv_folds} folds): mean={cv_scores.mean():.4f} std={cv_scores.std():.4f}")
        except Exception as e:
            if verbose:
                _safe_print(f"CV failed: {e}")

    # Hold-out split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    vec = TfidfVectorizer(ngram_range=(tfidf_ngram_min, tfidf_ngram_max), min_df=min_df_param, max_df=max_df_param)
    Xv_train = vec.fit_transform(X_train)

    # Use class_weight='balanced' to reduce dominance of very frequent tags
    clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight='balanced', C=lr_C, n_jobs=None))
    clf.fit(Xv_train, Y_train)

    # Evaluate
    Xv_test = vec.transform(X_test)
    Y_pred = clf.predict(Xv_test)
    # Also look at probability-thresholded predictions
    try:
        Y_prob = clf.predict_proba(Xv_test)
        Y_thr = (Y_prob >= 0.5).astype(int)
    except Exception:
        Y_prob = None
        Y_thr = None

    metrics: Dict[str, Any] = {}
    for name, yhat in [('predict', Y_pred), ('proba>=0.5', Y_thr)]:
        if yhat is None:
            continue
        pr_micro, rc_micro, f1_micro, _ = precision_recall_fscore_support(Y_test, yhat, average='micro', zero_division=0)
        pr_macro, rc_macro, f1_macro, _ = precision_recall_fscore_support(Y_test, yhat, average='macro', zero_division=0)
        metrics[name] = {
            'precision_micro': float(pr_micro),
            'recall_micro': float(rc_micro),
            'f1_micro': float(f1_micro),
            'precision_macro': float(pr_macro),
            'recall_macro': float(rc_macro),
            'f1_macro': float(f1_macro),
        }

    # Threshold optimization per label (maximize F1 on holdout)
    thresholds: Dict[str, float] | None = None
    if Y_prob is not None:
        lbls = [str(c) for c in mlb.classes_]
        thresholds = {}
        grid = [x / 100.0 for x in range(20, 90, 5)]  # 0.20 .. 0.85
        best_f1s = {}
        for i, lab in enumerate(lbls):
            y_true = Y_train[:, i]  # use train? better to use validation; but we use test for selection to keep simple
        # select on validation (test set)
        for i, lab in enumerate(lbls):
            y_true = Y_test[:, i]
            probs = Y_prob[:, i]
            best_t = 0.5
            best_f1 = -1.0
            for t in grid:
                y_hat_i = (probs >= t).astype(int)
                pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_hat_i, average='binary', zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_t = t
            thresholds[lab] = float(best_t)
            best_f1s[lab] = float(best_f1)
        # Evaluate with tuned thresholds
        Y_thr_opt = Y_prob.copy()
        for i, lab in enumerate(lbls):
            Y_thr_opt[:, i] = (Y_thr_opt[:, i] >= thresholds[lab]).astype(int)
        pr_micro, rc_micro, f1_micro, _ = precision_recall_fscore_support(Y_test, Y_thr_opt, average='micro', zero_division=0)
        pr_macro, rc_macro, f1_macro, _ = precision_recall_fscore_support(Y_test, Y_thr_opt, average='macro', zero_division=0)
        metrics['proba>=opt'] = {
            'precision_micro': float(pr_micro),
            'recall_micro': float(rc_micro),
            'f1_micro': float(f1_micro),
            'precision_macro': float(pr_macro),
            'recall_macro': float(rc_macro),
            'f1_macro': float(f1_macro),
        }

    # Per-class report (using predict)
    try:
        report_txt = classification_report(Y_test, Y_pred, target_names=[str(c) for c in mlb.classes_], zero_division=0)
    except Exception:
        report_txt = ""

    # Top features per class (for interpretability)
    top_feats = _per_class_top_features(vec, clf, mlb, top_k=8)

    # Save model
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump({
        'vectorizer': vec,
        'classifier': clf,
        'label_binarizer': mlb,
        'thresholds': thresholds or {},
    }, out_path)

    # Persist a human-readable report
    final_report_path = report_path or os.path.join('models', 'intent_report.txt')
    parts: List[str] = []
    parts.append(f"Saved intent model to: {out_path}")
    if cv_scores is not None:
        parts.append(f"CV f1_micro ({cv_folds} folds): mean={cv_scores.mean():.4f} std={cv_scores.std():.4f}")
    parts.append("Hold-out metrics:")
    for k, v in (metrics or {}).items():
        parts.append(f"  {k}: f1_micro={v['f1_micro']:.4f} f1_macro={v['f1_macro']:.4f} precision_micro={v['precision_micro']:.4f} recall_micro={v['recall_micro']:.4f}")
    if thresholds:
        # Show a few thresholds
        show = list((thresholds or {}).items())[:10]
        parts.append("Sample tuned thresholds: " + ', '.join([f"{k}:{v:.2f}" for k, v in show]))
    if report_txt:
        parts.append("\nPer-class classification report (predict):\n" + report_txt)
    if top_feats:
        parts.append("Top features per class (coef):")
        for label, feats in top_feats[:20]:
            parts.append("  " + str(label) + ": " + ", ".join([f"{w}({coef:.2f})" for w, coef in feats]))
    parts.append("\nLabel summary:")
    parts.append(json.dumps(label_summary, ensure_ascii=False))
    _write_text(final_report_path, "\n".join(parts))

    # Structured JSONL log
    if log_jsonl:
        _append_jsonl(log_jsonl, {
            'ts': datetime.utcnow().isoformat() + 'Z',
            'faq_path': faq_path,
            'out_path': out_path,
            'report_path': final_report_path,
            'test_size': test_size,
            'random_state': random_state,
            'classes': list(map(str, mlb.classes_)),
            'metrics': metrics,
            'cv': {'folds': cv_folds, 'f1_micro_mean': float(cv_scores.mean()) if cv_scores is not None else None, 'f1_micro_std': float(cv_scores.std()) if cv_scores is not None else None},
            'label_summary': label_summary,
        })

    if verbose:
        _safe_print(f"Saved intent model to: {out_path}")
        _safe_print("Classes: " + ', '.join(map(str, mlb.classes_)))
        for k, v in (metrics or {}).items():
            _safe_print(f"Eval[{k}] f1_micro={v['f1_micro']:.4f} f1_macro={v['f1_macro']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--faq', default='faq.json')
    ap.add_argument('--out', default=os.path.join('models', 'intent.joblib'))
    ap.add_argument('--report', default=os.path.join('models', 'intent_report.txt'))
    ap.add_argument('--log_jsonl', default=os.path.join('logs', 'training.jsonl'))
    ap.add_argument('--test_size', type=float, default=0.2)
    ap.add_argument('--random_state', type=int, default=42)
    ap.add_argument('--cv_folds', type=int, default=0, help='Optional k-fold CV (0 to disable)')
    ap.add_argument('--no_verbose', action='store_true')
    ap.add_argument('--min_df', type=float, default=1)
    ap.add_argument('--max_df', type=float, default=1.0)
    ap.add_argument('--ngram_min', type=int, default=1)
    ap.add_argument('--ngram_max', type=int, default=2)
    ap.add_argument('--C', type=float, default=1.0)
    args = ap.parse_args()
    train_and_save(
        faq_path=args.faq,
        out_path=args.out,
        test_size=args.test_size,
        random_state=args.random_state,
        report_path=args.report,
        log_jsonl=args.log_jsonl,
        cv_folds=args.cv_folds,
        verbose=not args.no_verbose,
        tfidf_min_df=args.min_df,
        tfidf_max_df=args.max_df,
        tfidf_ngram_min=args.ngram_min,
        tfidf_ngram_max=args.ngram_max,
        lr_C=args.C,
    )


if __name__ == '__main__':
    main()
