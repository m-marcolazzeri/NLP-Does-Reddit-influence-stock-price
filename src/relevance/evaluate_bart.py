#!/usr/bin/env python3
"""
evaluate_bart.py
────────────────
Calculates classification metrics for the BART zero-shot relevance classifier
by comparing its predictions against a hand-labeled evaluation set.

Usage
-----
    python src/relevance/evaluate_bart.py \
        --labels  data/relevance/rows_labeled_by_hand_3000.xlsx \
        --preds   data/relevance/relevance_predictions_v1.csv \
        --out     data/relevance/bart_eval_report.json

Inputs
------
  --labels  : Excel file with hand-labeled rows.
              Must contain columns: 'id', 'Relevance' (1 / 0 / -1).
  --preds   : CSV produced by predict_relevance_zeroshot.py (EVAL_MODE=False).
              Must contain columns: 'id', 'predicted_relevance'.
  --out     : (optional) path to save JSON report.

Label convention
----------------
  1   Relevant
  0   Borderline
 -1   Not relevant

Borderline rows are handled in two ways:
  - Strict:   0 → -1   (conservative: borderline = not confident → not relevant)
  - Liberal:  0 →  1   (lenient:      borderline = leaning relevant)
Both are reported so you can pick the view that fits your use case.
"""

from __future__ import annotations
import argparse, json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, cohen_kappa_score
)


def load_data(labels_path: Path, preds_path: Path) -> pd.DataFrame:
    labels = pd.read_excel(labels_path)[['id', 'Relevance']]
    preds  = pd.read_csv(preds_path, low_memory=False)[['id', 'matched_tickers',
                                                         'predicted_relevance',
                                                         'relevance_confidence']]
    df = labels.merge(preds, on='id', how='inner')
    print(f"[INFO] Matched {len(df):,} rows (labels={len(labels):,}, preds={len(preds):,})")
    missing = len(labels) - len(df)
    if missing:
        print(f"[WARN] {missing} labeled rows not found in predictions file (check IDs)")
    return df


def report_block(y_true, y_pred, title: str) -> dict:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)
    print(classification_report(y_true, y_pred,
          labels=[1, -1], target_names=['Relevant (1)', 'Not relevant (-1)'],
          zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[1, -1])
    cm_df = pd.DataFrame(cm,
                         index=['True  1', 'True -1'],
                         columns=['Pred  1', 'Pred -1'])
    print("Confusion matrix:")
    print(cm_df.to_string())
    kappa = cohen_kappa_score(y_true, y_pred)
    print(f"\nCohen's κ: {kappa:.4f}")

    from sklearn.metrics import precision_recall_fscore_support
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred,
                  labels=[1, -1], average='macro', zero_division=0)
    return dict(macro_precision=round(p,4), macro_recall=round(r,4),
                macro_f1=round(f,4), kappa=round(kappa,4),
                confusion_matrix=cm.tolist())


def main():
    parser = argparse.ArgumentParser(description="Evaluate BART relevance classifier")
    parser.add_argument('--labels', required=True, help='Hand-labeled Excel file')
    parser.add_argument('--preds',  required=True, help='BART predictions CSV')
    parser.add_argument('--out',    default=None,  help='JSON report output path')
    args = parser.parse_args()

    labels_path = Path(args.labels)
    preds_path  = Path(args.preds)

    df = load_data(labels_path, preds_path)

    print(f"\n[INFO] Label distribution (hand-labeled):")
    print(df['Relevance'].value_counts().sort_index().to_string())
    print(f"\n[INFO] Prediction distribution (BART):")
    print(df['predicted_relevance'].value_counts().sort_index().to_string())
    n_borderline = (df['Relevance'] == 0).sum()
    print(f"\n[INFO] Borderline rows in labels: {n_borderline} "
          f"({n_borderline/len(df)*100:.1f}%)")

    results = {}

    # ── Strict: borderline → -1 ──────────────────────────────────────────
    y_true_strict = df['Relevance'].replace(0, -1)
    y_pred_strict = df['predicted_relevance'].replace(0, -1)
    results['strict'] = report_block(y_true_strict, y_pred_strict,
                                     "STRICT  (borderline → -1)")

    # ── Liberal: borderline → 1 ──────────────────────────────────────────
    y_true_lib = df['Relevance'].replace(0, 1)
    y_pred_lib = df['predicted_relevance'].replace(0, 1)
    results['liberal'] = report_block(y_true_lib, y_pred_lib,
                                      "LIBERAL (borderline → 1)")

    # ── Per-stock breakdown ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  PER-STOCK BREAKDOWN  (strict, borderline → -1)")
    print('='*60)
    stock_results = {}
    for stock in sorted(df['matched_tickers'].dropna().unique()):
        sub = df[df['matched_tickers'] == stock]
        yt  = sub['Relevance'].replace(0, -1)
        yp  = sub['predicted_relevance'].replace(0, -1)
        from sklearn.metrics import f1_score
        f1 = f1_score(yt, yp, labels=[1, -1], average='macro', zero_division=0)
        acc = (yt == yp).mean()
        print(f"  {stock:6s}  n={len(sub):4d}  macro-F1={f1:.3f}  accuracy={acc:.3f}")
        stock_results[stock] = dict(n=len(sub), macro_f1=round(f1,4),
                                    accuracy=round(acc,4))
    results['per_stock'] = stock_results

    # ── Confidence analysis ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  CONFIDENCE ANALYSIS (BART confidence vs correctness)")
    print('='*60)
    y_true_s = df['Relevance'].replace(0, -1)
    y_pred_s = df['predicted_relevance'].replace(0, -1)
    df['correct'] = (y_true_s == y_pred_s)
    for bucket, lo, hi in [('low (<0.6)', 0, .6), ('mid (0.6-0.8)', .6, .8), ('high (>0.8)', .8, 1.01)]:
        mask = (df['relevance_confidence'] >= lo) & (df['relevance_confidence'] < hi)
        sub  = df[mask]
        if len(sub) == 0: continue
        acc  = sub['correct'].mean()
        print(f"  Confidence {bucket:15s}  n={len(sub):5d}  accuracy={acc:.3f}")

    # ── Save report ───────────────────────────────────────────────────────
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n[INFO] Report saved → {out_path}")

    print("\n[INFO] Done.")


if __name__ == '__main__':
    main()
