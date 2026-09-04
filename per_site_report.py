#!/usr/bin/env python3
"""
per_site_report.py
==================
Per-location performance table, computed from output already written by
run_case_study.py. No re-run required.

    python per_site_report.py                        # reads results_ml/
    python per_site_report.py --outdir results_ml_nested

Reports, for each sampling location:

  recall      of the samples truly from this site, the fraction assigned to it
              (row of the confusion matrix). This is the quantity most people
              mean by "accuracy for this location".
  precision   of the samples the model assigned to this site, the fraction that
              really were (column of the confusion matrix).
  F1          harmonic mean of the two; low unless both are reasonable.
  support     number of samples truly from this site.

Deliberately NOT reported: one-vs-rest "accuracy", (TP+TN)/N. With four classes
the true negatives dominate, so a site the model never predicts still scores
0.75. It is a misleading number and should not appear in a manuscript.

All intervals are 95% Jeffreys (Beta) intervals on a binomial proportion with
the stated denominator. With ten samples per class they are wide by necessity;
that width is a property of the study design, not of the analysis.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import scipy.stats as st


def jeffreys(k, n, conf=0.95):
    """95% Jeffreys interval for k successes out of n."""
    if n == 0:
        return (np.nan, np.nan)
    return st.beta.interval(conf, k + 0.5, n - k + 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results_ml")
    args = ap.parse_args()

    cm_path = os.path.join(args.outdir, "table_ml4_confusion_modal.csv")
    rec_path = os.path.join(args.outdir, "table_ml3_per_site_recall.csv")
    if not os.path.exists(cm_path):
        raise SystemExit(f"not found: {cm_path}\nRun run_case_study.py first.")

    cm = pd.read_csv(cm_path, index_col=0)
    labels = list(cm.index)
    M = cm.values.astype(int)

    rows = []
    for i, lab in enumerate(labels):
        tp = M[i, i]
        support = M[i, :].sum()          # truly this site
        predicted = M[:, i].sum()        # called this site
        recall = tp / support if support else np.nan
        precision = tp / predicted if predicted else np.nan
        f1 = (2 * precision * recall / (precision + recall)
              if predicted and support and (precision + recall) > 0 else 0.0)
        rlo, rhi = jeffreys(tp, support)
        plo, phi = jeffreys(tp, predicted) if predicted else (np.nan, np.nan)
        rows.append({
            "site": lab, "support": support, "n_predicted": predicted,
            "correct": tp,
            "recall": recall, "recall_ci_low": rlo, "recall_ci_high": rhi,
            "precision": precision, "precision_ci_low": plo, "precision_ci_high": phi,
            "f1": f1,
        })
    t = pd.DataFrame(rows)

    # seed-to-seed stability of recall, if the multi-seed table is present
    if os.path.exists(rec_path):
        sd = pd.read_csv(rec_path)[["site", "sd_recall"]]
        t = t.merge(sd, on="site", how="left")

    out = os.path.join(args.outdir, "table_ml8_per_site_full.csv")
    t.to_csv(out, index=False)

    print("\nPer-location performance (modal out-of-fold prediction)")
    print("=" * 78)
    disp = t[["site", "support", "correct", "recall", "recall_ci_low",
              "recall_ci_high", "precision", "f1"]]
    print(disp.round(3).to_string(index=False))

    chance = 1.0 / len(labels)
    print(f"\nChance level for {len(labels)} balanced classes: {chance:.3f}")
    print("\nSites whose 95% recall interval EXCLUDES chance "
          "(i.e. recovered better than guessing):")
    any_above = False
    for _, r in t.iterrows():
        if r["recall_ci_low"] > chance:
            print(f"    {r['site']}  recall {r['recall']:.2f} "
                  f"[{r['recall_ci_low']:.2f}-{r['recall_ci_high']:.2f}]")
            any_above = True
    if not any_above:
        print("    none - every interval includes the chance level")

    print("\nPairwise recall comparison (Fisher exact on correct/incorrect counts):")
    print("  Overlapping intervals or p > 0.05 means the two locations are NOT")
    print("  distinguishable; do not claim one classifies better than the other.")
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = t.iloc[i], t.iloc[j]
            table = [[a["correct"], a["support"] - a["correct"]],
                     [b["correct"], b["support"] - b["correct"]]]
            p = st.fisher_exact(table)[1]
            verdict = "DIFFERENT" if p < 0.05 else "not distinguishable"
            print(f"    {labels[i].split(' - ')[0]} vs {labels[j].split(' - ')[0]}: "
                  f"{a['recall']:.2f} vs {b['recall']:.2f}  p={p:.3f}  {verdict}")

    print(f"\nWritten: {out}")
    print("\nNote: one-vs-rest 'accuracy' is deliberately omitted. With four "
          "classes\nit is dominated by true negatives and would score ~0.75 even "
          "for a site\nthe model never predicts.")


if __name__ == "__main__":
    main()
