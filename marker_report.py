#!/usr/bin/env python3
"""
marker_report.py
================
Builds the supplementary marker table and the compositional statistics quoted in
the Results section, from output already written by run_case_study.py.

    python marker_report.py
    python marker_report.py --outdir results_ml --stability 0.5

Produces
--------
table_S13_marker_compounds.csv   the retained marker set, annotated by source
                                 class and grouped into three source categories,
                                 ready for the supplementary information
marker_enrichment.csv            hypergeometric enrichment / depletion of every
                                 source class among the retained markers

The retained set is defined by a single explicit criterion: a compound must have
survived univariate filtering in at least `--stability` of the folds. Report that
threshold alongside the count, because the count depends on it: raising the
threshold to 1.0 (retained in every fold) yields a smaller and more conservative
set. The script prints the count at several thresholds so the sensitivity is
visible rather than hidden.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd
from scipy.stats import hypergeom

from compound_annotation import CLASS_LABEL, annotate

# Three source categories used in the Results text.
GROUP = {
    "PFAS": "Industrial / technical",
    "OPE": "Industrial / technical",
    "SURFACTANT": "Industrial / technical",
    "PLASTIC": "Industrial / technical",
    "INDUSTRIAL": "Industrial / technical",
    "CORROSION": "Industrial / technical",
    "TRWP": "Industrial / technical",
    "LIFESTYLE": "Consumer / lifestyle",
    "PPCP": "Consumer / lifestyle",
    "PESTICIDE": "Consumer / lifestyle",
    "NATURAL": "Biogenic / background",
    "UNKNOWN": "Unassigned",
}
GROUP_ORDER = ["Industrial / technical", "Consumer / lifestyle",
               "Biogenic / background", "Unassigned"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results_ml")
    ap.add_argument("--stability", type=float, default=0.5)
    ap.add_argument("--datadir", default=".")
    args = ap.parse_args()

    path = os.path.join(args.outdir, "table_ml6_shap_target_site.csv")
    if not os.path.exists(path):
        raise SystemExit(f"not found: {path}\nRun run_case_study.py first.")
    d = pd.read_csv(path)

    ident = pd.read_excel(os.path.join(
        args.datadir, "CL_1_2a_2b_identifications.xlsx"), nrows=1)
    all_compounds = [c for c in ident.columns if c != "Sample"]
    all_classes, _, _ = annotate(all_compounds)
    N = len(all_compounds)
    full = pd.Series(all_classes).value_counts()

    print("Sensitivity of the marker count to the stability threshold:")
    for t in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
        print(f"    retained in >= {t*100:3.0f}% of folds : "
              f"{int((d.selection_stability >= t - 1e-9).sum()):3d} compounds")

    sel = d[d.selection_stability >= args.stability - 1e-9].copy()
    sel = sel.sort_values("mean_abs_shap_target", ascending=False)
    cls, prio, notes = annotate(sel["compound"].tolist())
    sel["source_class"] = [CLASS_LABEL[c] for c in cls]
    sel["source_category"] = [GROUP[c] for c in cls]
    sel["reported_use"] = notes
    sel["priority_contaminant"] = prio
    n = len(sel)

    out = sel[["compound", "source_category", "source_class", "reported_use",
               "priority_contaminant", "selection_stability",
               "mean_abs_shap_target"]]
    p_tab = os.path.join(args.outdir, "table_S13_marker_compounds.csv")
    out.to_csv(p_tab, index=False)

    print(f"\nRetained marker set at stability >= {args.stability}: {n} compounds")
    print("\nPartition into source categories:")
    g = pd.Series([GROUP[c] for c in cls]).value_counts()
    gf = pd.Series([GROUP[c] for c in all_classes]).value_counts()
    for k in GROUP_ORDER:
        print(f"    {k:24s} {int(g.get(k,0)):2d} / {n}  "
              f"({g.get(k,0)/n*100:4.1f}%)    dataset: "
              f"{int(gf.get(k,0))}/{N} ({gf.get(k,0)/N*100:.1f}%)")

    rows = []
    for c in full.index:
        K = int(full[c])
        k = int(sum(1 for x in cls if x == c))
        exp = n * K / N
        pe = hypergeom.sf(k - 1, N, K, n) if k > 0 else 1.0
        pdp = hypergeom.cdf(k, N, K, n)
        rows.append({"source_class": CLASS_LABEL[c], "in_dataset": K,
                     "in_markers": k, "expected": exp,
                     "fold_change": k / exp if exp else 0.0,
                     "p_enrichment": pe, "p_depletion": pdp})
    e = pd.DataFrame(rows).sort_values("p_enrichment")
    p_enr = os.path.join(args.outdir, "marker_enrichment.csv")
    e.to_csv(p_enr, index=False)

    print("\nSource-class enrichment among the markers "
          f"(hypergeometric, N={N}, draw={n}):")
    print(e.round(4).to_string(index=False))
    print("\nSignificant at p < 0.05:")
    for _, r in e.iterrows():
        if r.p_enrichment < .05:
            print(f"    ENRICHED  {r.source_class}: {int(r.in_markers)} observed vs "
                  f"{r.expected:.2f} expected, {r.fold_change:.1f}-fold, "
                  f"p = {r.p_enrichment:.4f}")
        elif r.p_depletion < .05:
            print(f"    DEPLETED  {r.source_class}: {int(r.in_markers)} observed vs "
                  f"{r.expected:.2f} expected, p = {r.p_depletion:.4f}")

    print(f"\nWritten: {p_tab}\n         {p_enr}")
    print("\nState the stability threshold wherever the marker count is quoted; "
          "the count\ndepends on it, as the sensitivity listing above shows.")


if __name__ == "__main__":
    main()
