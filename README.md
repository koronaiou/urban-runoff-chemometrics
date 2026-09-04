# Urban Runoff as a Vector of Emerging Contaminants: Chemical Fingerprinting via Advanced HRMS Workflow and Machine Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

<img width="1328" height="531" alt="graphical abstract" src="https://github.com/user-attachments/assets/47385a94-b3d5-4a32-8849-454ac25e7813" />

Complete analytical workflow for a non-target screening study of urban runoff in
Thessaloniki, Greece: spectral library preparation, and an explainable
machine-learning pipeline testing whether a sample's sampling location can be
predicted from its chemical profile.

The supervised analysis is a **chemical fingerprinting framework**, not a source
apportionment tool. With 40 samples across four locations the dataset is below
the size at which classification supports confident inference about individual
sites, and no source attribution is derived from model output. What the workflow
delivers is a validated procedure, transferable to larger campaigns, together
with compounds whose association with location is reproducible across
independent rainfall events.

---

## Study design

| Parameter | Details |
|---|---|
| Location | Thessaloniki, Central Macedonia, Greece |
| Sampling points | 4 urban catchments |
| Samples | 40 (10 per location) |
| Sampling occasions | 10 wet-weather days, November–December 2024, in 4 rain episodes |
| Analysis | LC–high-resolution mass spectrometry, non-target screening |
| Compounds | 175 identified at Schymanski confidence levels 1, 2a and 2b |

### Sampling locations and classification performance

Recall is the proportion of a location's ten samples correctly assigned to it,
from out-of-fold predictions under event-blocked cross-validation. Chance = 0.25.

| Location | Character | Recall | 95% CI |
|---|---|---|---|
| Lagkada Str | Industrial/residential interface | 0.80 | 0.50–0.96 |
| Agias Sofias Str | Urban commercial district | 0.70 | 0.39–0.91 |
| Ethnikis Aminis Str | Residential/commercial mixed | 0.30 | 0.09–0.61 |
| Thessaloniki Port | Maritime industrial zone | 0.20 | 0.04–0.50 |

Only Lagkada and Agias Sofias exceed chance; of six pairwise comparisons only
Lagkada versus the Port differs significantly (Fisher exact p = 0.023). The
supported contrast is between the two better-recovered and the two chance-level
locations, not between individual sites.

---

## Part A — Spectral library preparation

Three notebooks in `library_preparation/` build the dual-library architecture
(experimental plus in-silico) used for compound annotation in Compound
Discoverer. They are run once, before the analysis pipeline; outputs are
imported into mzVault.

### 1. MoNA database curation — `1_MoNA-database-curation.ipynb`

Curates experimental MS/MS spectra from MassBank of North America so that only
high-quality HRMS spectra support Level 2a identifications.

- **Input:** MoNA MSP export
- **Processing:** spectrum quality filtering (≥3 peaks, MS² required); decimal
  precision filter retaining spectra where ≥90% of fragment peaks carry ≥4
  decimal places in m/z; precursor m/z extraction from comments (supports
  [M+H]⁺, [M−H]⁻ and other adducts); metadata promotion (InChIKey, SMILES,
  retention time)
- **Output:** `MoNAcuratedwithprecursors.msp`

### 2. NORMAN database curation — `2_norman_database_curation.ipynb`

Filters and standardises the NORMAN Suspect Database into a focused, MS-ready
suspect list of urban-relevant contaminants.

- **Input:** NORMAN SusDat, filtered by 14 use categories (biocides, PFAS,
  industrial chemicals, PPCPs, plastic additives and others)
- **Processing:** RDKit SMILES standardisation (salt removal, fragment
  selection, neutralisation, tautomer canonicalisation); molecular weight filter
  100–1000 Da; duplicate removal on canonical SMILES
- **Output:** `CuratedSuspectList.csv`

### 3. In-silico library generation — `3_in-silico-libraries.ipynb`

Matches experimental precursors against NORMAN suspects and formats predicted
MS/MS spectra for mzVault.

1. Extract MS² precursor m/z values from experimental `.mzML` files
2. Match against the curated suspect list at 5 ppm tolerance
3. Export matched suspects as SMILES for CFM-ID
4. Run CFM-ID batch prediction at 10, 20 and 40 eV (**external tool, not
   included** — see CFM-ID documentation)
5. Parse CFM-ID output into MSP libraries

- **Output:** `cfmidpredicted_10eV_PI.msp` and equivalents per collision energy
  and ionisation mode

---

## Part B — Machine learning pipeline

### Data

| File | Contents |
|---|---|
| `CL_1_2a_2b_identifications.xlsx` | 40 × 176 — sample key and peak areas for 175 compounds. Zeros are non-detects. |
| `sample_details_rain_days.xlsx` | 40 × 4 — sample name, sampling point, date, day of rain |
| `vip16_dual_criterion.csv` | The 16 markers meeting VIP > 1 and site-attributable variance > 20%, used for the concordance test |

### Code

| File | Role |
|---|---|
| `run_case_study.py` | Entry point — runs the full analysis, writes all figures and tables |
| `nested_selection.py` | Cross-validation engine: search spaces, event-blocked splitters, nested selection, permutation test |
| `preprocessing.py` | Fold-internal implementation of the intensity preparation sequence |
| `data_prep.py` | Loading, joining, construction of grouping variables |
| `compound_annotation.py` | Manual source-class assignment for all 175 compounds; independent of the data |
| `per_site_report.py` | Per-location recall, precision, F1, Jeffreys intervals, Fisher exact tests |
| `marker_report.py` | Marker compound table and source-class enrichment statistics |

### Installation

```bash
git clone https://github.com/koronaiou/urban-water-runoff.git
cd urban-water-runoff
pip install -r requirements.txt
```

Python 3.10 or later; developed on 3.13.2. Verify with
`python run_case_study.py --quick` (about 2 minutes). Ignore the numbers it
prints — that reduced configuration is noisy and is only a functional check.

### Reproducing the published results

**1. Main analysis** — approximately 17 min on 4 cores.

```bash
python run_case_study.py --seeds 10 --trials 40 --perm 300 --jobs 4 \
       --vip-csv vip16_dual_criterion.csv
```

**2. Permutation test** — re-runs the full nested selection within every
permutation, so the null includes the cost of algorithm selection. About 2 h.

```bash
python run_case_study.py --seeds 2 --trials 10 --perm 300 --perm-mode nested \
       --jobs 4 --vip-csv vip16_dual_criterion.csv \
       --outdir results_ml_nested --figdir figures_ml_nested
```

**3. Derived tables** — seconds.

```bash
python per_site_report.py --outdir results_ml
python marker_report.py   --outdir results_ml
```

### Where each output comes from

Everything is taken from `results_ml/` and `figures_ml/` **except the permutation
test**, which comes from the `_nested` directories. A same-named permutation
figure exists in both; only the `_nested` version corresponds to the reported
p-value.

| Item | File | Directory |
|---|---|---|
| Algorithm stability | `FigML1_algorithm_stability.png` | `figures_ml/` |
| Per-site performance | `FigML2_site_performance.png` | `figures_ml/` |
| **Permutation test** | `FigML3_permutation.png` | **`figures_ml_nested/`** |
| SHAP summary | `FigML4_shap_beeswarm_target.png` | `figures_ml/` |
| Local SHAP | `FigML5_shap_local_target.png` | `figures_ml/` |
| VIP–SHAP concordance | `FigML6_vip_shap_convergence.png` | `figures_ml/` |
| Overall performance | `table_ml1_per_seed_metrics.csv` | `results_ml/` |
| Algorithm and k per fold | `table_ml2_fold_selections.csv` | `results_ml/` |
| Per-location metrics | `table_ml8_per_site_full.csv` | `results_ml/` |
| Marker compounds | `table_S13_marker_compounds.csv` | `results_ml/` |
| Source-class enrichment | `marker_enrichment.csv` | `results_ml/` |
| Concordance | `table_ml7_vip_shap_convergence.csv` | `results_ml/` |
| **Permutation p-value** | `permutation_summary.json` | **`results_ml_nested/`** |

---

## Methodology

### Validation design

| Component | Setting |
|---|---|
| Outer CV | Leave-one-event-out, blocked by sampling date (10 folds; 36 train / 4 test) |
| Inner CV | 3-fold stratified group CV on the training partition, same blocking |
| Repetitions | 10 random seeds |
| Optimisation | Optuna TPE, 40 trials per outer fold |
| Baseline | Majority-class classifier fitted per fold (balanced accuracy 0.250) |

### Intensity preparation

Occurrence filter (detected in ≥10% of training samples) → half-minimum
imputation → log₁₀ → max-scaling to [0, 1]. Every data-derived quantity is
estimated within the training partition only.

### Candidate algorithms

Random forest, LightGBM, XGBoost, CatBoost, linear SVC and multinomial logistic
regression. The algorithm is **not fixed in advance**: it is selected inside each
training partition alongside the retained feature count and the hyperparameters.

### Headline results

| Metric | Value | Baseline |
|---|---|---|
| Balanced accuracy | 0.480 ± 0.037 | 0.250 |
| Macro F1 | 0.459 ± 0.043 | 0.100 |
| MCC | 0.314 ± 0.047 | 0.000 |
| Permutation test | p = 0.0100 (300 permutations, nested null) | — |
| VIP–SHAP concordance | 15 of 16 markers retained (p = 6.3 × 10⁻¹³) | — |

Across 100 fold-level decisions no algorithm predominated (LinearSVC 32%,
logistic regression 22%, CatBoost 18%, random forest 12%, LightGBM 10%,
XGBoost 6%), indicating that algorithm ranking is not resolvable at this sample
size. The framework transfers; its outcome must be re-determined on new data.

---

## Design notes

**Preprocessing is fitted inside each training fold.** The occurrence filter,
imputation constants and scaling maxima are all estimated from data. Computing
them once across all 40 samples suits a descriptive ordination but would let
withheld samples inform the model evaluated on them.

**Folds are blocked by rainfall event.** The 40 samples are repeated measurements
of four fixed locations across sequential events and are not independent. Each
fold withholds one complete sampling occasion; the inner CV uses the same
blocking.

**The algorithm is selected inside the training folds.** Running several
algorithms and reporting the best returns the maximum of correlated estimates,
which is upwardly biased. Algorithm identity is instead one more decision taken
by the inner search, so the reported score estimates the whole selection
procedure.

**Attributions are deterministic.** The permutation SHAP estimator is seeded per
fold and repetition, so repeated runs reproduce identical values, not merely
identical rankings.

## Interpreting the output

The dispersion across repetitions (± 0.037) measures procedural stability, not
the uncertainty of the estimate; the latter is governed by ten samples per class
and is substantially larger. Significance comes from the permutation test.

The size of the marker set depends on the selection-stability threshold.
`marker_report.py` prints the count at every threshold; state the threshold
wherever the count is quoted.

Highly ranked compounds are those that vary between locations, which is not the
same as those that are abundant or hazardous. Ubiquitous compounds — including
the tyre- and road-wear markers that dominate the inventory — are uninformative
for discrimination by construction.

---

## References

1. Schymanski, E.L. et al. (2014). Identifying small molecules via high
   resolution mass spectrometry: communicating confidence. *Environ. Sci.
   Technol.* 48, 2097–2098.
2. Varma, S., Simon, R. (2006). Bias in error estimation when using
   cross-validation for model selection. *BMC Bioinformatics* 7, 91.
3. Cawley, G.C., Talbot, N.L.C. (2010). On over-fitting in model selection and
   subsequent selection bias in performance evaluation. *J. Mach. Learn. Res.*
   11, 2079–2107.
4. Roberts, D.R. et al. (2017). Cross-validation strategies for data with
   temporal, spatial, hierarchical, or phylogenetic structure. *Ecography* 40,
   913–929.
5. Lundberg, S.M., Lee, S.-I. (2017). A unified approach to interpreting model
   predictions. *Adv. Neural Inf. Process. Syst.* 30.
6. Akiba, T. et al. (2019). Optuna: a next-generation hyperparameter
   optimization framework. *Proc. 25th ACM SIGKDD*, 2623–2631.

---

## Licence

Code is released under the MIT Licence — see `LICENSE`. Data files are released
under [CC BY 4.0 / as specified by the corresponding author].

## Citation

If you use this code or these data, please cite:

> Koronaiou, L.-A., Alampanos, V., Eftimov, T., Abrahamsson, D., Lambropoulou,
> D.A. Urban Runoff as a Vector of Emerging Contaminants: Chemical
> Fingerprinting via Advanced HRMS Workflow and Machine Learning.

## Contact

Corresponding author: Dr D. Lambropoulou — dlambro@chem.auth.gr
Tel: +30 2310 997687 · Fax: +30 2310 997799
