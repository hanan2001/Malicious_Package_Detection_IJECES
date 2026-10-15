# Replication Package — IJECES-2026-5223

Replication package for the paper:

**"Explainable Detection of Malicious Packages Using Point-in-Time Metadata"**
H. Moufid, M. El Ghazouani, M. A. El Kiram
*International Journal of Electrical and Computer Engineering Systems* (under review).

Every numerical value reported in the manuscript is produced by the single
script in this repository and is traceable to `Results/final_results.json`.

---

## Repository contents

| Path | Description |
|---|---|
| `final_pipeline_ijeces.py` | Main script. Reproduces every table and figure of the paper. |
| `Notebook/final_pipeline_ijeces.ipynb` | The same pipeline as a Colab notebook. |
| `Dataset/dataset_v5.csv` | Point-in-time metadata for the 3,330 package versions used in the paper. |
| `Tables/` | Tables 1a–11 as CSV, plus `ALL_TABLES.md` (Markdown, ready to paste). |
| `Figures/` | Figures 2–6 in PNG (600 dpi), PDF and SVG. |
| `Results/final_results.json` | Every reported value, in one file. |
| `Results/MANUSCRIPT_VALUES.md` | Maps each table and figure to the section of the paper that cites it. |
| `requirements.txt` | Python dependencies. |

---

## How to run

The script does **not** download anything: the dataset is included in this
repository under `Dataset/`.

### Locally

```bash
git clone https://github.com/hanan2001/Malicious_Package_Detection_IJECES.git
cd Malicious_Package_Detection_IJECES
pip install -r requirements.txt
python final_pipeline_ijeces.py
```

The script reads `Dataset/dataset_v5.csv` and writes its outputs to
`./IJECES_revision` (tables, figures, `final_results.json`). Both locations can be
changed at the top of the script.

### In Google Colab

Open `Notebook/final_pipeline_ijeces.ipynb`, upload `Dataset/dataset_v5.csv`
(or mount the drive folder that contains it), set the input path in the
configuration block, then run all cells.

Runtime is about 15 minutes on CPU. The random seed is fixed (`SEED = 42`).

### Journal-style figures

`make_figures_journal.py` redraws Figs. 1–6 from the pipeline outputs
(`Results/final_results.json`, `Tables/table03_features.csv`,
`Results/shap_values.npz`) without changing any value: IJECES column widths
(88 / 180 mm), 7–8 pt sans-serif text, colour-blind-safe palette, vector PDF/SVG
and 600 dpi PNG/TIFF in `Figures/journal/`.

```bash
python make_figures_journal.py
```

### Reproducibility check

The reported values were generated on 15 August 2026 (`Results/final_results.json`).
Re-running the script on `Dataset/dataset_v5.csv` with Python 3.10,
scikit-learn 1.3.2, CatBoost 1.2.5 and SHAP 0.44.1 (the versions pinned in
`requirements.txt`) reproduces Tables 1a, 1b, 2, 4, 6, 7, 8, 9 and 10 and the
calibration metrics exactly. Mean |SHAP| values (Table 3) differ by at most
0.0005, the cross-ecosystem transfer values (Table 11) by at most 0.0014, and
the random-forest comparison of Table 5a by at most 0.0002 in AU-PR and one
discordant pair in the McNemar counts; these differences come from library
versions and do not change any reported conclusion.

---

## Mapping between output files and the paper

File names follow the generation script's internal numbering, which does not
always coincide with the numbering in the paper. The correspondence is:

### Tables

| Paper | File in `Tables/` |
|---|---|
| Table 1a — Dataset composition | `table01_dataset.csv` |
| Table 1b — Partition sizes | `table01b_partitions.csv` |
| Table 2 — Feature space reduction | `table02_reduction.csv` |
| Table 3 — The 18 selected features | `table03_features.csv` |
| Table 4 — Model performance on the test partition | `table04_performance.csv` |
| Table 5a — Paired comparison against alternative selectors | `table08_paired_tests.csv` |
| Table 5b — Published systems | *(literature values; not generated)* |
| Table 6 — Gain decomposition | `table06_gain_decomposition.csv` |
| Table 7 — Feature-set size sweep | `table07_k_sweep.csv` |
| Table 8 — Metadata-family ablation | `table09_family_ablation.csv` |
| Table 9 — Robustness to perturbations | `table10_robustness.csv` |
| Table 10 — Alert quality at operational prevalence | `table05_prevalence.csv` |
| Table 11 — Cross-ecosystem transfer | `table11_cross_ecosystem.csv` |

### Figures

| Paper | File in `Figures/` |
|---|---|
| Fig. 1 — Pipeline overview | *(schematic; not generated)* |
| Fig. 2 — Distribution of selected features by category | `figure2_feature_categories.*` |
| Fig. 3 — Performance versus feature-set size | `figure4_k_sweep.*` |
| Fig. 4 — Confusion matrix | `figure3_confusion_matrix.*` |
| Fig. 5 — SHAP summary plot | `figure5_shap_summary.*` |
| Fig. 6 — Reliability diagram | `figure6_calibration.*` |

---

## Experimental protocol

The pipeline enforces the leakage controls described in Section 2.3 of the
paper, in this order:

1. Group-aware partitioning into training, validation and test. Every version
   is assigned to a group by the transitive closure of shared package name,
   shared maintainer account, and name-similarity-family membership
   (single-linkage clustering of canonicalised names, normalised Levenshtein
   threshold 0.85). No group appears in more than one partition.
2. Imputation, winsorisation (1st/99th percentiles) and standardisation fitted
   on the training partition alone.
3. Ecosystem pre-filter computed on the training partition: attributes whose
   standard deviation is zero within at least one registry are removed
   (456 → 52).
4. Embedded selection on the training partition: CatBoost gain importance,
   top *k* = 18, with *k* fixed a priori (52 → 18).
5. Hyperparameter search by grouped five-fold cross-validation within the
   training partition.
6. Threshold: a threshold maximising validation PPV at 0.1% prevalence (with
   validation recall >= 0.70) is computed first; because its recall did not
   transfer (validation-test recall difference above the pre-declared
   tolerance of 0.20), all reported results use the default, untuned
   threshold of 0.5. This fallback check is the only point at which
   test-partition recall is consulted.
7. A single evaluation on the test partition.

**Class imbalance.** A single mechanism is used: CatBoost balanced class
weighting (`auto_class_weights="Balanced"`), for the selection model, the
grouped cross-validated hyperparameter search and the final model. No
resampling (undersampling or oversampling) is applied; validation and test
partitions keep their natural class distribution. The alternative selectors
(random forest, L1-logistic) use scikit-learn's `class_weight="balanced"`.

---

## Key reported values

| Quantity | Value |
|---|---|
| Versions / independent groups | 3,330 / 1,490 |
| Train / validation / test | 1,713 / 649 / 968 |
| Test confusion matrix (TN/FP/FN/TP) | 273 / 4 / 72 / 619 |
| AU-PR (95% CI) | 0.9919 (0.9886–0.9945) |
| AU-PR positive-class baseline | 0.7138 |
| AUC-ROC (95% CI) | 0.9766 (0.9676–0.9841) |
| MCC | 0.8317 |
| Brier score / ECE (10 equal-width bins) | 0.0676 / 0.1005 |
| PPV at 0.1% prevalence (FPR upper bound) | 2.67% (336 alerts per 10,000) |
| Random 18 of 52 post-filter attributes (AU-PR / AUC-ROC) | 0.9825 / 0.9598 |
| Ecosystem-only classifier AUC-ROC (all 3,330 versions) | 0.8036 |
| Decision threshold | 0.5 (default; see protocol step 6) |

---

## Dataset

`Dataset/dataset_v5.csv` contains one row per package version, with the
point-in-time metadata attributes and the columns `package`, `version`,
`ecosystem`, `t_p` (publication timestamp of the evaluated version) and
`label`.

A version is labelled malicious if and only if its advisory record attributes
at least one of: credential or token exfiltration, unauthorised network egress
at install time, arbitrary code execution through install hooks, cryptocurrency
wallet or clipboard tampering, or impersonation of a named target through
typosquatting or dependency confusion. Records without an attributed behaviour
are excluded.

Malicious advisories are drawn from the Open Source Security Foundation
malicious-packages corpus (https://github.com/ossf/malicious-packages), which
aggregates GitHub Security Advisories and Snyk, cross-checked against the
National Vulnerability Database. Benign versions are drawn from the NPM and
PyPI registry APIs over the same ecosystem-specific windows as the malicious
versions (22 March 2024 – 5 August 2026 for NPM; 10 January 2022 – 19 July 2026
for PyPI). Deduplication is by exact match on package name, version and
ecosystem.

---

## Citation

```bibtex
@article{moufid2026explainable,
  title   = {Explainable Detection of Malicious Packages Using Point-in-Time Metadata},
  author  = {Moufid, Hanan and El Ghazouani, Mohamed and El Kiram, Moulay Ahmed},
  journal = {International Journal of Electrical and Computer Engineering Systems},
  year    = {2026},
  note    = {Under review}
}
```

---


