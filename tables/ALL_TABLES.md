**Table 1. Dataset composition after temporal matching.**

| Ecosystem | Versions | Malicious | Benign | Malicious % |
|---|---|---|---|---|
| NPM | 2505 | 2092 | 413 | 83.5% |
| PYPI | 825 | 79 | 746 | 9.6% |
| Total | 3330 | 2171 | 1159 | 65.2% |


---

**Table 1b. Partition sizes. Partitioning operates on group identifiers, so sizes are not exact fractions of the dataset.**

| Partition | Versions | Malicious | Benign |
|---|---|---|---|
| Training | 1713 | 1012 | 701 |
| Validation | 649 | 468 | 181 |
| Test | 968 | 691 | 277 |


---

**Table 2. Feature space reduction: 456 to 18 attributes (96.1% reduction).**

| Stage | Features in | Features out | Removed |
|---|---|---|---|
| Extracted point-in-time attributes | - | 456 | - |
| Ecosystem pre-filter (vman scope) | 456 | 52 | 404 |
| Embedded selection, top-k (a priori) | 52 | 18 | 34 |


---

**Table 3. The 18 selected features, ordered by mean absolute SHAP value (CatBoost, test partition, n = 968).**

| # | Feature | Category | Definition | Mean |SHAP| |
|---|---|---|---|---|
| 1 | pit_has_repository | Identity & links | Repository field present at t_p (boolean) | 1.1487 |
| 2 | pit_author_email_domain_score | Authorship & maintainers | Ordinal score of the author email domain (organisational, free-mail, disposable, missing); local part discarded | 0.9058 |
| 3 | pit_has_install_hook | Build & install | Install, preinstall or postinstall hook declared (boolean) | 0.9047 |
| 4 | pit_days_since_first | Release history | Days between the first release and t_p | 0.8987 |
| 5 | pit_long_description_len | Description & keywords | Character length of the long description at t_p | 0.8905 |
| 6 | pit_n_maintainers | Authorship & maintainers | Number of maintainer accounts listed at t_p | 0.8365 |
| 7 | pit_has_engines | Build & install | Engines field present in the manifest (boolean) | 0.7964 |
| 8 | pit_n_keywords | Description & keywords | Number of declared keywords at t_p | 0.7753 |
| 9 | vman_author_len | Authorship & maintainers | Character length of the author field in the published manifest | 0.7519 |
| 10 | pit_median_gap_days | Release history | Median inter-release interval before t_p | 0.7014 |
| 11 | pit_ver_is_zero_major | Versioning | Version string begins with 0. (boolean) | 0.63 |
| 12 | pit_name_has_scope | Identity & links | Package name carries an organisation scope (boolean) | 0.5931 |
| 13 | pit_scripts_max_len | Build & install | Longest declared script, in characters | 0.4755 |
| 14 | pit_description_len | Description & keywords | Character length of the summary at t_p | 0.4007 |
| 15 | pit_std_gap_days | Release history | Standard deviation of inter-release intervals before t_p | 0.3787 |
| 16 | pit_n_dependencies | Dependencies | Declared runtime dependencies at t_p | 0.365 |
| 17 | vman_description_len | Description & keywords | Character length of the description in the published manifest | 0.2901 |
| 18 | pit_description_entropy | Description & keywords | Shannon entropy of the summary at t_p | 0.2222 |


---

**Table 4. Model performance on the test partition (n = 968), CatBoost with 18 features under the corrected protocol.**

| Metric | Value |
|---|---|
| AU-PR | 0.9918893902712611 |
| AU-PR 95% CI | [0.9886, 0.9945] |
| AU-PR baseline | 0.7138429752066116 |
| AU-PR lift | 0.27804641506464955 |
| AUC-ROC | 0.9766492343540204 |
| AUC-ROC 95% CI | [0.9676, 0.9841] |
| MCC | 0.8317255236468101 |
| F1 (positive class) | 0.9421613394216134 |
| Accuracy | 0.9214876033057852 |
| Precision | 0.9935794542536116 |
| Recall | 0.8958031837916064 |
| Specificity | 0.9855595667870036 |
| False-positive rate | 0.01444043321299639 |
| Brier score | 0.06755687920227792 |
| Confusion matrix (TN/FP/FN/TP) | 273 / 4 / 72 / 619 |


---

**Table 5. Alert quality at operational prevalence, threshold 0.500 (default 0.5 (validation-optimised threshold did not transfer)). A false-positive rate measured at zero on 277 benign instances is not zero; the upper95 rows apply the rule of three.**

| Prevalence | FPR estimate | Recall | FPR | PPV | Alerts per 10,000 | True positives per 10,000 | Missed per 10,000 |
|---|---|---|---|---|---|---|---|
| 1.00% | point | 0.8958 | 0.0144 | 0.3852 | 232.5406 | 89.5803 | 10.4197 |
| 1.00% | upper95 | 0.8958 | 0.0327 | 0.2165 | 413.6938 | 89.5803 | 10.4197 |
| 0.10% | point | 0.8958 | 0.0144 | 0.0585 | 153.218 | 8.958 | 1.042 |
| 0.10% | upper95 | 0.8958 | 0.0327 | 0.0267 | 336.018 | 8.958 | 1.042 |
| 0.01% | point | 0.8958 | 0.0144 | 0.0062 | 145.2857 | 0.8958 | 0.1042 |
| 0.01% | upper95 | 0.8958 | 0.0327 | 0.0027 | 328.2504 | 0.8958 | 0.1042 |


---

**Table 6. Gain decomposition, 15 random draws each. The pre-filter contributes +0.0716 in AU-PR and the selection +0.0094, against a 95% bootstrap interval width of 0.0059.**

| Feature set | AU-PR | AUC-ROC | Mean proxies per draw |
|---|---|---|---|
| Random 18 of 456 (full space) | 0.911 | 0.8253 | 15.8667 |
| Random 18 of 52 (post-filter) | 0.9825 |  | 0.0 |
| Selected 18 (ours) | 0.9919 | 0.9766 | 0.0 |


---

**Table 7. Performance as a function of feature-set size. Differences from k = 18 are compared with the bootstrap interval width (0.0059); k was declared a priori and this sweep is reported, not used for the choice.**

| k | AU-PR | AUC-ROC | Delta vs k=18 | Within CI |
|---|---|---|---|---|
| 5 | 0.9784 | 0.9485 | -0.0135 | False |
| 10 | 0.9883 | 0.9691 | -0.0036 | True |
| 15 | 0.9897 | 0.9702 | -0.0022 | True |
| 18 | 0.9919 | 0.9766 | 0.0 | True |
| 25 | 0.9917 | 0.9752 | -0.0002 | True |
| 40 | 0.9926 | 0.9778 | 0.0007 | True |


---

**Table 8. Paired comparison against alternative selectors at a matched budget of 18 features. Bonferroni correction across 3 comparisons. Positive deltas favour our selection.**

| Comparison | AU-PR (alt.) | Delta AU-PR | CI low | CI high | p (bootstrap, Bonf.) | b | c | p (McNemar, Bonf.) | Jaccard |
|---|---|---|---|---|---|---|---|---|---|
| Ours vs Mutual information | 0.9807 | 0.0111 | 0.0056 | 0.0178 | 0.0 | 86 | 17 | 0.0 | 0.3846 |
| Ours vs RF importance | 0.9888 | 0.0031 | 0.0012 | 0.0053 | 0.0 | 46 | 14 | 0.0001 | 0.5 |
| Ours vs L1 logistic | 0.993 | -0.001 | -0.0025 | 0.0003 | 0.381 | 4 | 23 | 0.0009 | 0.44 |


---

**Table 9. Metadata-family ablation. Deltas are compared with the bootstrap interval width (0.0059).**

| Family withheld | Features removed | AU-PR | AUC-ROC | Delta AU-PR |
|---|---|---|---|---|
| Release history | 3 | 0.9859 | 0.9611 | -0.006 |
| Description & keywords | 5 | 0.9877 | 0.9644 | -0.0041 |
| Identity & links | 2 | 0.9889 | 0.9698 | -0.003 |
| Build & install | 3 | 0.9896 | 0.9718 | -0.0023 |
| Authorship & maintainers | 3 | 0.991 | 0.975 | -0.0009 |
| Dependencies | 1 | 0.9911 | 0.9735 | -0.0008 |
| Versioning | 1 | 0.9911 | 0.9745 | -0.0008 |
| None | 0 | 0.9919 | 0.9766 | 0.0 |


---

**Table 10. Robustness. Reference without perturbation: AUC-ROC 0.9766, AU-PR 0.9919.**

| Test | Definition | AUC-ROC (median) | IQR | AU-PR (median) | IQR  | Runs |
|---|---|---|---|---|---|---|
| Label noise, 10% | 10% of training labels inverted uniformly at random | 0.9605 | 0.005 | 0.9858 | 0.0027 | 10 |
| Feature corruption, 20% | 20% of test values replaced by draws from the training marginal | 0.9418 | 0.0057 | 0.9756 | 0.0029 | 10 |


---

**Table 11. Cross-ecosystem transfer. A classifier using only the ecosystem of origin attains AUC-ROC 0.8036, so 63.7% of discrimination above chance is attributable to registry identity.**

| Direction | AUC-ROC | AU-PR | AU-PR baseline | n test | n malicious |
|---|---|---|---|---|---|
| npm->pypi | 0.5893 | 0.1129 | 0.0958 | 825 | 79 |
| pypi->npm | 0.5433 | 0.8741 | 0.8351 | 2505 | 2092 |
| Ecosystem of origin only (no metadata) | 0.8036 |  |  | 3330 | 2171 |
