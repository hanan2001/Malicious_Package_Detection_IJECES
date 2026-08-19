# Valeurs a reporter — jcp-4466772

Genere le 2026-08-15T18:58:58.871264+00:00. Toute valeur du manuscrit doit venir d'ici.

## Tables (dans /content/drive/MyDrive/JCP_revision/tables/, aussi en Markdown dans ALL_TABLES.md)

| Fichier | Section du manuscrit |
|---|---|
| table01_dataset, table01b_partitions | 2.3, Table 1 |
| table02_reduction | 2.4, Table 2 |
| table03_features | 3.1, Table 3 |
| table04_performance | 3.2, Table 4 |
| table05_prevalence | 3.6, Table 5 |
| table06_gain_decomposition | 3.2, Table 6 |
| table07_k_sweep | 3.2, Table 7 |
| table08_paired_tests | 3.2, Table 8 |
| table09_family_ablation | 3.3, Table 9 |
| table10_robustness | 3.5, Table 10 |
| table11_cross_ecosystem | 3.5, Table 11 |

## Figures (dans /content/drive/MyDrive/JCP_revision/figures/, en PNG 600 dpi, PDF et SVG)

| Fichier | Section |
|---|---|
| figure2_feature_categories | 3.1 |
| figure3_confusion_matrix | 3.2 |
| figure4_k_sweep | 3.2 |
| figure5_shap_summary | 3.3 |
| figure6_calibration | 3.6 |

La Figure 1 (schema du pipeline) se genere separement.

## Chiffres pour l'abstract

- Espace de features : 456 -> 52 -> 18
- AU-PR 0.9919 (baseline 0.7138, lift 0.2780)
- AUC-ROC 0.9766 | MCC 0.8317
- Rappel au seuil 0.500 : 0.896 | precision 0.994
- VPP a 0.1% : 0.0267 (336 alertes/10k)
- Pre-filtre +0.0716 | selection +0.0094 | IC 0.0059
- Selection au-dessus du bruit
- Stabilite Jaccard 0.693, 16/18 stables
- Confondant ecosysteme : 63.7%
- Calibration : Brier 0.0676, ECE 0.1005
- SHAP : stabilite 0.724, accord permutation 0.509

## Phrases dont la formulation depend du resultat

Selection : "the selection contributes measurably beyond the pre-filter"
k : "k = 18 and k = 40 are equivalent; smaller budgets are significantly worse"
