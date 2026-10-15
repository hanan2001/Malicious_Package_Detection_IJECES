# Valeurs a reporter

Genere le 2026-08-15T18:58:58.871264+00:00. Toute valeur du manuscrit doit venir d'ici.

## Tables (dans /content/drive/MyDrive/JCP_revision/tables/, aussi en Markdown dans ALL_TABLES.md)

| Fichier | Manuscrit |
|---|---|
| table01_dataset, table01b_partitions | 2.2, Tables 1a et 1b |
| table02_reduction | 2.3, Table 2 |
| table03_features | 3.1, Table 3 |
| table04_performance | 3.2, Table 4 |
| table08_paired_tests | 3.4, Table 5a |
| table06_gain_decomposition | 3.1, Table 6 |
| table07_k_sweep | 3.1, Table 7 |
| table09_family_ablation | 3.3, Table 8 |
| table10_robustness | 3.5, Table 9 |
| table05_prevalence | 3.5, Table 10 |
| table11_cross_ecosystem | 3.7, Table 11 |

## Figures (dans /content/drive/MyDrive/JCP_revision/figures/, en PNG 600 dpi, PDF et SVG)

| Fichier | Manuscrit |
|---|---|
| figure2_feature_categories | 2.3, Fig. 2 |
| figure4_k_sweep | 3.1, Fig. 3 |
| figure3_confusion_matrix | 3.2, Fig. 4 |
| figure5_shap_summary | 3.3, Fig. 5 |
| figure6_calibration | 3.6, Fig. 6 |

La Figure 1 (schema du pipeline) se genere separement.

## Chiffres pour l'abstract

- Espace de features : 456 -> 52 -> 18
- AU-PR 0.9919 (baseline 0.7138, lift 0.2780)
- AUC-ROC 0.9766 | MCC 0.8317
- Rappel au seuil 0.500 : 0.896 | precision 0.994
- VPP a 0.1% : 0.0267 (336 alertes/10k)
- Pre-filtre +0.0716 | selection +0.0094 | IC 0.0059
- Tirage aleatoire 18/52 : AU-PR 0.9825, AUC-ROC 0.9598
- Selection au-dessus du bruit
- Stabilite Jaccard 0.693, 16/18 stables
- Confondant ecosysteme : 63.7%
- Calibration : Brier 0.0676, ECE 0.1005
- SHAP : stabilite 0.724, accord permutation 0.509

## Phrases dont la formulation depend du resultat

Selection : "the selection contributes measurably beyond the pre-filter"
k : "k = 18 and k = 40 are equivalent; smaller budgets are significantly worse"
