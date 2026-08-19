# Replication Package - IJECES 2026

This repository contains the replication package for the paper:
**"Explainable Detection of Malicious Packages Using Point‑in‑Time Metadata"**

## Contents
- `final_pipeline.py` / `IJECES_CLAUDE.ipynb` : Main code to reproduce all results.
- `tables/` : CSV files for Tables 1 to 11.
- `figures/` : Figures 2 to 6 in PNG, PDF and SVG.
- `final_partitions.csv` : Train/Val/Test split for every package version.

## How to run
1. Open the notebook in Google Colab.
2. Run all cells. The script will automatically download the OSSF dataset and cache it locally.
3. All outputs will be saved in the `tables/` and `figures/` folders.

## Requirements
See the imports at the top of the notebook (pandas, numpy, scikit-learn, catboost, shap...).