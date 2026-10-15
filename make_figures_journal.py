# -*- coding: utf-8 -*-
"""
Regenere les figures du manuscrit IJECES-2026-5223 dans un style "grand journal".

Le CONTENU est inchange : toutes les valeurs sont lues dans les sorties du pipeline
(Results/final_results.json, Tables/*.csv, Results/shap_values.npz). Seuls changent
le style et la mise en forme :
  - largeurs de colonne du gabarit IJECES (A4, deux colonnes) : 88 mm / 180 mm ;
  - police sans serif 7-8 pt, traits fins, axes discrets, pas de titre dans la figure
    (le titre est dans la legende du manuscrit), etiquettes de panneau (a), (b) ;
  - palette sure pour les daltoniens : indigo #3A5BA0, rouge rose #D1495B ;
  - sorties vectorielles (PDF polices incorporees, SVG) + PNG et TIFF 600 dpi.

Usage :
    python make_figures_journal.py                # ecrit dans Figures/journal/
    python make_figures_journal.py --out autre/dossier

La Figure 5 (SHAP) a besoin de Results/shap_values.npz, ecrit par
final_pipeline_ijeces.py ; s'il manque, la figure est sautee avec un message.
"""
import argparse, json, os, textwrap
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ------------------------------------------------------------------ style
MM = 1 / 25.4
COL1, COL2 = 88 * MM, 180 * MM            # une colonne, pleine largeur (IJECES, A4)
# palette validee (daltonisme, contraste) : indigo + rouge rose
BLUE, VERM, INK, MUTED, GRID = "#3A5BA0", "#D1495B", "#1A1A1A", "#5F5F5F", "#E3E3E3"
BOX_FILL, BOX_EDGE = "#F3F4F9", "#AEB6D3"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.8, "axes.linewidth": 0.6, "axes.edgecolor": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.direction": "out", "ytick.direction": "out",
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.2, "lines.markersize": 4,
    "legend.frameon": False, "savefig.dpi": 600, "figure.dpi": 150,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})

def panel(ax, tag, x=-0.14, y=1.04):
    ax.text(x, y, tag, transform=ax.transAxes, fontsize=8.5, fontweight="bold",
            va="bottom", ha="left", color=INK)

def ygrid(ax):
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)

def xgrid(ax):
    ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)

def save(fig, out, name):
    for ext in ("pdf", "svg", "png", "tiff"):
        kw = {"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tiff" else {}
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight",
                    pad_inches=0.02, facecolor="white", **kw)
    plt.close(fig); print("  ", name)

# ------------------------------------------------------------------ Figure 1
def fig1_pipeline(R, out):
    rho_stab = R["shap"]["rank_stability_spearman"]
    rho_perm = R["shap"]["agreement_permutation_spearman"]
    boxes = [
        ("Data collection", [
            "Malicious packages: OSSF corpus (aggregates GHSA, Snyk)",
            "Benign packages: NPM and PyPI registry APIs",
            "Temporal matching: benign drawn from the malicious window",
            "Group-aware split: package, maintainer, name-similarity family"]),
        ("Feature engineering", [
            "456 point-in-time features: vman_* (manifest) + pit_* (release history)",
            "cur_* features removed: post-publication registry state",
            "Standardisation fitted on training partition only",
            "Test partition excluded from preprocessing and selection"]),
        ("Feature selection", [
            "Ecosystem pre-filter: removes 404 attributes constant within a registry",
            "Embedded selection, top-k: CatBoost importance, k = 18",
            "456 → 52 → 18 features"]),
        ("Model training", [
            "CatBoost, single model",
            "Balanced class weights",
            "Grouped 5-fold cross-validation",
            "Grid search: depth, learning rate, L2, iterations",
            "Default threshold 0.5"]),
        ("XAI and interpretability", [
            "SHAP summary plot: global attribution, applied post hoc",
            f"Attribution stability: Spearman ρ = {rho_stab:.3f} across resamples",
            f"Permutation importance: agreement with SHAP = {rho_perm:.3f}"]),
        ("Evaluation", [
            "AU-PR, AUC-ROC and 95% CIs",
            "MCC, F1 and recall",
            "Calibration (Brier, ECE)",
            "PPV and alerts per 10,000",
            "Paired tests",
            "Robustness and cross-ecosystem validation"]),
    ]
    W, H = COL2, 78 * MM
    fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 3); ax.set_ylim(0, 2); ax.axis("off")
    bw, bh, head = 0.90, 0.86, 0.15
    pos = [(0.05, 1.07), (1.05, 1.07), (2.05, 1.07), (0.05, 0.07), (1.05, 0.07), (2.05, 0.07)]
    for (title, lines), (x, y) in zip(boxes, pos):
        ax.add_patch(FancyBboxPatch((x, y), bw, bh, boxstyle="round,pad=0,rounding_size=0.025",
                                    fc=BOX_FILL, ec=BOX_EDGE, lw=0.6))
        ax.add_patch(FancyBboxPatch((x, y + bh - head), bw, head,
                                    boxstyle="round,pad=0,rounding_size=0.025",
                                    fc=BLUE, ec=BLUE, lw=0.6))
        ax.text(x + bw / 2, y + bh - head / 2, title.upper(), ha="center", va="center",
                color="white", fontsize=7, fontweight="bold")
        ty = y + bh - head - 0.06
        for ln in lines:
            wrapped = textwrap.wrap(ln, 43)
            for k, part in enumerate(wrapped):
                ax.text(x + 0.05 + (0 if k == 0 else 0.035), ty, ("•  " if k == 0 else "") + part,
                        ha="left", va="top", fontsize=6.4, color=INK)
                ty -= 0.072
            ty -= 0.02
    arrow = dict(arrowstyle="-|>,head_length=3,head_width=1.8", color=MUTED, lw=0.9)
    for x0 in (0.95, 1.95):
        for yc in (1.07 + bh / 2, 0.07 + bh / 2):
            ax.add_patch(FancyArrowPatch((x0 + 0.01, yc), (x0 + 0.09, yc), **arrow))
    ax.add_patch(FancyArrowPatch((2.5, 1.06), (0.5, 0.94),
                                 connectionstyle="angle3,angleA=-90,angleB=0", **arrow))
    save(fig, out, "fig1_pipeline")

# ------------------------------------------------------------------ Figure 2
def fig2_categories(T3, out):
    agg = T3.groupby("Category").agg(n=("Feature", "size"), shap=("Mean |SHAP|", "sum"))
    agg = agg.sort_values("shap")                 # plus grande attribution en haut
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL2, 52 * MM), sharey=True,
                                 gridspec_kw={"wspace": 0.12})
    y = np.arange(len(agg))
    for ax, col, fmt, lab in [(a1, "n", "{:.0f}", "Number of selected features"),
                              (a2, "shap", "{:.2f}", "Aggregate mean |SHAP|")]:
        v = agg[col].values
        ax.barh(y, v, height=0.62, color=BLUE, edgecolor="white", lw=0.8)
        for yy, vv in zip(y, v):
            ax.text(vv + v.max() * 0.015, yy, fmt.format(vv), va="center", fontsize=6.8, color=INK)
        ax.set_xlim(0, v.max() * 1.15); ax.set_xlabel(lab); xgrid(ax)
        ax.tick_params(axis="y", length=0)
    a1.set_yticks(y); a1.set_yticklabels(agg.index)
    a1.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    panel(a1, "(a)", x=-0.42); panel(a2, "(b)", x=-0.04)
    save(fig, out, "fig2_feature_categories")

# ------------------------------------------------------------------ Figure 3
def fig3_k_sweep(R, out):
    sw = pd.DataFrame(R["k_sweep"]); K = int(R["config"]["k_features"])
    ci = R["performance"]["ci95"]
    fig, axes = plt.subplots(2, 1, figsize=(COL1, 82 * MM), sharex=True,
                             gridspec_kw={"hspace": 0.28})
    for ax, col, key, color, mk, tag in [(axes[0], "AU-PR", "au_pr", BLUE, "o", "(a)"),
                                         (axes[1], "AUC-ROC", "auc_roc", VERM, "s", "(b)")]:
        ref = float(sw.loc[sw.k == K, col].iloc[0]); w = ci[key][1] - ci[key][0]
        ax.axhspan(ref - w, ref + w, color=color, alpha=0.10, lw=0,
                   label=f"|Δ| ≤ 95% CI width ({w:.4f})")
        ax.axvline(K, color=MUTED, ls=(0, (2, 2)), lw=0.7)
        ax.plot(sw.k, sw[col], color=color, marker=mk, ms=4, mec="white", mew=0.6, lw=1.2)
        ax.set_ylabel(col); ygrid(ax); panel(ax, tag, x=-0.2)
        ax.legend(loc="lower right", handlelength=1.2)
    axes[1].set_xticks(sw.k.astype(int)); axes[1].set_xlabel("Number of selected features, k")
    save(fig, out, "fig3_k_sweep")

# ------------------------------------------------------------------ Figure 4
def fig4_confusion(R, out):
    P = R["performance"]
    cm = np.array([[P["TN"], P["FP"]], [P["FN"], P["TP"]]])
    rowp = cm / cm.sum(1, keepdims=True)
    cmap = LinearSegmentedColormap.from_list("blue_seq", ["#F3F4F9", "#B7C1E0", "#5B74B8", "#22346B"])
    fig, ax = plt.subplots(figsize=(72 * MM, 66 * MM))
    ax.imshow(rowp, cmap=cmap, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            c = "white" if rowp[i, j] > 0.5 else INK
            ax.text(j, i - 0.08, f"{cm[i, j]}", ha="center", va="center", fontsize=9,
                    fontweight="bold", color=c)
            ax.text(j, i + 0.2, f"{100 * rowp[i, j]:.1f}%", ha="center", va="center",
                    fontsize=6.8, color=c)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Benign", "Malicious"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Benign", "Malicious"], rotation=90, va="center")
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_xticks([0.5], minor=True); ax.set_yticks([0.5], minor=True)
    ax.grid(which="minor", color="white", lw=2); ax.tick_params(which="minor", length=0)
    save(fig, out, "fig4_confusion_matrix")

# ------------------------------------------------------------------ Figure 5
def _swarm_offsets(v, nbins=60, width=0.38, rng=np.random.default_rng(0)):
    """Decalage vertical proportionnel a la densite locale (style beeswarm)."""
    edges = np.linspace(v.min(), v.max() + 1e-12, nbins + 1)
    idx = np.clip(np.digitize(v, edges) - 1, 0, nbins - 1)
    counts = np.bincount(idx, minlength=nbins); off = np.zeros_like(v, dtype=float)
    cmax = counts.max()
    for b in np.where(counts > 0)[0]:
        m = np.where(idx == b)[0]; n = len(m)
        spread = width * np.sqrt(n / cmax)
        off[m] = rng.permutation(np.linspace(-spread, spread, n)) if n > 1 else 0.0
    return off

def fig5_shap(npz_path, out):
    if not os.path.exists(npz_path):
        print("   fig5_shap_summary saute :", npz_path, "absent (relancer final_pipeline_ijeces.py)")
        return
    d = np.load(npz_path, allow_pickle=True)
    sv, X, feats = d["shap"], d["X"], list(d["features"])
    order = np.argsort(np.abs(sv).mean(0))          # plus importante en haut
    cmap = LinearSegmentedColormap.from_list("shap_cvd", [BLUE, "#D3D0DC", VERM])
    fig, ax = plt.subplots(figsize=(COL1 * 0.74, 100 * MM))
    for row, j in enumerate(order):
        x = X[:, j]; lo, hi = np.nanpercentile(x, [5, 95])
        c = np.clip((x - lo) / (hi - lo if hi > lo else 1), 0, 1)
        y = row + _swarm_offsets(sv[:, j])
        s = np.argsort(np.random.default_rng(j).random(len(x)))
        ax.scatter(sv[s, j], y[s], c=c[s], cmap=cmap, vmin=0, vmax=1, s=2.2, lw=0,
                   alpha=0.85, rasterized=True)
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([feats[j] for j in order], fontsize=6.3)
    ax.set_ylim(-0.7, len(order) - 0.3); ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("SHAP value (impact on model output, log-odds)"); xgrid(ax)
    sm = plt.cm.ScalarMappable(cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02, aspect=40, ticks=[0, 1])
    cb.ax.set_yticklabels(["Low", "High"], fontsize=6.5); cb.outline.set_visible(False)
    cb.set_label("Feature value", fontsize=6.8, labelpad=-4)
    save(fig, out, "fig5_shap_summary")

# ------------------------------------------------------------------ Figure 6
def fig6_calibration(R, out):
    C = R["calibration"]; cur = pd.DataFrame(C["curve"])
    fig, ax = plt.subplots(figsize=(COL1 * 0.8, COL1 * 0.75))
    ax.plot([0, 1], [0, 1], color=MUTED, ls=(0, (3, 2)), lw=0.8, label="Perfect calibration")
    ax.plot(cur.mean_predicted, cur.observed, color=BLUE, marker="o", ms=4, mec="white",
            mew=0.6, lw=1.2, label="CatBoost")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.set_aspect("equal")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed fraction of malicious")
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.legend(loc="upper left", handlelength=1.6)
    ax.text(0.97, 0.04, f"Brier = {C['brier']:.4f}\nECE = {C['ece_10bins']:.4f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.8, color=INK)
    save(fig, out, "fig6_calibration")

# ------------------------------------------------------------------ main
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--results", default=os.path.join(here, "Results", "final_results.json"))
    ap.add_argument("--tables", default=os.path.join(here, "Tables"))
    ap.add_argument("--shap", default=os.path.join(here, "Results", "shap_values.npz"))
    ap.add_argument("--out", default=os.path.join(here, "Figures", "journal"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    R = json.load(open(a.results, encoding="utf8"))
    T3 = pd.read_csv(os.path.join(a.tables, "table03_features.csv"))
    print("Figures ->", a.out)
    fig1_pipeline(R, a.out); fig2_categories(T3, a.out); fig3_k_sweep(R, a.out)
    fig4_confusion(R, a.out); fig5_shap(a.shap, a.out); fig6_calibration(R, a.out)

if __name__ == "__main__":
    main()
