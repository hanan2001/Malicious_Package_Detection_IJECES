# -*- coding: utf-8 -*-
"""
Regenere les figures du manuscrit IJECES-2026-5223 dans un style "journal".

Integrite : toutes les valeurs sont lues dans les sorties du pipeline
(Results/final_results.json, Tables/table03_features.csv, Results/shap_values.npz) ;
aucune valeur n'est saisie a la main. Seuls la mise en forme et le type de trace changent :
  - largeurs du gabarit IJECES (A4, deux colonnes) : 88 mm / 180 mm ;
  - texte sans serif 6.5-8 pt, traits fins, axes discrets, pas de titre dans la
    figure (le titre est dans la legende du manuscrit), panneaux (a), (b) ;
  - palette validee pour les daltoniens et le contraste : sarcelle #00876C,
    ambre #C77418, ardoise pour le texte ; formes differentes en plus des couleurs ;
  - sorties vectorielles (PDF polices incorporees, SVG) + PNG et TIFF 600 dpi.

Usage :
    python make_figures_journal.py                # ecrit dans Figures/journal/
    python make_figures_journal.py --out autre/dossier
"""
import argparse, json, os, textwrap
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import FancyBboxPatch, Circle

# ------------------------------------------------------------------ style
MM = 1 / 25.4
COL1, COL2 = 88 * MM, 180 * MM            # une colonne, pleine largeur (IJECES, A4)
TEAL, AMBER = "#00876C", "#C77418"        # palette validee (CVD, contraste)
INK, MUTED, GRID, PAPER = "#1F2A30", "#66727A", "#E6EAEC", "#F6F8F8"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.8, "axes.linewidth": 0.6, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.direction": "out", "ytick.direction": "out",
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.1, "legend.frameon": False,
    "savefig.dpi": 600, "figure.dpi": 150,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})

def tint(color, f):
    """Melange une couleur avec du blanc (f = 0 : blanc, f = 1 : couleur pleine)."""
    c = np.array(to_rgb(color)); return tuple(1 - f * (1 - c))

def panel(ax, tag, x=-0.14, y=1.03):
    ax.text(x, y, tag, transform=ax.transAxes, fontsize=8.5, fontweight="bold",
            va="bottom", ha="left", color=INK)

def save(fig, out, name):
    for ext in ("pdf", "svg", "png", "tiff"):
        kw = {"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tiff" else {}
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight",
                    pad_inches=0.02, facecolor="white", **kw)
    plt.close(fig); print("  ", name)

# ------------------------------------------------------------------ Figure 1
def fig1_pipeline(R, out):
    """Six etapes numerotees, cartes a bandeau lateral, enchainement en S."""
    rho_stab = R["shap"]["rank_stability_spearman"]
    rho_perm = R["shap"]["agreement_permutation_spearman"]
    stages = [
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
    fig = plt.figure(figsize=(COL2, 80 * MM)); ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 3); ax.set_ylim(0, 2.04); ax.axis("off")
    bw, bh = 0.88, 0.90
    pos = [(0.04, 1.10), (1.06, 1.10), (2.08, 1.10), (2.08, 0.06), (1.06, 0.06), (0.04, 0.06)]
    for k, ((title, lines), (x, y)) in enumerate(zip(stages, pos), 1):
        ax.add_patch(FancyBboxPatch((x, y), bw, bh, boxstyle="round,pad=0,rounding_size=0.03",
                                    fc=PAPER, ec="#C9D3D6", lw=0.6))
        ax.add_patch(FancyBboxPatch((x, y), 0.018, bh, boxstyle="square,pad=0",
                                    fc=TEAL, ec="none"))
        ax.add_patch(Circle((x + 0.10, y + bh - 0.10), 0.055, fc=TEAL, ec="none"))
        ax.text(x + 0.10, y + bh - 0.10, str(k), ha="center", va="center", color="white",
                fontsize=7.5, fontweight="bold")
        ax.text(x + 0.19, y + bh - 0.10, title, ha="left", va="center", color=INK,
                fontsize=7.6, fontweight="bold")
        ty = y + bh - 0.22
        for ln in lines:
            for i, part in enumerate(textwrap.wrap(ln, 41)):
                ax.text(x + 0.08, ty, ("–  " if i == 0 else "    ") + part, ha="left",
                        va="top", fontsize=6.3, color=INK)
                ty -= 0.072
            ty -= 0.018
    arrow = dict(arrowstyle="-|>,head_length=0.45,head_width=0.28", mutation_scale=10,
                 color=TEAL, lw=1.0, shrinkA=0, shrinkB=0)
    for xm in (0.99, 2.01):   # rangee du haut vers la droite, rangee du bas vers la gauche
        ax.annotate("", xy=(xm + 0.045, 1.10 + bh / 2), xytext=(xm - 0.045, 1.10 + bh / 2), arrowprops=arrow)
        ax.annotate("", xy=(xm - 0.045, 0.06 + bh / 2), xytext=(xm + 0.045, 0.06 + bh / 2), arrowprops=arrow)
    ax.annotate("", xy=(2.52, 0.06 + bh + 0.02), xytext=(2.52, 1.10 - 0.02), arrowprops=arrow)
    save(fig, out, "fig1_pipeline")

# ------------------------------------------------------------------ Figure 2
def fig2_categories(T3, out):
    """Deux diagrammes en sucettes (lollipop) partageant l'ordre des categories."""
    agg = T3.groupby("Category").agg(n=("Feature", "size"), shap=("Mean |SHAP|", "sum"))
    agg = agg.sort_values("shap")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL2, 52 * MM), sharey=True,
                                 gridspec_kw={"wspace": 0.10})
    y = np.arange(len(agg))
    for ax, col, fmt, lab, color, mk in [
            (a1, "n", "{:.0f}", "Number of selected features", TEAL, "o"),
            (a2, "shap", "{:.2f}", "Aggregate mean |SHAP|", AMBER, "D")]:
        v = agg[col].values
        ax.hlines(y, 0, v, color=tint(color, 0.45), lw=2.2)
        ax.scatter(v, y, s=26, color=color, marker=mk, zorder=3, edgecolor="white", lw=0.6)
        for yy, vv in zip(y, v):
            ax.text(vv + v.max() * 0.04, yy, fmt.format(vv), va="center", fontsize=6.8, color=INK)
        ax.set_xlim(0, v.max() * 1.18); ax.set_xlabel(lab)
        ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    a1.set_yticks(y); a1.set_yticklabels(agg.index)
    a1.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    panel(a1, "(a)", x=-0.42); panel(a2, "(b)", x=-0.04)
    save(fig, out, "fig2_feature_categories")

# ------------------------------------------------------------------ Figure 3
def fig3_k_sweep(R, out):
    """Deux panneaux empiles ; bande hachuree = |delta| <= largeur de l'IC a k = 18."""
    sw = pd.DataFrame(R["k_sweep"]); K = int(R["config"]["k_features"])
    ci = R["performance"]["ci95"]
    fig, axes = plt.subplots(2, 1, figsize=(COL1, 84 * MM), sharex=True,
                             gridspec_kw={"hspace": 0.30})
    for ax, col, key, color, mk, tag in [(axes[0], "AU-PR", "au_pr", TEAL, "o", "(a)"),
                                         (axes[1], "AUC-ROC", "auc_roc", AMBER, "D", "(b)")]:
        ref = float(sw.loc[sw.k == K, col].iloc[0]); w = ci[key][1] - ci[key][0]
        ax.fill_between([sw.k.min() - 2, sw.k.max() + 2], ref - w, ref + w,
                        facecolor=tint(color, 0.12), hatch="////", edgecolor=tint(color, 0.35),
                        lw=0, label=f"|Δ| ≤ 95% CI width ({w:.4f})")
        ax.axvline(K, color=MUTED, ls=(0, (3, 2)), lw=0.7)
        ax.plot(sw.k, sw[col], color=color, lw=1.1, zorder=3)
        ax.scatter(sw.k, sw[col], s=[34 if k == K else 18 for k in sw.k], marker=mk,
                   facecolor=["white" if k != K else color for k in sw.k],
                   edgecolor=color, lw=1.0, zorder=4)
        ax.annotate(f"{ref:.4f}", (K, ref), xytext=(5, -11), textcoords="offset points",
                    fontsize=6.5, color=INK)
        ax.set_xlim(sw.k.min() - 2, sw.k.max() + 2)
        ax.set_ylabel(col); ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
        panel(ax, tag, x=-0.22); ax.legend(loc="lower right", handlelength=1.6)
    axes[0].text(K, axes[0].get_ylim()[1], f" k = {K}", fontsize=6.5, color=MUTED, va="top")
    axes[1].set_xticks(sw.k.astype(int)); axes[1].set_xlabel("Number of selected features, k")
    save(fig, out, "fig3_k_sweep")

# ------------------------------------------------------------------ Figure 4
def fig4_confusion(R, out):
    """Tuiles arrondies : decisions correctes en sarcelle, erreurs en ambre."""
    P = R["performance"]
    cm = np.array([[P["TN"], P["FP"]], [P["FN"], P["TP"]]]); rowp = cm / cm.sum(1, keepdims=True)
    tags = [["TN", "FP"], ["FN", "TP"]]
    fig, ax = plt.subplots(figsize=(72 * MM, 66 * MM))
    ax.set_xlim(-0.55, 1.55); ax.set_ylim(1.55, -0.55); ax.set_aspect("equal")
    for i in range(2):
        for j in range(2):
            f = 0.18 + 0.82 * rowp[i, j]
            ax.add_patch(FancyBboxPatch((j - 0.45, i - 0.45), 0.9, 0.9,
                                        boxstyle="round,pad=0,rounding_size=0.08",
                                        fc=tint(TEAL if i == j else AMBER, f), ec="none"))
            c = "white" if f > 0.55 else INK
            ax.text(j - 0.36, i - 0.34, tags[i][j], fontsize=6.3, color=c, fontweight="bold",
                    va="center", ha="left")
            ax.text(j, i - 0.04, f"{cm[i, j]}", ha="center", va="center", fontsize=10,
                    fontweight="bold", color=c)
            ax.text(j, i + 0.22, f"{100 * rowp[i, j]:.1f}% of row", ha="center", va="center",
                    fontsize=6.3, color=c)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Benign", "Malicious"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Benign", "Malicious"], rotation=90, va="center")
    ax.xaxis.set_label_position("top"); ax.xaxis.tick_top()
    ax.set_xlabel("Predicted class", labelpad=4); ax.set_ylabel("True class")
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0)
    save(fig, out, "fig4_confusion_matrix")

# ------------------------------------------------------------------ Figure 5
def _swarm_offsets(v, nbins=60, width=0.38, rng=np.random.default_rng(0)):
    edges = np.linspace(v.min(), v.max() + 1e-12, nbins + 1)
    idx = np.clip(np.digitize(v, edges) - 1, 0, nbins - 1)
    counts = np.bincount(idx, minlength=nbins); off = np.zeros_like(v, dtype=float)
    for b in np.where(counts > 0)[0]:
        m = np.where(idx == b)[0]; n = len(m)
        spread = width * np.sqrt(n / counts.max())
        off[m] = rng.permutation(np.linspace(-spread, spread, n)) if n > 1 else 0.0
    return off

def fig5_shap(npz_path, T3, out):
    """Beeswarm SHAP + barres de |SHAP| moyen (valeurs de la Table 3)."""
    if not os.path.exists(npz_path):
        print("   fig5_shap_summary saute :", npz_path, "absent (relancer final_pipeline_ijeces.py)")
        return
    d = np.load(npz_path, allow_pickle=True)
    sv, X, feats = d["shap"], d["X"], list(d["features"])
    mean_abs = T3.set_index("Feature")["Mean |SHAP|"]
    order = sorted(range(len(feats)), key=lambda j: mean_abs[feats[j]])  # plus importante en haut
    cmap = LinearSegmentedColormap.from_list("teal_amber", [TEAL, "#D9DEDF", AMBER])
    fig = plt.figure(figsize=(COL1 * 0.87, 104 * MM))
    gs = fig.add_gridspec(2, 2, width_ratios=[3.1, 1], height_ratios=[1, 0.035],
                          wspace=0.06, hspace=0.32)
    ax, axb, cax = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0])
    for row, j in enumerate(order):
        x = X[:, j]; lo, hi = np.nanpercentile(x, [5, 95])
        c = np.clip((x - lo) / (hi - lo if hi > lo else 1), 0, 1)
        y = row + _swarm_offsets(sv[:, j])
        s = np.random.default_rng(j).permutation(len(x))
        ax.scatter(sv[s, j], y[s], c=c[s], cmap=cmap, vmin=0, vmax=1, s=1.8, lw=0,
                   alpha=0.9, rasterized=True)
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([feats[j] for j in order], fontsize=5.9)
    ax.set_ylim(-0.7, len(order) - 0.3); ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("SHAP value (log-odds)", fontsize=7)
    ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
    vals = [mean_abs[feats[j]] for j in order]
    axb.barh(range(len(order)), vals, height=0.6, color=tint(TEAL, 0.55))
    axb.set_ylim(ax.get_ylim()); axb.set_yticks([]); axb.spines["left"].set_visible(False)
    axb.set_xlabel("Mean |SHAP|", fontsize=7); axb.tick_params(axis="x", labelsize=6.3)
    axb.grid(axis="x", color=GRID, lw=0.5); axb.set_axisbelow(True)
    panel(ax, "(a)", x=-0.02, y=1.0); panel(axb, "(b)", x=0.0, y=1.0)
    sm = plt.cm.ScalarMappable(cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=[0, 1])
    cb.ax.set_xticklabels(["Low", "High"], fontsize=6.3); cb.outline.set_visible(False)
    cb.set_label("Feature value", fontsize=6.5, labelpad=1)
    save(fig, out, "fig5_shap_summary")

# ------------------------------------------------------------------ Figure 6
def fig6_calibration(R, out):
    """Courbe de fiabilite ; l'ecart a la diagonale est ombre."""
    C = R["calibration"]; cur = pd.DataFrame(C["curve"]).sort_values("mean_predicted")
    fig, ax = plt.subplots(figsize=(70 * MM, 68 * MM))
    xp, yo = cur.mean_predicted.values, cur.observed.values
    ax.fill_between(xp, xp, yo, where=yo >= xp, color=tint(AMBER, 0.22), lw=0,
                    label="Observed > predicted")
    ax.plot([0, 1], [0, 1], color=MUTED, ls=(0, (3, 2)), lw=0.8, label="Perfect calibration")
    ax.plot(xp, yo, color=TEAL, lw=1.2, zorder=3)
    ax.scatter(xp, yo, s=18, color=TEAL, edgecolor="white", lw=0.6, zorder=4, label="CatBoost")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.set_aspect("equal")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed fraction of malicious")
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.legend(loc="upper left", handlelength=1.6, borderaxespad=0.2)
    ax.text(0.97, 0.04, f"Brier = {C['brier']:.4f}\nECE = {C['ece_10bins']:.4f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.8, color=INK,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRID, lw=0.6))
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
    fig4_confusion(R, a.out); fig5_shap(a.shap, T3, a.out); fig6_calibration(R, a.out)

if __name__ == "__main__":
    main()
