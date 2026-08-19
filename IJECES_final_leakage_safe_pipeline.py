"""
IJECES FINAL PIPELINE — Leakage-safe manuscript revision
=========================================================
A single reproducible pipeline for the IJECES revision.

IMPORTANT
---------
- The TEST set is never used for feature selection, hyperparameter tuning,
  selector choice, or threshold selection.
- Preprocessing and feature selection are fitted only on training data.
- Inner model-selection CV is group-aware and re-runs preprocessing/feature
  selection inside every inner fold (strict nested selection).
- Validation is used for threshold selection only.
- Temporal and cross-ecosystem tests are independent evaluations.
- All tables/figures are generated from observed results; no result values
  are hard-coded.

INPUT
-----
CSV containing at least:
package, version, ecosystem, t_p, label
and preferably maintainer. Numeric metadata columns are detected automatically.

The script deliberately does NOT perform a label-derived temporal matching
window. If dataset_v5.csv was itself constructed using labels to select its
time window, that upstream dataset construction must be replaced before the
reported experiment. This script cannot undo upstream leakage.

FEATURE PROTOCOL
----------------
Point-in-time numeric metadata
 -> remove cur_* current-state attributes and known leakage fields
 -> ecosystem-proxy removal (learned on training only)
 -> VarianceThreshold(0.01)
 -> Mutual Information (top 100)
 -> Boruta (max_iter=50, perc=90, alpha=0.05)
 -> CatBoost RFE to 30
 -> Pearson correlation pruning (0.80)
 -> fixed feature budget of 18

The parameters above reproduce the documented corrected protocol rather than
being selected from the test set.

Outputs
-------
Manuscript-ready CSV tables, PNG figures, JSON results, package/split manifest,
and a run manifest containing versions and input SHA-256.
"""

import os, re, json, itertools, warnings, hashlib, platform, sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, RFE, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, average_precision_score,
    confusion_matrix, brier_score_loss, precision_recall_curve, roc_curve
)
from sklearn.model_selection import (
    StratifiedGroupKFold, GroupShuffleSplit, ParameterGrid
)
from scipy.stats import binomtest, beta as beta_dist
from catboost import CatBoostClassifier

try:
    from boruta import BorutaPy
except Exception as exc:
    raise ImportError(
        "BorutaPy is required. In Colab run: pip install boruta"
    ) from exc

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

# =====================================================================
# CONFIGURATION
# =====================================================================
CFG = {
    "input_csv": "/content/drive/MyDrive/JCP_revision/dataset_v5.csv",
    "output_dir": "/content/drive/MyDrive/JCP_revision/IJECES_FINAL",

    "test_frac": 0.25,
    "val_frac_of_trainval": 0.20,
    "n_inner_splits": 5,

    "name_similarity": 0.85,

    # Point-in-time / leakage exclusions
    "exclude_prefixes": ["cur_"],
    "leakage_columns": [
        "exists", "error_type", "downloads", "last_serial", "yanked",
        "deprecated", "status", "is_removed", "_npmUser"
    ],

    # Main protocol: two ecosystems, remove vman ecosystem proxies.
    # "all" can be run as a sensitivity analysis.
    "proxy_scope": "vman",

    # Documented corrected feature-selection parameters
    "variance_threshold": 0.01,
    "mi_k": 100,
    "boruta_max_iter": 50,
    "boruta_perc": 90,
    "boruta_alpha": 0.05,
    "rfe_target": 30,
    "correlation_threshold": 0.80,
    "final_k": 18,

    # CatBoost tuning
    "catboost_grid": {
        "depth": [4, 6, 8],
        "learning_rate": [0.03, 0.05, 0.10],
        "l2_leaf_reg": [1, 3, 5],
        "iterations": [500],
    },

    # Operational analysis
    "target_prevalence": 0.001,
    "prevalences": [0.01, 0.001, 0.0001],
    "min_recall_validation": 0.70,
    "threshold_round_decimals": 3,

    # Bootstrap
    "n_boot": 2000,
    "n_paired_boot": 2000,
    "n_shap_stability": 10,

    # Reproducibility
    "save_package_ids": True,
}

OUT = Path(CFG["output_dir"])
OUT.mkdir(parents=True, exist_ok=True)

META = [
    "package", "version", "ecosystem", "t_p", "maintainer",
    "label", "group_id", "canon", "advisory_id"
]

# =====================================================================
# BASIC HELPERS
# =====================================================================
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canon_name(x):
    return re.sub(r"[^a-z0-9]", "", str(x).lower())


def name_similarity(a, b):
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    if not la or not lb:
        return 0.0
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (ca != cb)
            ))
        prev = cur
    return 1.0 - prev[lb] / max(la, lb)


class UnionFind:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def build_groups(df):
    """
    Group versions to prevent package/maintainer/name-family leakage.

    Package identity + maintainer are grouped, and highly similar canonical
    package names are merged. This function does not use labels.
    """
    d = df.copy()
    uf = UnionFind()
    d["canon"] = d["package"].map(canon_name)

    for _, row in d.iterrows():
        p = row["canon"]
        uf.find(("pkg", p))
        maintainer = str(row.get("maintainer", "")).strip().lower()
        if maintainer and maintainer not in ("nan", "0", ""):
            uf.union(("pkg", p), ("mnt", maintainer))

    blocks = {}
    for n in sorted(d["canon"].unique()):
        blocks.setdefault((n[:2], len(n) // 4), []).append(n)

    similar_pairs = 0
    for block in blocks.values():
        for a, b in itertools.combinations(block, 2):
            if abs(len(a) - len(b)) <= 3 and \
               name_similarity(a, b) >= CFG["name_similarity"]:
                uf.union(("pkg", a), ("pkg", b))
                similar_pairs += 1

    d["group_id"] = d["canon"].map(
        lambda n: str(uf.find(("pkg", n)))
    )
    return d, similar_pairs


def numeric_features(df):
    X = df.drop(
        columns=[c for c in META if c in df.columns],
        errors="ignore"
    ).select_dtypes(include=[np.number]).copy()

    keep = []
    for c in X.columns:
        if any(c.startswith(p) for p in CFG["exclude_prefixes"]):
            continue
        if c in CFG["leakage_columns"]:
            continue
        keep.append(c)
    return X[keep]


def metrics(y, score, threshold=0.5):
    y = np.asarray(y)
    score = np.asarray(score)
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y, pred, labels=[0, 1]
    ).ravel()

    return {
        "n": int(len(y)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "auc_roc": float(roc_auc_score(y, score)),
        "au_pr": float(average_precision_score(y, score)),
        "au_pr_baseline": float(np.mean(y)),
        "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
        "fpr": float(fp / (tn + fp)) if tn + fp else np.nan,
        "brier": float(brier_score_loss(y, score)),
        "threshold": float(threshold),
    }


def bootstrap_ci(y, score, metric_fn, n=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    y, score = np.asarray(y), np.asarray(score)
    vals = []

    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(metric_fn(y[idx], score[idx]))

    return [
        float(np.percentile(vals, 2.5)),
        float(np.percentile(vals, 97.5))
    ]


def fpr_upper95(fp, n_benign):
    if n_benign <= 0:
        return np.nan
    if fp == 0:
        return 3.0 / n_benign
    if fp >= n_benign:
        return 1.0
    return float(beta_dist.ppf(0.95, fp + 1, n_benign - fp))


def operational_metrics(y, score, threshold, prevalence):
    m = metrics(y, score, threshold)
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y, pred, labels=[0, 1]
    ).ravel()

    rows = []
    for fpr_type, fpr in [
        ("point", m["fpr"]),
        ("upper95", fpr_upper95(fp, tn + fp))
    ]:
        tpr = m["recall"]
        denominator = tpr * prevalence + fpr * (1 - prevalence)
        ppv = (
            (tpr * prevalence) / denominator
            if denominator > 0 else np.nan
        )
        rows.append({
            "prevalence": prevalence,
            "fpr_type": fpr_type,
            "tpr": tpr,
            "fpr": fpr,
            "ppv": ppv,
            "alerts_per_10k": 10000 * denominator,
            "tp_per_10k": 10000 * tpr * prevalence,
            "missed_per_10k": 10000 * (1 - tpr) * prevalence,
        })
    return rows


def paired_bootstrap_delta(y, score_a, score_b, n=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    a = np.asarray(score_a)
    b = np.asarray(score_b)
    vals = []

    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(
            average_precision_score(y[idx], a[idx])
            - average_precision_score(y[idx], b[idx])
        )

    vals = np.asarray(vals)
    p = 2 * min(np.mean(vals <= 0), np.mean(vals >= 0))

    return {
        "delta_aupr": float(vals.mean()),
        "ci_low": float(np.percentile(vals, 2.5)),
        "ci_high": float(np.percentile(vals, 97.5)),
        "p_two_sided": float(min(p, 1.0)),
    }


def mcnemar_exact(y, pred_a, pred_b):
    y = np.asarray(y)
    a_ok = np.asarray(pred_a) == y
    b_ok = np.asarray(pred_b) == y
    b = int(np.sum(a_ok & ~b_ok))
    c = int(np.sum(~a_ok & b_ok))
    p = (
        float(binomtest(min(b, c), b + c, 0.5).pvalue)
        if b + c else 1.0
    )
    return {"b": b, "c": c, "p_exact": p}


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return np.nan
    return len(a & b) / len(a | b)

# =====================================================================
# TRAIN-ONLY PREPROCESSOR
# =====================================================================
class TrainPreprocessor:
    def __init__(self, features):
        self.features = list(features)
        self.imputer = SimpleImputer(strategy="median")
        self.lo = None
        self.hi = None
        self.scaler = StandardScaler()

    def fit(self, df):
        X = numeric_features(df).reindex(
            columns=self.features, fill_value=0
        )
        A = self.imputer.fit_transform(X)
        self.lo = np.nanpercentile(A, 1, axis=0)
        self.hi = np.nanpercentile(A, 99, axis=0)
        A = np.clip(A, self.lo, self.hi)
        self.scaler.fit(A)
        return self

    def transform(self, df):
        X = numeric_features(df).reindex(
            columns=self.features, fill_value=0
        )
        A = self.imputer.transform(X)
        A = np.clip(A, self.lo, self.hi)
        return pd.DataFrame(
            self.scaler.transform(A),
            columns=self.features,
            index=df.index
        )

# =====================================================================
# TRAIN-ONLY ECOSYSTEM PROXY FILTER
# =====================================================================
def fit_proxy_filter(train_df, full_features, scope="vman"):
    """
    Removes attributes that are constant within at least one ecosystem.
    The decision is learned from TRAIN ONLY.
    """
    if train_df["ecosystem"].nunique() < 2:
        return [], list(full_features)

    X = numeric_features(train_df).reindex(
        columns=full_features, fill_value=0
    )

    if scope == "vman":
        candidates = [
            c for c in X.columns if c.startswith("vman_")
        ]
    elif scope == "all":
        candidates = list(X.columns)
    else:
        candidates = []

    proxies = []
    for c in candidates:
        std_by_eco = train_df.assign(
            _x=X[c]
        ).groupby("ecosystem")["_x"].std()
        if (std_by_eco.fillna(0) == 0).any():
            proxies.append(c)

    post = [c for c in full_features if c not in proxies]
    return proxies, post

# =====================================================================
# FIVE-STAGE HYBRID FEATURE SELECTION
# =====================================================================
def hybrid_select(X_train, y_train, random_state=SEED):
    """
    Fit all feature-selection operations ONLY on X_train/y_train.

    Documented protocol:
      1. Variance Threshold = 0.01
      2. Mutual Information = top 100
      3. Boruta = max_iter 50, perc 90, alpha .05
      4. CatBoost RFE = 30 features
      5. Pearson correlation pruning = .80
      6. A-priori final budget = 18
    """
    X = X_train.copy()
    f0 = list(X.columns)
    stages = []

    # Stage 1
    vt = VarianceThreshold(
        threshold=CFG["variance_threshold"]
    )
    Xv = vt.fit_transform(X)
    fv = list(X.columns[vt.get_support()])
    stages.append(("Variance Threshold", len(f0), len(fv)))
    X = pd.DataFrame(Xv, columns=fv, index=X.index)

    # Stage 2
    if len(fv) > CFG["mi_k"]:
        mi = mutual_info_classif(
            X, y_train, random_state=random_state
        )
        keep = np.argsort(mi)[::-1][:CFG["mi_k"]]
        fmi = [fv[i] for i in keep]
    else:
        fmi = fv.copy()
    stages.append(("Mutual Information", len(fv), len(fmi)))
    X = X[fmi]

    # Stage 3
    boruta_estimator = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1
    )

    boruta = BorutaPy(
        estimator=boruta_estimator,
        n_estimators="auto",
        max_iter=CFG["boruta_max_iter"],
        perc=CFG["boruta_perc"],
        alpha=CFG["boruta_alpha"],
        two_step=True,
        random_state=random_state,
        verbose=0
    )

    boruta.fit(X.values, np.asarray(y_train))
    fb = list(X.columns[boruta.support_])

    # Deterministic fallback if Boruta confirms none.
    if len(fb) == 0:
        fb = list(X.columns[boruta.support_weak_])
    if len(fb) == 0:
        fb = list(X.columns)

    stages.append(("Boruta", len(fmi), len(fb)))
    X = X[fb]

    # Stage 4
    rfe_target = min(CFG["rfe_target"], X.shape[1])
    if X.shape[1] > rfe_target:
        base = CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=3,
            verbose=0,
            random_state=random_state,
            auto_class_weights="Balanced",
            allow_writing_files=False
        )
        rfe = RFE(
            estimator=base,
            n_features_to_select=rfe_target,
            step=1
        )
        rfe.fit(X, y_train)
        frfe = list(X.columns[rfe.support_])
    else:
        frfe = list(X.columns)

    stages.append(("RFE (CatBoost)", len(fb), len(frfe)))
    X = X[frfe]

    # Stage 5
    corr = X.corr(method="pearson").abs()
    upper = corr.where(
        np.triu(np.ones(corr.shape), k=1).astype(bool)
    )

    to_drop = set()
    for col in upper.columns:
        if any(
            upper[col] > CFG["correlation_threshold"]
        ):
            to_drop.add(col)

    fc = [c for c in X.columns if c not in to_drop]
    stages.append(("Correlation Pruning", len(frfe), len(fc)))
    X = X[fc]

    # Final fixed feature budget
    if len(fc) > CFG["final_k"]:
        budget_model = CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=3,
            verbose=0,
            random_state=random_state,
            auto_class_weights="Balanced",
            allow_writing_files=False
        )
        budget_model.fit(X, y_train)
        imp = pd.Series(
            budget_model.get_feature_importance(),
            index=X.columns
        ).sort_values(ascending=False)
        final_features = list(
            imp.head(CFG["final_k"]).index
        )
    else:
        final_features = list(X.columns)

    stages.append(
        ("Feature Budget", len(fc), len(final_features))
    )

    return final_features, pd.DataFrame(
        stages,
        columns=["stage", "features_in", "features_out"]
    )

# =====================================================================
# CATBOOST
# =====================================================================
def make_catboost():
    return CatBoostClassifier(
        verbose=0,
        random_state=SEED,
        auto_class_weights="Balanced",
        allow_writing_files=False
    )


def nested_group_cv_select_params(train_df, post_features):
    """
    STRICT INNER CV.

    For every candidate hyperparameter configuration and every fold:
      - proxy filtering is fitted on fold-train only;
      - imputation/winsorisation/standardisation are fitted on fold-train only;
      - feature selection is fitted on fold-train only;
      - the model is evaluated on fold-validation only.

    Therefore the CV score is not based on a feature selector fitted on the
    entire outer training partition.
    """
    cv = StratifiedGroupKFold(
        n_splits=CFG["n_inner_splits"],
        shuffle=True,
        random_state=SEED
    )

    grid = list(ParameterGrid(CFG["catboost_grid"]))
    rows = []

    for hp in grid:
        fold_scores = []

        for fold, (i_tr, i_va) in enumerate(
            cv.split(
                train_df,
                train_df["label"],
                groups=train_df["group_id"]
            )
        ):
            fold_tr = train_df.iloc[i_tr].copy()
            fold_va = train_df.iloc[i_va].copy()

            fold_full = list(
                numeric_features(fold_tr).columns
            )
            _, fold_post = fit_proxy_filter(
                fold_tr,
                fold_full,
                CFG["proxy_scope"]
            )
            fold_post = [
                f for f in post_features if f in fold_post
            ]

            prep = TrainPreprocessor(fold_post).fit(fold_tr)
            Xtr0 = prep.transform(fold_tr)
            Xva0 = prep.transform(fold_va)

            selected, _ = hybrid_select(
                Xtr0,
                fold_tr["label"].values,
                SEED + fold
            )

            model = make_catboost()
            model.set_params(**hp)
            model.fit(
                Xtr0[selected],
                fold_tr["label"].values
            )

            score = model.predict_proba(
                Xva0[selected]
            )[:, 1]

            fold_scores.append(
                average_precision_score(
                    fold_va["label"].values,
                    score
                )
            )

        rows.append({
            **hp,
            "mean_aupr": float(np.mean(fold_scores)),
            "std_aupr": float(
                np.std(fold_scores, ddof=1)
            )
        })

    table = pd.DataFrame(rows).sort_values(
        ["mean_aupr", "std_aupr"],
        ascending=[False, True]
    ).reset_index(drop=True)

    best = table.iloc[0]
    best_params = {
        k: best[k] for k in CFG["catboost_grid"].keys()
    }

    return best_params, float(best["mean_aupr"]), table

# =====================================================================
# VALIDATION-ONLY THRESHOLD
# =====================================================================
def choose_threshold_validation(y_val, s_val):
    rows = []

    for threshold in np.unique(
        np.round(
            s_val,
            CFG["threshold_round_decimals"]
        )
    ):
        m = metrics(y_val, s_val, float(threshold))

        if m["recall"] < CFG["min_recall_validation"]:
            continue

        op = operational_metrics(
            y_val,
            s_val,
            float(threshold),
            CFG["target_prevalence"]
        )

        upper = [
            r for r in op
            if r["fpr_type"] == "upper95"
        ][0]

        rows.append({
            "threshold": float(threshold),
            "recall_val": m["recall"],
            "ppv_upper95": upper["ppv"],
            "alerts_per_10k_upper95":
                upper["alerts_per_10k"]
        })

    if not rows:
        return (
            0.5,
            pd.DataFrame(),
            "default_0.5_no_candidate"
        )

    table = pd.DataFrame(rows).sort_values(
        ["ppv_upper95", "recall_val"],
        ascending=[False, False]
    )

    return (
        float(table.iloc[0]["threshold"]),
        table,
        "max_validation_PPV_upper95"
    )

# =====================================================================
# MAIN GROUP-AWARE SPLIT
# =====================================================================
def make_main_split(df):
    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=CFG["test_frac"],
        random_state=SEED
    )
    i_tv, i_te = next(
        gss.split(
            df,
            df["label"],
            groups=df["group_id"]
        )
    )

    trainval = df.iloc[i_tv].copy()
    test = df.iloc[i_te].copy()

    gss2 = GroupShuffleSplit(
        n_splits=1,
        test_size=CFG["val_frac_of_trainval"],
        random_state=SEED + 1
    )
    i_tr, i_va = next(
        gss2.split(
            trainval,
            trainval["label"],
            groups=trainval["group_id"]
        )
    )

    train = trainval.iloc[i_tr].copy()
    val = trainval.iloc[i_va].copy()

    return train, val, test


def assert_group_disjoint(*parts):
    for (name_a, a), (name_b, b) in itertools.combinations(
        parts, 2
    ):
        overlap = (
            set(a["group_id"]) &
            set(b["group_id"])
        )
        if overlap:
            raise RuntimeError(
                f"Group leakage: {name_a} vs {name_b}: "
                f"{len(overlap)} overlapping groups"
            )

# =====================================================================
# MAIN PROTOCOL
# =====================================================================
def fit_main_protocol(df):
    train, val, test = make_main_split(df)

    assert_group_disjoint(
        ("train", train),
        ("validation", val),
        ("test", test)
    )

    full_features = list(
        numeric_features(train).columns
    )

    proxies, post_features = fit_proxy_filter(
        train,
        full_features,
        CFG["proxy_scope"]
    )

    prep = TrainPreprocessor(
        post_features
    ).fit(train)

    Xtr0 = prep.transform(train)
    Xva0 = prep.transform(val)
    Xte0 = prep.transform(test)

    selected, stage_table = hybrid_select(
        Xtr0,
        train["label"].values,
        SEED
    )

    Xtr = Xtr0[selected]
    Xva = Xva0[selected]
    Xte = Xte0[selected]

    best_params, cv_score, nested_cv_table = (
        nested_group_cv_select_params(
            train,
            post_features
        )
    )

    model = make_catboost()
    model.set_params(**best_params)
    model.fit(
        Xtr,
        train["label"].values
    )

    s_val = model.predict_proba(Xva)[:, 1]
    s_test = model.predict_proba(Xte)[:, 1]

    threshold, threshold_table, threshold_rule = (
        choose_threshold_validation(
            val["label"].values,
            s_val
        )
    )

    m_test = metrics(
        test["label"].values,
        s_test,
        threshold
    )

    m_default = metrics(
        test["label"].values,
        s_test,
        0.5
    )

    ci_auc = bootstrap_ci(
        test["label"].values,
        s_test,
        roc_auc_score,
        CFG["n_boot"]
    )

    ci_aupr = bootstrap_ci(
        test["label"].values,
        s_test,
        average_precision_score,
        CFG["n_boot"]
    )

    op_rows = []
    for prevalence in CFG["prevalences"]:
        op_rows.extend(
            operational_metrics(
                test["label"].values,
                s_test,
                threshold,
                prevalence
            )
        )

    return {
        "train": train,
        "val": val,
        "test": test,
        "full_features": full_features,
        "proxies": proxies,
        "post_features": post_features,
        "preprocessor": prep,
        "selected": selected,
        "stage_table": stage_table,
        "model": model,
        "best_params": best_params,
        "nested_cv_score_aupr": cv_score,
        "nested_cv_table": nested_cv_table,
        "s_val": s_val,
        "s_test": s_test,
        "threshold": threshold,
        "threshold_rule": threshold_rule,
        "threshold_table": threshold_table,
        "metrics_test": m_test,
        "metrics_test_default": m_default,
        "ci_auc": ci_auc,
        "ci_aupr": ci_aupr,
        "operational": pd.DataFrame(op_rows),
    }

# =====================================================================
# BASELINES — SAME SPLIT / SAME SELECTED FEATURES
# =====================================================================
def run_baselines(main):
    train, val, test = (
        main["train"],
        main["val"],
        main["test"]
    )

    prep = main["preprocessor"]
    selected = main["selected"]

    Xtr = prep.transform(train)[selected]
    Xva = prep.transform(val)[selected]
    Xte = prep.transform(test)[selected]

    ytr = train.label.values
    yva = val.label.values
    yte = test.label.values

    models = {
        "CatBoost": make_catboost(),
        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1
        ),
        "Logistic Regression": LogisticRegression(
            penalty="l2",
            class_weight="balanced",
            max_iter=3000,
            random_state=SEED
        ),
        "HistGradientBoosting":
            HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.05,
                max_leaf_nodes=31,
                random_state=SEED
            )
    }

    rows = []
    scores = {}

    for name, model in models.items():
        if name == "CatBoost":
            model.set_params(
                **main["best_params"]
            )

        model.fit(Xtr, ytr)

        s_val = model.predict_proba(Xva)[:, 1]
        s_test = model.predict_proba(Xte)[:, 1]

        threshold, _, rule = (
            choose_threshold_validation(
                yva,
                s_val
            )
        )

        m = metrics(
            yte,
            s_test,
            threshold
        )

        rows.append({
            "model": name,
            "threshold": threshold,
            "threshold_rule": rule,
            **m
        })

        scores[name] = s_test

    return pd.DataFrame(rows), scores

# =====================================================================
# ABLATION
# =====================================================================
def run_ablation(main):
    train, val, test = (
        main["train"],
        main["val"],
        main["test"]
    )

    prep = main["preprocessor"]

    Xtr0 = prep.transform(train)
    Xva0 = prep.transform(val)
    Xte0 = prep.transform(test)

    ytr = train.label.values
    yva = val.label.values
    yte = test.label.values

    candidates = {}

    # Post-filter space
    candidates["Post-filter only"] = list(
        Xtr0.columns
    )

    # Variance
    vt = VarianceThreshold(
        CFG["variance_threshold"]
    ).fit(Xtr0)

    candidates["Variance"] = list(
        Xtr0.columns[vt.get_support()]
    )

    # Variance + MI
    fv = candidates["Variance"]

    mi = mutual_info_classif(
        Xtr0[fv],
        ytr,
        random_state=SEED
    )

    keep = np.argsort(mi)[::-1][
        :min(CFG["mi_k"], len(fv))
    ]

    candidates["Variance + MI"] = [
        fv[i] for i in keep
    ]

    # Full hybrid
    selected, _ = hybrid_select(
        Xtr0,
        ytr,
        SEED
    )

    candidates["Full hybrid"] = selected

    rows = []
    scores = {}

    for name, features in candidates.items():
        model = make_catboost()
        model.set_params(
            **main["best_params"]
        )
        model.fit(
            Xtr0[features],
            ytr
        )

        s_val = model.predict_proba(
            Xva0[features]
        )[:, 1]

        s_test = model.predict_proba(
            Xte0[features]
        )[:, 1]

        threshold, _, rule = (
            choose_threshold_validation(
                yva,
                s_val
            )
        )

        m = metrics(
            yte,
            s_test,
            threshold
        )

        rows.append({
            "ablation": name,
            "n_features": len(features),
            "threshold": threshold,
            "threshold_rule": rule,
            **m
        })

        scores[name] = s_test

    return pd.DataFrame(rows), scores

# =====================================================================
# TEMPORAL VALIDATION
# =====================================================================
def temporal_validation(df):
    dates = pd.to_datetime(
        df["t_p"],
        utc=True,
        errors="coerce"
    )

    cutoff = dates.quantile(0.50)

    early = df[dates < cutoff].copy()
    late = df[dates >= cutoff].copy()

    # Validation is drawn from the historical period only.
    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=CFG["val_frac_of_trainval"],
        random_state=SEED + 10
    )

    i_tr, i_va = next(
        gss.split(
            early,
            early.label,
            groups=early.group_id
        )
    )

    train = early.iloc[i_tr].copy()
    val = early.iloc[i_va].copy()

    # Conservative unseen-group future test.
    seen = (
        set(train.group_id) |
        set(val.group_id)
    )

    test = late[
        ~late.group_id.isin(seen)
    ].copy()

    if test.label.nunique() < 2:
        return None, {
            "reason":
                "temporal test has fewer than two classes "
                "after unseen-group filtering",
            "cutoff": str(cutoff)
        }

    full_features = list(
        numeric_features(train).columns
    )

    _, post = fit_proxy_filter(
        train,
        full_features,
        CFG["proxy_scope"]
    )

    prep = TrainPreprocessor(
        post
    ).fit(train)

    Xtr0 = prep.transform(train)
    Xva0 = prep.transform(val)
    Xte0 = prep.transform(test)

    selected, stage = hybrid_select(
        Xtr0,
        train.label.values,
        SEED
    )

    best_hp, cv_score, _ = (
        nested_group_cv_select_params(
            train,
            post
        )
    )

    model = make_catboost()
    model.set_params(**best_hp)
    model.fit(
        Xtr0[selected],
        train.label.values
    )

    s_val = model.predict_proba(
        Xva0[selected]
    )[:, 1]

    s_test = model.predict_proba(
        Xte0[selected]
    )[:, 1]

    threshold, _, rule = (
        choose_threshold_validation(
            val.label.values,
            s_val
        )
    )

    m = metrics(
        test.label.values,
        s_test,
        threshold
    )

    return {
        "status": "ok",
        "cutoff": str(cutoff),
        "train_n": len(train),
        "validation_n": len(val),
        "test_n": len(test),
        "train_end": str(
            pd.to_datetime(train.t_p).max()
        ),
        "test_start": str(
            pd.to_datetime(test.t_p).min()
        ),
        "threshold": threshold,
        "threshold_rule": rule,
        "selected": selected,
        "metrics": m,
        "auc_ci": bootstrap_ci(
            test.label.values,
            s_test,
            roc_auc_score,
            1000
        ),
        "aupr_ci": bootstrap_ci(
            test.label.values,
            s_test,
            average_precision_score,
            1000
        )
    }, None

# =====================================================================
# CROSS-ECOSYSTEM TRANSFER
# =====================================================================
def cross_ecosystem(df, source, target):
    source_df = df[
        df.ecosystem == source
    ].copy()

    target_df = df[
        df.ecosystem == target
    ].copy()

    if len(source_df) < 100 or len(target_df) < 100:
        return {
            "source": source,
            "target": target,
            "status": "insufficient_data"
        }

    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=CFG["val_frac_of_trainval"],
        random_state=SEED + 20
    )

    i_tr, i_va = next(
        gss.split(
            source_df,
            source_df.label,
            groups=source_df.group_id
        )
    )

    train = source_df.iloc[i_tr].copy()
    val = source_df.iloc[i_va].copy()

    overlap = (
        set(train.group_id) &
        set(target_df.group_id)
    )

    test = target_df[
        ~target_df.group_id.isin(overlap)
    ].copy()

    if test.label.nunique() < 2:
        return {
            "source": source,
            "target": target,
            "status": "target_has_fewer_than_two_classes"
        }

    full_features = list(
        numeric_features(train).columns
    )

    # Proxy decision is learned from SOURCE only.
    _, post = fit_proxy_filter(
        train,
        full_features,
        CFG["proxy_scope"]
    )

    prep = TrainPreprocessor(
        post
    ).fit(train)

    Xtr0 = prep.transform(train)
    Xva0 = prep.transform(val)
    Xte0 = prep.transform(test)

    selected, _ = hybrid_select(
        Xtr0,
        train.label.values,
        SEED
    )

    best_hp, cv_score, _ = (
        nested_group_cv_select_params(
            train,
            post
        )
    )

    model = make_catboost()
    model.set_params(**best_hp)
    model.fit(
        Xtr0[selected],
        train.label.values
    )

    s_val = model.predict_proba(
        Xva0[selected]
    )[:, 1]

    s_test = model.predict_proba(
        Xte0[selected]
    )[:, 1]

    threshold, _, rule = (
        choose_threshold_validation(
            val.label.values,
            s_val
        )
    )

    m = metrics(
        test.label.values,
        s_test,
        threshold
    )

    return {
        "source": source,
        "target": target,
        "status": "ok",
        "train_n": len(train),
        "validation_n": len(val),
        "test_n": len(test),
        "target_group_overlap_removed": len(overlap),
        "selected_k": len(selected),
        "threshold": threshold,
        "metrics": m,
        "auc_ci_low": bootstrap_ci(
            test.label.values,
            s_test,
            roc_auc_score,
            1000
        )[0],
        "auc_ci_high": bootstrap_ci(
            test.label.values,
            s_test,
            roc_auc_score,
            1000
        )[1],
        "aupr_ci_low": bootstrap_ci(
            test.label.values,
            s_test,
            average_precision_score,
            1000
        )[0],
        "aupr_ci_high": bootstrap_ci(
            test.label.values,
            s_test,
            average_precision_score,
            1000
        )[1],
    }

# =====================================================================
# SHAP
# =====================================================================
def run_shap(main):
    if not SHAP_AVAILABLE:
        return {
            "status": "shap_not_installed"
        }

    test = main["test"]
    X_test = (
        main["preprocessor"]
        .transform(test)[main["selected"]]
    )

    model = main["model"]

    n = min(1000, len(X_test))
    sample = X_test.sample(
        n=n,
        random_state=SEED
    )

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    if isinstance(shap_values, list):
        shap_values = shap_values[-1]

    shap_values = np.asarray(shap_values)

    global_imp = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=sample.columns
    ).sort_values(ascending=False)

    global_imp.to_csv(
        OUT / "table_shap_global.csv",
        header=["mean_abs_shap"]
    )

    # Global bar
    plt.figure(figsize=(8, 6))
    global_imp.head(18).sort_values().plot(
        kind="barh"
    )
    plt.xlabel("Mean absolute SHAP value")
    plt.ylabel("Feature")
    plt.title("Global SHAP feature importance")
    plt.tight_layout()
    plt.savefig(
        OUT / "figure_shap_global.png",
        dpi=300
    )
    plt.close()

    # Beeswarm
    try:
        shap.summary_plot(
            shap_values,
            sample,
            show=False,
            max_display=min(
                18,
                len(sample.columns)
            )
        )
        plt.tight_layout()
        plt.savefig(
            OUT / "figure_shap_beeswarm.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()
    except Exception:
        pass

    # Local TP / FP / FN / TN cases
    scores = main["s_test"]
    y = test.label.values
    pred = (
        scores >= main["threshold"]
    ).astype(int)

    masks = {
        "TP": (y == 1) & (pred == 1),
        "FP": (y == 0) & (pred == 1),
        "FN": (y == 1) & (pred == 0),
        "TN": (y == 0) & (pred == 0),
    }

    cases = {}

    for label, mask in masks.items():
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue

        if label in ("TP", "FP"):
            j = idx[np.argmax(scores[idx])]
        else:
            j = idx[np.argmin(scores[idx])]

        row = (
            main["preprocessor"]
            .transform(test.iloc[[j]])[
                main["selected"]
            ]
        )

        cases[label] = {
            "row_index": int(test.index[j]),
            "package": str(test.iloc[j]["package"]),
            "version": str(test.iloc[j]["version"]),
            "true_label": int(y[j]),
            "score": float(scores[j]),
            "shap_values": row.iloc[0].to_dict()
        }

    with open(
        OUT / "shap_case_studies.json",
        "w"
    ) as f:
        json.dump(
            cases,
            f,
            indent=2
        )

    # Interaction analysis on a smaller independent explanation sample.
    try:
        inter_sample = sample.iloc[
            :min(300, len(sample))
        ]

        interaction = (
            explainer
            .shap_interaction_values(
                inter_sample
            )
        )

        if isinstance(interaction, list):
            interaction = interaction[-1]

        interaction = np.asarray(
            interaction
        )

        strength = np.mean(
            np.abs(interaction),
            axis=0
        )

        np.fill_diagonal(
            strength,
            0
        )

        pairs = []

        for i in range(
            strength.shape[0]
        ):
            for j in range(
                i + 1,
                strength.shape[1]
            ):
                pairs.append((
                    sample.columns[i],
                    sample.columns[j],
                    strength[i, j]
                ))

        pairs = sorted(
            pairs,
            key=lambda x: x[2],
            reverse=True
        )[:20]

        pd.DataFrame(
            pairs,
            columns=[
                "feature_1",
                "feature_2",
                "mean_abs_interaction"
            ]
        ).to_csv(
            OUT / "table_shap_interactions.csv",
            index=False
        )

    except Exception as exc:
        with open(
            OUT / "shap_interaction_status.json",
            "w"
        ) as f:
            json.dump(
                {"status": "failed",
                 "reason": str(exc)},
                f,
                indent=2
            )

    # Explanation stability
    rng = np.random.default_rng(SEED)
    top_sets = []

    for run in range(
        CFG["n_shap_stability"]
    ):
        idx = rng.choice(
            len(X_test),
            size=min(500, len(X_test)),
            replace=False
        )

        sv = explainer.shap_values(
            X_test.iloc[idx]
        )

        if isinstance(sv, list):
            sv = sv[-1]

        importance = np.abs(
            np.asarray(sv)
        ).mean(axis=0)

        top = list(
            pd.Series(
                importance,
                index=X_test.columns
            )
            .sort_values(ascending=False)
            .head(10)
            .index
        )

        top_sets.append(top)

    jaccards = [
        jaccard(a, b)
        for a, b in itertools.combinations(
            top_sets, 2
        )
    ]

    pd.DataFrame({
        "metric": [
            "mean_top10_jaccard",
            "runs"
        ],
        "value": [
            float(np.mean(jaccards))
            if jaccards else np.nan,
            len(top_sets)
        ]
    }).to_csv(
        OUT / "table_shap_stability.csv",
        index=False
    )

    return {
        "status": "ok",
        "mean_top10_jaccard":
            float(np.mean(jaccards))
            if jaccards else np.nan,
        "case_studies": cases,
        "global_importance":
            global_imp.to_dict()
    }

# =====================================================================
# FIGURES
# =====================================================================
def save_confusion_matrix(y, score, threshold):
    pred = (
        score >= threshold
    ).astype(int)

    cm = confusion_matrix(
        y,
        pred,
        labels=[0, 1]
    )

    plt.figure(figsize=(5, 4))
    plt.imshow(cm)

    plt.xticks(
        [0, 1],
        ["Benign", "Malicious"]
    )
    plt.yticks(
        [0, 1],
        ["Benign", "Malicious"]
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(
        "Confusion matrix — final test set"
    )

    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                int(cm[i, j]),
                ha="center",
                va="center"
            )

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(
        OUT / "figure_confusion_matrix.png",
        dpi=300
    )
    plt.close()


def save_pr_roc(y, score):
    precision, recall, _ = (
        precision_recall_curve(
            y,
            score
        )
    )

    fpr, tpr, _ = roc_curve(
        y,
        score
    )

    plt.figure(figsize=(6, 5))
    plt.plot(
        recall,
        precision
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(
        "Precision–Recall curve"
    )
    plt.tight_layout()
    plt.savefig(
        OUT / "figure_precision_recall.png",
        dpi=300
    )
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.plot(
        fpr,
        tpr
    )
    plt.plot(
        [0, 1],
        [0, 1],
        "--"
    )
    plt.xlabel(
        "False-positive rate"
    )
    plt.ylabel(
        "True-positive rate"
    )
    plt.title("ROC curve")
    plt.tight_layout()
    plt.savefig(
        OUT / "figure_roc.png",
        dpi=300
    )
    plt.close()


def save_calibration(y, score):
    bins = np.linspace(
        0, 1, 11
    )

    rows = []

    for lo, hi in zip(
        bins[:-1],
        bins[1:]
    ):
        mask = (
            (score >= lo) &
            (
                score < hi
                if hi < 1
                else score <= hi
            )
        )

        if mask.sum():
            rows.append({
                "bin_center":
                    (lo + hi) / 2,
                "mean_predicted":
                    float(np.mean(score[mask])),
                "observed_rate":
                    float(np.mean(y[mask])),
                "n": int(mask.sum())
            })

    table = pd.DataFrame(rows)

    table.to_csv(
        OUT / "table_calibration.csv",
        index=False
    )

    plt.figure(figsize=(6, 5))
    plt.plot(
        table["mean_predicted"],
        table["observed_rate"],
        marker="o"
    )
    plt.plot(
        [0, 1],
        [0, 1],
        "--"
    )
    plt.xlabel(
        "Mean predicted probability"
    )
    plt.ylabel(
        "Observed frequency"
    )
    plt.title(
        "Calibration curve"
    )
    plt.tight_layout()
    plt.savefig(
        OUT / "figure_calibration.png",
        dpi=300
    )
    plt.close()


def save_feature_stages(stage_table):
    x = np.arange(
        len(stage_table)
    )

    plt.figure(figsize=(9, 5))
    plt.plot(
        x,
        stage_table["features_out"],
        marker="o"
    )

    plt.xticks(
        x,
        stage_table["stage"],
        rotation=35,
        ha="right"
    )

    plt.ylabel(
        "Number of retained features"
    )
    plt.title(
        "Feature-selection stages"
    )
    plt.tight_layout()
    plt.savefig(
        OUT / "figure_feature_selection_stages.png",
        dpi=300
    )
    plt.close()


def save_operational_ppv(op_table):
    upper = op_table[
        op_table.fpr_type == "upper95"
    ].copy()

    plt.figure(figsize=(7, 5))
    plt.plot(
        upper["prevalence"] * 100,
        upper["ppv"] * 100,
        marker="o"
    )

    plt.xscale("log")
    plt.xlabel(
        "Assumed malicious prevalence (%)"
    )
    plt.ylabel("PPV (%)")
    plt.title(
        "Operational PPV under low prevalence"
    )
    plt.tight_layout()
    plt.savefig(
        OUT / "figure_operational_ppv.png",
        dpi=300
    )
    plt.close()

# =====================================================================
# TABLES
# =====================================================================
def save_tables(main, baselines, ablation,
                temporal_result, cross_results,
                statistical_table):

    # Table: dataset and split
    pd.DataFrame([{
        "total_n": (
            len(main["train"]) +
            len(main["val"]) +
            len(main["test"])
        ),
        "train_n": len(main["train"]),
        "validation_n": len(main["val"]),
        "test_n": len(main["test"]),
        "train_malicious_rate":
            main["train"].label.mean(),
        "validation_malicious_rate":
            main["val"].label.mean(),
        "test_malicious_rate":
            main["test"].label.mean(),
        "train_groups":
            main["train"].group_id.nunique(),
        "validation_groups":
            main["val"].group_id.nunique(),
        "test_groups":
            main["test"].group_id.nunique(),
        "full_features":
            len(main["full_features"]),
        "proxy_removed":
            len(main["proxies"]),
        "postfilter_features":
            len(main["post_features"]),
        "selected_features":
            len(main["selected"])
    }]).to_csv(
        OUT / "table_dataset_split.csv",
        index=False
    )

    # Feature stages
    main["stage_table"].to_csv(
        OUT / "table_feature_selection_stages.csv",
        index=False
    )

    # Selected features
    pd.DataFrame({
        "rank": np.arange(
            1,
            len(main["selected"]) + 1
        ),
        "feature": main["selected"]
    }).to_csv(
        OUT / "table_selected_features.csv",
        index=False
    )

    # Main test performance
    m = main["metrics_test"]

    pd.DataFrame([{
        "threshold":
            main["threshold"],
        "threshold_rule":
            main["threshold_rule"],
        "accuracy":
            m["accuracy"],
        "precision":
            m["precision"],
        "recall":
            m["recall"],
        "f1":
            m["f1"],
        "mcc":
            m["mcc"],
        "auc_roc":
            m["auc_roc"],
        "auc_roc_ci_low":
            main["ci_auc"][0],
        "auc_roc_ci_high":
            main["ci_auc"][1],
        "au_pr":
            m["au_pr"],
        "au_pr_ci_low":
            main["ci_aupr"][0],
        "au_pr_ci_high":
            main["ci_aupr"][1],
        "brier":
            m["brier"],
        "TN": m["TN"],
        "FP": m["FP"],
        "FN": m["FN"],
        "TP": m["TP"]
    }]).to_csv(
        OUT / "table_main_test_performance.csv",
        index=False
    )

    # Nested CV hyperparameter results
    main["nested_cv_table"].to_csv(
        OUT / "table_nested_cv.csv",
        index=False
    )

    # Baselines
    baselines.to_csv(
        OUT / "table_baseline_comparison.csv",
        index=False
    )

    # Ablation
    ablation.to_csv(
        OUT / "table_ablation.csv",
        index=False
    )

    # Temporal
    if temporal_result is not None:
        pd.DataFrame([{
            "status":
                temporal_result["status"],
            "cutoff":
                temporal_result["cutoff"],
            "train_n":
                temporal_result["train_n"],
            "validation_n":
                temporal_result["validation_n"],
            "test_n":
                temporal_result["test_n"],
            "threshold":
                temporal_result["threshold"],
            **temporal_result["metrics"],
            "auc_ci_low":
                temporal_result["auc_ci"][0],
            "auc_ci_high":
                temporal_result["auc_ci"][1],
            "aupr_ci_low":
                temporal_result["aupr_ci"][0],
            "aupr_ci_high":
                temporal_result["aupr_ci"][1]
        }]).to_csv(
            OUT / "table_temporal_validation.csv",
            index=False
        )

    # Cross ecosystem
    pd.DataFrame(
        cross_results
    ).to_csv(
        OUT / "table_cross_ecosystem.csv",
        index=False
    )

    # Operational
    main["operational"].to_csv(
        OUT / "table_operational_prevalence.csv",
        index=False
    )

    # Statistics
    statistical_table.to_csv(
        OUT / "table_statistical_comparisons.csv",
        index=False
    )

# =====================================================================
# MANIFEST
# =====================================================================
def save_manifest(df, input_path):
    manifest = {
        "generated_at_utc":
            datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn":
            __import__("sklearn").__version__,
        "catboost":
            __import__("catboost").__version__,
        "shap_available":
            SHAP_AVAILABLE,
        "input_csv":
            str(input_path),
        "input_sha256":
            sha256_file(input_path),
        "rows_loaded":
            len(df),
        "columns_loaded":
            len(df.columns),
        "config": CFG
    }

    with open(
        OUT / "run_manifest.json",
        "w"
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
            default=str
        )

# =====================================================================
# MAIN
# =====================================================================
def main():
    print("=" * 80)
    print("IJECES FINAL LEAKAGE-SAFE PIPELINE")
    print("=" * 80)

    input_path = Path(
        CFG["input_csv"]
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {input_path}"
        )

    df = pd.read_csv(
        input_path,
        low_memory=False
    )

    required = {
        "package",
        "version",
        "ecosystem",
        "t_p",
        "label"
    }

    missing = (
        required -
        set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df["t_p"] = pd.to_datetime(
        df["t_p"],
        utc=True,
        errors="coerce"
    )

    df = df.dropna(
        subset=["t_p", "label"]
    ).copy()

    df["label"] = df["label"].astype(int)

    # No label-derived temporal window here.
    # The input must be point-in-time reconstructed upstream.
    df, similar_pairs = build_groups(df)

    before = len(df)
    df = df.drop_duplicates().reset_index(
        drop=True
    )
    duplicates_removed = before - len(df)

    save_manifest(
        df,
        input_path
    )

    if CFG["save_package_ids"]:
        cols = [
            c for c in [
                "package",
                "version",
                "ecosystem",
                "t_p",
                "label",
                "group_id"
            ]
            if c in df.columns
        ]

        df[cols].to_csv(
            OUT / "dataset_package_manifest.csv",
            index=False
        )

    print(
        f"Loaded: {before} rows"
    )
    print(
        f"Exact duplicates removed: "
        f"{duplicates_removed}"
    )
    print(
        f"Groups: {df.group_id.nunique()}"
    )
    print(
        f"Ecosystems: "
        f"{df.ecosystem.value_counts().to_dict()}"
    )
    print(
        f"Similar-name pairs merged: "
        f"{similar_pairs}"
    )

    # ---------------------------------------------------------------
    # MAIN PROTOCOL
    # ---------------------------------------------------------------
    main_result = fit_main_protocol(
        df
    )

    print("\nMAIN PROTOCOL")
    print(
        "Selected features:",
        main_result["selected"]
    )
    print(
        "Best parameters:",
        main_result["best_params"]
    )
    print(
        "Validation-selected threshold:",
        main_result["threshold"]
    )
    print(
        "TEST:",
        main_result["metrics_test"]
    )

    # ---------------------------------------------------------------
    # BASELINES
    # ---------------------------------------------------------------
    baselines, baseline_scores = (
        run_baselines(
            main_result
        )
    )

    # ---------------------------------------------------------------
    # ABLATION
    # ---------------------------------------------------------------
    ablation, ablation_scores = (
        run_ablation(
            main_result
        )
    )

    # ---------------------------------------------------------------
    # TEMPORAL
    # ---------------------------------------------------------------
    temporal_result, temporal_error = (
        temporal_validation(df)
    )

    if temporal_error:
        with open(
            OUT / "temporal_status.json",
            "w"
        ) as f:
            json.dump(
                temporal_error,
                f,
                indent=2
            )

    # ---------------------------------------------------------------
    # CROSS ECOSYSTEM
    # ---------------------------------------------------------------
    ecosystems = sorted(
        df.ecosystem.dropna().unique()
    )

    cross_results = []

    if {"npm", "pypi"}.issubset(
        set(ecosystems)
    ):
        cross_results.append(
            cross_ecosystem(
                df,
                "npm",
                "pypi"
            )
        )
        cross_results.append(
            cross_ecosystem(
                df,
                "pypi",
                "npm"
            )
        )
    else:
        cross_results.append({
            "status":
                "npm_and_pypi_not_both_present",
            "ecosystems":
                ecosystems
        })

    # ---------------------------------------------------------------
    # SHAP
    # ---------------------------------------------------------------
    shap_result = run_shap(
        main_result
    )

    # ---------------------------------------------------------------
    # STATISTICS
    # ---------------------------------------------------------------
    y_test = (
        main_result["test"].label.values
    )

    main_score = (
        main_result["s_test"]
    )

    main_pred = (
        main_score >=
        main_result["threshold"]
    ).astype(int)

    statistical_rows = []

    for name, score in baseline_scores.items():
        if name == "CatBoost":
            continue

        threshold = float(
            baselines.loc[
                baselines.model == name,
                "threshold"
            ].iloc[0]
        )

        pred = (
            score >= threshold
        ).astype(int)

        mc = mcnemar_exact(
            y_test,
            main_pred,
            pred
        )

        pb = paired_bootstrap_delta(
            y_test,
            main_score,
            score,
            CFG["n_paired_boot"]
        )

        statistical_rows.append({
            "comparison":
                "Main CatBoost vs " + name,
            **mc,
            **pb
        })

    statistical_table = pd.DataFrame(
        statistical_rows
    )

    # ---------------------------------------------------------------
    # FIGURES
    # ---------------------------------------------------------------
    save_confusion_matrix(
        main_result["test"].label.values,
        main_result["s_test"],
        main_result["threshold"]
    )

    save_pr_roc(
        main_result["test"].label.values,
        main_result["s_test"]
    )

    save_calibration(
        main_result["test"].label.values,
        main_result["s_test"]
    )

    save_feature_stages(
        main_result["stage_table"]
    )

    save_operational_ppv(
        main_result["operational"]
    )

    # ---------------------------------------------------------------
    # TABLES
    # ---------------------------------------------------------------
    save_tables(
        main_result,
        baselines,
        ablation,
        temporal_result,
        cross_results,
        statistical_table
    )

    # ---------------------------------------------------------------
    # Final JSON
    # ---------------------------------------------------------------
    results = {
        "main": {
            "selected_features":
                main_result["selected"],
            "threshold":
                main_result["threshold"],
            "threshold_rule":
                main_result["threshold_rule"],
            "best_params":
                main_result["best_params"],
            "nested_cv_score_aupr":
                main_result["nested_cv_score_aupr"],
            "metrics_test":
                main_result["metrics_test"],
            "metrics_test_default":
                main_result["metrics_test_default"],
            "auc_ci":
                main_result["ci_auc"],
            "aupr_ci":
                main_result["ci_aupr"],
            "proxies":
                main_result["proxies"],
            "stage_table":
                main_result[
                    "stage_table"
                ].to_dict("records")
        },
        "baselines":
            baselines.to_dict("records"),
        "ablation":
            ablation.to_dict("records"),
        "temporal":
            temporal_result,
        "cross_ecosystem":
            cross_results,
        "shap":
            shap_result,
        "statistics":
            statistical_table.to_dict("records")
    }

    with open(
        OUT / "ijeces_final_results.json",
        "w"
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
            default=str
        )

    print("\n" + "=" * 80)
    print("PIPELINE FINISHED")
    print("=" * 80)
    print("Output directory:")
    print(OUT)
    print("\nMain tables:")
    for f in [
        "table_dataset_split.csv",
        "table_feature_selection_stages.csv",
        "table_selected_features.csv",
        "table_nested_cv.csv",
        "table_main_test_performance.csv",
        "table_baseline_comparison.csv",
        "table_ablation.csv",
        "table_temporal_validation.csv",
        "table_cross_ecosystem.csv",
        "table_operational_prevalence.csv",
        "table_statistical_comparisons.csv",
        "table_calibration.csv",
        "table_shap_global.csv",
        "table_shap_interactions.csv",
        "table_shap_stability.csv"
    ]:
        print("  ", f)

    print("\nMain figures:")
    for f in [
        "figure_feature_selection_stages.png",
        "figure_confusion_matrix.png",
        "figure_precision_recall.png",
        "figure_roc.png",
        "figure_calibration.png",
        "figure_operational_ppv.png",
        "figure_shap_global.png",
        "figure_shap_beeswarm.png"
    ]:
        print("  ", f)


if __name__ == "__main__":
    main()
