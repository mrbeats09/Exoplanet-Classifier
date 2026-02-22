import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
# Use a non-interactive backend to avoid blocking in headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.isotonic import IsotonicRegression
from catboost import CatBoostClassifier

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None


# -----------------------------
# Helper / Feature Engineering
# -----------------------------

def _longest_run_below(arr: np.ndarray, threshold: float) -> int:
    """Return the length of the longest consecutive run where arr < threshold (ignoring NaNs)."""
    count = 0
    best = 0
    for v in arr:
        if np.isfinite(v) and v < threshold:
            count += 1
            if count > best:
                best = count
        else:
            count = 0
    return best


def _row_stats(arr: np.ndarray) -> Dict[str, float]:
    """Compute robust statistics for a 1D time series (per-row features)."""
    arr = np.asarray(arr, dtype=float)
    arr = np.where(np.isfinite(arr), arr, np.nan)

    med = np.nanmedian(arr)
    mad = np.nanmedian(np.abs(arr - med))
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    p10 = np.nanpercentile(arr, 10) if np.isfinite(arr).any() else np.nan
    p90 = np.nanpercentile(arr, 90) if np.isfinite(arr).any() else np.nan
    q25 = np.nanpercentile(arr, 25) if np.isfinite(arr).any() else np.nan
    iqr = p90 - q25 if np.isfinite(p90) and np.isfinite(q25) else np.nan

    # Linear trend (slope)
    idx = np.arange(len(arr), dtype=float)
    if np.isfinite(arr).sum() >= 2:
        arr_fit = np.copy(arr)
        if np.isnan(arr_fit).any():
            arr_fit[np.isnan(arr_fit)] = med if np.isfinite(med) else 0.0
        slope = np.polyfit(idx, arr_fit, 1)[0]
    else:
        slope = 0.0

    # Skew proxy
    skew_proxy = (mean - med) / (std + 1e-8)

    # Outlier counts (transit-like dips)
    low3mad_th = med - 3 * (mad + 1e-8) if np.isfinite(med) else -np.inf
    low5mad_th = med - 5 * (mad + 1e-8) if np.isfinite(med) else -np.inf
    low3mad = int(np.nansum(arr < low3mad_th))
    low5mad = int(np.nansum(arr < low5mad_th))

    # Dip strength / width oriented toward recall of real transits
    amin = np.nanmin(arr) if np.isfinite(arr).any() else np.nan
    amax = np.nanmax(arr) if np.isfinite(arr).any() else np.nan
    dip_depth = (med - amin) if np.isfinite(med) and np.isfinite(amin) else 0.0
    snr_dip = dip_depth / (mad + 1e-8) if np.isfinite(dip_depth) and np.isfinite(mad) else 0.0
    ptp = (amax - amin) if np.isfinite(amax) and np.isfinite(amin) else 0.0

    # Width at half-min: points below median - 0.5 * dip_depth
    half_min_th = med - 0.5 * dip_depth if np.isfinite(med) and np.isfinite(dip_depth) else -np.inf
    width_half_min = int(np.nansum(arr < half_min_th)) if np.isfinite(half_min_th) else 0

    # Longest consecutive dip below 3*MAD
    longest_low3 = _longest_run_below(arr, low3mad_th) if np.isfinite(low3mad_th) else 0

    return {
        "mean": float(mean) if np.isfinite(mean) else 0.0,
        "std": float(std) if np.isfinite(std) else 0.0,
        "mad": float(mad) if np.isfinite(mad) else 0.0,
        "p10": float(p10) if np.isfinite(p10) else 0.0,
        "p90": float(p90) if np.isfinite(p90) else 0.0,
        "iqr": float(iqr) if np.isfinite(iqr) else 0.0,
        "skew_proxy": float(skew_proxy) if np.isfinite(skew_proxy) else 0.0,
        "slope": float(slope),
        "low3mad": float(low3mad),
        "low5mad": float(low5mad),
        "dip_depth": float(dip_depth) if np.isfinite(dip_depth) else 0.0,
        "snr_dip": float(snr_dip) if np.isfinite(snr_dip) else 0.0,
        "ptp": float(ptp) if np.isfinite(ptp) else 0.0,
        "width_half_min": float(width_half_min),
        "longest_low3": float(longest_low3),
    }


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Pearson correlation, robust to NaNs/constant arrays."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return 0.0
    aa = a[mask]
    bb = b[mask]
    if np.std(aa) < 1e-12 or np.std(bb) < 1e-12:
        return 0.0
    return float(np.corrcoef(aa, bb)[0, 1])


def _engineer_features(X: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, list]]:
    """
    Create additional features aimed at reducing false positives and recovering recall.

    - Forward/backward-fill imputation across time-like axes
    - Flux per-row median normalization
    - Centroids centered and scaled per-row
    - Drop constant features
    - Add robust flux/centroid descriptors (depth/width/SNR, correlations)
    """
    groups = {
        "flux_cols": [c for c in X.columns if c.startswith("f_")],
        "cc_cols": [c for c in X.columns if c.startswith("cc_")],
        "cr_cols": [c for c in X.columns if c.startswith("cr_")],
    }

    X_proc = X.copy()

    # Impute gaps across time-like axes with ffill/bfill per row
    X_proc = X_proc.ffill(axis=1).bfill(axis=1)

    # Normalize flux rows by median to reduce scale variance
    if groups["flux_cols"]:
        flux_median = X_proc[groups["flux_cols"]].median(axis=1).replace(0, np.nan)
        X_proc[groups["flux_cols"]] = X_proc[groups["flux_cols"]].div(flux_median, axis=0)

    # Center and scale centroid tracks per row
    for key in ("cc_cols", "cr_cols"):
        cols = groups[key]
        if cols:
            Xc = X_proc[cols]
            Xc = Xc.sub(Xc.mean(axis=1), axis=0)
            std = Xc.std(axis=1).replace(0, np.nan).add(1e-8)
            X_proc[cols] = Xc.div(std, axis=0)

    # Drop constant (zero-variance) columns to remove redundancies
    col_std = X_proc.std(axis=0, numeric_only=True)
    keep_cols = col_std[(col_std > 0) & np.isfinite(col_std)].index.tolist()
    X_proc = X_proc[keep_cols]

    # Add robust summary features from flux
    eng_feats = {}
    if groups["flux_cols"]:
        fA = X_proc[groups["flux_cols"]].to_numpy()
        stats = [_row_stats(row) for row in fA]
        for k in stats[0].keys():
            eng_feats[f"flux_{k}"] = np.array([s[k] for s in stats], dtype=float)

        # Autocorrelation at lag-1 proxy
        ac1 = []
        for row in fA:
            ac1.append(_safe_corr(row[:-1], row[1:]) if row.size >= 2 else 0.0)
        eng_feats["flux_ac1"] = np.array(ac1, dtype=float)

    # Centroid motion descriptors and correlation with flux (systematics detector)
    if groups["cc_cols"] or groups["cr_cols"]:
        ccA = X_proc[groups["cc_cols"]].to_numpy() if groups["cc_cols"] else None
        crA = X_proc[groups["cr_cols"]].to_numpy() if groups["cr_cols"] else None
        if groups["flux_cols"]:
            fA = X_proc[groups["flux_cols"]].to_numpy()
        else:
            fA = None

        cc_std = np.nanstd(ccA, axis=1) if ccA is not None and ccA.size else np.zeros(len(X_proc))
        cr_std = np.nanstd(crA, axis=1) if crA is not None and crA.size else np.zeros(len(X_proc))
        eng_feats["centroid_std_sum"] = cc_std + cr_std
        eng_feats["centroid_std_diff"] = cc_std - cr_std

        # Correlations of flux with centroid motion (higher implies systematics -> likely FP)
        corr_cc = []
        corr_cr = []
        for i in range(len(X_proc)):
            if fA is None:
                corr_cc.append(0.0)
                corr_cr.append(0.0)
                continue
            fi = fA[i]
            if ccA is not None and ccA.size:
                corr_cc.append(_safe_corr(fi, ccA[i]))
            else:
                corr_cc.append(0.0)
            if crA is not None and crA.size:
                corr_cr.append(_safe_corr(fi, crA[i]))
            else:
                corr_cr.append(0.0)
        eng_feats["corr_flux_cc"] = np.array(corr_cc, dtype=float)
        eng_feats["corr_flux_cr"] = np.array(corr_cr, dtype=float)

    if eng_feats:
        X_eng = pd.DataFrame(eng_feats, index=X_proc.index)
        X_proc = pd.concat([X_proc, X_eng], axis=1)

    return X_proc, groups


# -----------------------------
# Threshold & Veto Search
# -----------------------------

def _apply_veto(
    X: pd.DataFrame,
    prob: np.ndarray,
    threshold: float,
    corr_veto: Optional[float],
    centstd_quantile: Optional[float],
) -> np.ndarray:
    """
    Apply a rule-based veto to demote likely systematics-driven positives based on
    flux-centroid correlation and centroid motion magnitude.
    If required columns are missing or params are None, returns simple thresholding.
    """
    y_pred = (prob >= threshold).astype(int)

    needed_cols = {"corr_flux_cc", "corr_flux_cr", "centroid_std_sum"}
    if not needed_cols.issubset(set(X.columns)):
        return y_pred
    if corr_veto is None or centstd_quantile is None:
        return y_pred

    # Compute quantile cutoff for centroid_std_sum on the full set (robust)
    q_cut = float(np.nanquantile(X["centroid_std_sum"].values, centstd_quantile))

    pos_idx = np.where(y_pred == 1)[0]
    if pos_idx.size == 0:
        return y_pred
    x_pos = X.iloc[pos_idx]

    corr_flag = (x_pos["corr_flux_cc"].values > corr_veto) | (x_pos["corr_flux_cr"].values > corr_veto)
    cent_flag = x_pos["centroid_std_sum"].values > q_cut
    veto_mask = corr_flag & cent_flag

    if veto_mask.any():
        y_pred[pos_idx[veto_mask]] = 0
    return y_pred


def _evaluate_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def _search_threshold_and_veto(
    y_true: np.ndarray,
    prob: np.ndarray,
    X: pd.DataFrame,
    recall_floor: Optional[float] = 0.80,
    beta: float = 0.8,
) -> Tuple[float, Optional[float], Optional[float], Dict[str, float]]:
    """
    Grid-search threshold and veto parameters to directly minimize total errors (FP+FN)
    subject to an optional recall floor. Falls back to best F-beta if needed.
    Returns: (threshold, corr_veto, centstd_quantile, metrics)
    """
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob).astype(float)

    # Threshold candidates from unique probabilities plus a small uniform grid
    thr_candidates = np.unique(np.clip(prob, 0, 1))
    if thr_candidates.size > 100:
        # subsample for efficiency while covering the range
        thr_candidates = np.linspace(thr_candidates.min(), thr_candidates.max(), 200)
    thr_candidates = np.concatenate(([0.0], thr_candidates, [1.0]))

    # Veto grids (include None to allow no-veto case)
    have_cols = {"corr_flux_cc", "corr_flux_cr", "centroid_std_sum"}.issubset(set(X.columns))
    corr_grid = [None] if not have_cols else [None, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    q_grid = [None] if not have_cols else [None, 0.7, 0.8, 0.9]

    best = {
        "score": np.inf,  # minimize FP+FN
        "precision": 0.0,
        "recall": 0.0,
        "fbeta": -1.0,
        "thr": 0.5,
        "corr_veto": None,
        "cent_q": None,
        "fp": None,
        "fn": None,
    }

    fbeta_best = {"fbeta": -1.0, "thr": 0.5, "corr_veto": None, "cent_q": None}

    for t in thr_candidates:
        base_pred = (prob >= t).astype(int)
        # Fast path compute base metrics
        p, r, f, _ = precision_recall_fscore_support(y_true, base_pred, beta=beta, average="binary", zero_division=0)
        if f > fbeta_best["fbeta"]:
            fbeta_best = {"fbeta": float(f), "thr": float(t), "corr_veto": None, "cent_q": None}

        for cv in corr_grid:
            for qv in q_grid:
                y_pred = _apply_veto(X, prob, t, cv, qv)
                counts = _evaluate_counts(y_true, y_pred)
                total_err = counts["fp"] + counts["fn"]
                p2, r2, f2, _ = precision_recall_fscore_support(y_true, y_pred, beta=beta, average="binary", zero_division=0)

                # Enforce recall floor if provided
                if recall_floor is not None and r2 < recall_floor:
                    continue

                # Prefer fewer total errors, then fewer FP, then higher precision, then higher recall
                better = False
                if total_err < best["score"]:
                    better = True
                elif total_err == best["score"]:
                    if counts["fp"] < (best["fp"] if best["fp"] is not None else 1e9):
                        better = True
                    elif counts["fp"] == (best["fp"] if best["fp"] is not None else 1e9):
                        if p2 > best["precision"]:
                            better = True
                        elif p2 == best["precision"] and r2 > best["recall"]:
                            better = True

                if better:
                    best.update(
                        {
                            "score": int(total_err),
                            "precision": float(p2),
                            "recall": float(r2),
                            "fbeta": float(f2),
                            "thr": float(t),
                            "corr_veto": None if cv is None else float(cv),
                            "cent_q": None if qv is None else float(qv),
                            "fp": int(counts["fp"]),
                            "fn": int(counts["fn"]),
                        }
                    )

    # If no candidate meets recall floor, fall back to best F-beta without veto
    if best["fp"] is None:
        t = fbeta_best["thr"]
        y_pred = (prob >= t).astype(int)
        counts = _evaluate_counts(y_true, y_pred)
        p2, r2, f2, _ = precision_recall_fscore_support(y_true, y_pred, beta=beta, average="binary", zero_division=0)
        return t, None, None, {
            "precision": float(p2),
            "recall": float(r2),
            "fbeta": float(f2),
            "fp": int(counts["fp"]),
            "fn": int(counts["fn"]),
            "total_error": int(counts["fp"] + counts["fn"]),
        }

    return (
        best["thr"],
        best["corr_veto"],
        best["cent_q"],
        {
            "precision": best["precision"],
            "recall": best["recall"],
            "fbeta": best["fbeta"],
            "fp": best["fp"],
            "fn": best["fn"],
            "total_error": best["score"],
        },
    )


# -----------------------------
# Main Training Function
# -----------------------------

def train_model(
    data_path: str = "tess_training_data.csv",
    save_fig: str = "confusion_matrix.png",
    save_model_path: str = "catboost_exoplanet.cbm",
    show_plot: bool = False,
    verbose: int = 100,
    n_splits: int = 5,
    beta_threshold: float = 0.8,  # recall leaning for fallback
    recall_floor: float = 0.80,   # primary constraint in search
    save_threshold_path: str = "catboost_threshold.json",
    save_calibrator_path: str = "calibrator_isotonic.pkl",
    neg_class_weight: float = 1.2,  # now bias toward negatives to curb FPs
    pos_class_weight: float = 1.0,
    bagging_temperature: float = 1.0,
    rsm: float = 0.9,  # random subspace sampling of features
):
    """
    Train CatBoost with feature engineering, isotonic calibration, and a
    threshold+veto search that minimizes total FP+FN subject to a recall floor.

    Goals: further reduce false positives while maintaining high recall, aiming
    for minimal combined misclassifications.
    """
    print("1. Loading and Validating Data...")

    if not os.path.isfile(data_path):
        raise FileNotFoundError(
            f"Data file not found: {data_path}. Ensure the CSV exists and the path is correct."
        )

    df = pd.read_csv(data_path)

    if "label" not in df.columns:
        raise ValueError("Required column 'label' not found in dataset.")

    label_mapping = {"KP": 1, "CP": 1, "FP": 0}
    y = df["label"].map(label_mapping)

    valid_mask = y.notna()
    if valid_mask.sum() != len(df):
        print(f"   Warning: Dropping {len(df) - valid_mask.sum()} rows with unrecognized/NaN labels.")
    df = df.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].astype(int).reset_index(drop=True)

    drop_cols = [c for c in ["label", "tic_id"] if c in df.columns]
    X_raw = df.drop(columns=drop_cols)

    # Feature engineering and preprocessing
    print("2. Preprocessing + Feature Engineering...")
    X, groups = _engineer_features(X_raw)

    print(f"   Flux feature cols: {len(groups['flux_cols'])}")
    print(f"   Column-centroid cols: {len(groups['cc_cols'])}")
    print(f"   Row-centroid cols: {len(groups['cr_cols'])}")
    print(f"   Final feature count (after engineering): {X.shape[1]}")
    class_counts = y.value_counts().to_dict()
    print(f"   Class distribution: {class_counts}")

    # Prepare CV
    print("3. Stratified K-Fold Training...")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_prob = np.zeros(len(X), dtype=float)
    best_iters = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        print(f"   Fold {fold}/{n_splits}...")
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        model = CatBoostClassifier(
            iterations=1500,
            learning_rate=0.03,
            depth=5,
            l2_leaf_reg=7,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=42 + fold,  # variation across folds
            verbose=verbose,
            class_weights=[neg_class_weight, pos_class_weight],
            bootstrap_type="Bayesian",
            bagging_temperature=bagging_temperature,
            rsm=rsm,
            random_strength=1.0,
        )

        model.fit(
            X_tr,
            y_tr,
            eval_set=(X_va, y_va),
            early_stopping_rounds=150,
            use_best_model=True,
        )

        best_iters.append(model.get_best_iteration())
        prob_va = model.predict_proba(X_va)[:, 1]
        oof_prob[va_idx] = prob_va

    # Calibrate probabilities using isotonic regression on OOF predictions
    print("4. Calibrating probabilities (Isotonic Regression)...")
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(oof_prob, y.values)
    oof_prob_cal = iso.predict(oof_prob)

    # Search threshold and veto settings to minimize FP+FN under recall floor
    print("5. Selecting Threshold and Veto from OOF calibrated predictions...")
    thr, corr_veto, cent_q, metrics = _search_threshold_and_veto(
        y_true=y.values, prob=oof_prob_cal, X=X, recall_floor=recall_floor, beta=beta_threshold
    )

    # Final OOF evaluation with chosen settings
    y_oof_pred = _apply_veto(X, oof_prob_cal, thr, corr_veto, cent_q)
    oof_acc = accuracy_score(y, y_oof_pred)
    cm = confusion_matrix(y, y_oof_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    print(
        f"   Chosen thr: {thr:.4f} | corr_veto: {corr_veto} | cent_q: {cent_q} | "
        f"P: {metrics['precision']:.3f} R: {metrics['recall']:.3f} F{beta_threshold:.1f}: {metrics['fbeta']:.3f}"
    )
    print(f"   OOF Acc: {oof_acc * 100:.2f}% | FP: {fp} | FN: {fn} | Total errors: {fp + fn}")

    # Plot confusion matrix
    print("6. Generating Confusion Matrix (OOF)...")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["False Positive (0)", "Planet (1)"])
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Purples, ax=ax)
    plt.title(
        "CatBoost TESS Exoplanet Classifier\n"
        f"OOF Acc: {oof_acc * 100:.2f}% | Thr: {thr:.3f} | P: {metrics['precision']:.3f} R: {metrics['recall']:.3f} | FP:{fp} FN:{fn}"
    )
    plt.tight_layout()
    try:
        plt.savefig(save_fig, dpi=150)
        print(f"   Confusion matrix saved to: {save_fig}")
    except Exception as e:
        print(f"   Warning: Failed to save figure to {save_fig}: {e}")
    plt.close(fig)

    # Train final model on all data with median best iterations
    print("7. Training Final Model on All Data...")
    final_iterations = int(np.median(best_iters)) if best_iters else 800
    final_iterations = max(300, final_iterations)

    final_model = CatBoostClassifier(
        iterations=final_iterations,
        learning_rate=0.03,
        depth=5,
        l2_leaf_reg=7,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=verbose,
        class_weights=[neg_class_weight, pos_class_weight],
        bootstrap_type="Bayesian",
        bagging_temperature=bagging_temperature,
        rsm=rsm,
        random_strength=1.0,
    )

    final_model.fit(X, y, verbose=verbose)

    # Persist artifacts
    try:
        final_model.save_model(save_model_path)
        print(f"8. Model saved to: {save_model_path}")
    except Exception as e:
        print(f"   Warning: Failed to save model to {save_model_path}: {e}")

    # Save threshold and veto parameters for downstream inference
    threshold_payload = {
        "threshold": float(thr),
        "corr_veto": None if corr_veto is None else float(corr_veto),
        "centroid_std_quantile": None if cent_q is None else float(cent_q),
    }
    try:
        with open(save_threshold_path, "w") as f:
            json.dump(threshold_payload, f, indent=2)
        print(f"   Threshold and veto params saved to: {save_threshold_path}")
    except Exception as e:
        print(f"   Warning: Failed to save threshold to {save_threshold_path}: {e}")

    # Save calibrator
    try:
        if joblib is not None:
            joblib.dump(iso, save_calibrator_path)
        else:
            with open(save_calibrator_path, "wb") as f:
                pickle.dump(iso, f)
        print(f"   Calibrator saved to: {save_calibrator_path}")
    except Exception as e:
        print(f"   Warning: Failed to save calibrator to {save_calibrator_path}: {e}")

    results = {
        "accuracy": float(oof_acc),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f_beta": float(metrics["fbeta"]),
        "fp": int(fp),
        "fn": int(fn),
        "total_error": int(fp + fn),
        "class_distribution": class_counts,
        "n_splits": int(n_splits),
        "best_iterations_median": int(final_iterations),
        "confusion_matrix_path": save_fig,
        "model_path": save_model_path,
        "threshold_path": save_threshold_path,
        "calibrator_path": save_calibrator_path,
        "params": {
            "beta_threshold": float(beta_threshold),
            "recall_floor": float(recall_floor),
            "class_weights": [float(neg_class_weight), float(pos_class_weight)],
            "bagging_temperature": float(bagging_temperature),
            "rsm": float(rsm),
        },
    }

    return final_model, results


if __name__ == "__main__":
    train_model()
