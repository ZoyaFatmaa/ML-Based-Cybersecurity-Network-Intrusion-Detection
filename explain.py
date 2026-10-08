"""SHAP and gain importance for the XGBoost model.

Writes these files to output_dir:
  xgb_gain_importance.csv / .png     built-in gain importance
  shap_global_importance.csv / .png  mean |SHAP| per feature, all classes combined
  shap_per_class_importance.csv      the same, split by class (multiclass only)
  shap_class_heatmap.png             top features vs classes (multiclass only)
  shap_beeswarm_<class>.png          only if you ask for one
"""
import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def sample_per_class(X, y, per_class: int = 200, random_state: int = 42):
    """Take up to `per_class` rows from each class.

    A plain random sample is about 80% BENIGN and hardly has any Heartbleed,
    Infiltration or SQL injection rows, so the rare attacks would not show up in the plots.
    """
    rng = np.random.default_rng(random_state)
    picked = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        picked.append(rng.choice(idx, size=min(per_class, len(idx)), replace=False))
    picked = np.sort(np.concatenate(picked))
    return X[picked], y[picked]


def to_class_array(shap_values) -> np.ndarray:
    """Put SHAP output into one shape: (samples, features, outputs).

    Older shap versions return a list of (samples, features) arrays for multiclass,
    newer ones return one 3D array, and binary gives a plain 2D array. This handles
    all three so the rest of the code doesn't need to.
    """
    if isinstance(shap_values, list):
        return np.stack([np.asarray(v) for v in shap_values], axis=-1)
    arr = np.asarray(shap_values)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3:
        raise ValueError(f"Unexpected SHAP output shape: {arr.shape}")
    return arr


def _barh(series: pd.Series, title: str, xlabel: str, path: str, top_n: int):
    top = series.sort_values(ascending=False).head(top_n)[::-1]
    plt.figure(figsize=(9, 0.4 * len(top) + 1.5))
    plt.barh(top.index, top.values, color="#2b6cb0")
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close("all")


def save_gain_importance(model, feature_names: list, output_dir: str, top_n: int = 15) -> pd.Series:
    imp = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    imp.rename("Gain_importance").to_csv(os.path.join(output_dir, "xgb_gain_importance.csv"))
    _barh(imp, f"XGBoost Top {top_n} Features (gain)", "Gain importance",
          os.path.join(output_dir, "xgb_gain_importance.png"), top_n)
    return imp


def save_shap_outputs(arr: np.ndarray, feature_names: list, class_names: list, output_dir: str, top_n: int = 15):
    n_features, n_outputs = arr.shape[1], arr.shape[2]
    if n_features != len(feature_names):
        raise ValueError(f"SHAP has {n_features} features but {len(feature_names)} names were given")
    if n_outputs > 1 and n_outputs != len(class_names):
        raise ValueError(f"SHAP has {n_outputs} outputs but {len(class_names)} class names were given")

    mean_abs = np.abs(arr).mean(axis=0)  # shape: features x outputs
    global_imp = pd.Series(mean_abs.sum(axis=1), index=feature_names).sort_values(ascending=False)
    global_imp.rename("Mean_abs_SHAP").to_csv(os.path.join(output_dir, "shap_global_importance.csv"))
    _barh(global_imp, f"Top {top_n} Features by mean |SHAP|", "Mean |SHAP value| (summed over classes)",
          os.path.join(output_dir, "shap_global_importance.png"), top_n)

    if n_outputs == 1:
        return global_imp, None

    per_class = pd.DataFrame(mean_abs, index=feature_names, columns=class_names)
    per_class.loc[global_imp.index].to_csv(os.path.join(output_dir, "shap_per_class_importance.csv"))

    # Heatmap: for each class, the share of its total SHAP that each top feature accounts for
    share = per_class / per_class.sum(axis=0).replace(0, np.nan)
    top_feats = global_imp.head(top_n).index
    heat = share.loc[top_feats].fillna(0)
    fig, ax = plt.subplots(figsize=(max(8, 0.75 * len(class_names)), 0.45 * len(top_feats) + 3))
    im = ax.imshow(heat.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=60, ha="right")
    ax.set_yticks(range(len(top_feats)))
    ax.set_yticklabels(top_feats)
    fig.colorbar(im, ax=ax, label="Share of class's total |SHAP|")
    ax.set_title("Which features drive each attack class (XGBoost)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_class_heatmap.png"), dpi=150)
    plt.close("all")
    return global_imp, per_class


def explain_xgboost(
    model,
    X_test,
    y_test,
    feature_names: list,
    class_names: list,
    output_dir: str = "artifacts",
    per_class: int = 200,
    top_n: int = 15,
    random_state: int = 42,
    beeswarm_class: str = None,
):
    try:
        import shap
    except ImportError as e:
        raise ImportError("SHAP is not installed. Run: pip install shap") from e

    os.makedirs(output_dir, exist_ok=True)

    save_gain_importance(model, feature_names, output_dir, top_n)
    print(f"Saved XGBoost gain importance to {output_dir}/xgb_gain_importance.png")

    X_s, y_s = sample_per_class(np.asarray(X_test), np.asarray(y_test), per_class, random_state)
    X_df = pd.DataFrame(X_s, columns=feature_names)
    print(f"Computing SHAP values on {len(X_df)} test rows (up to {per_class} per class) ...")

    explainer = shap.TreeExplainer(model)
    arr = to_class_array(explainer.shap_values(X_df))

    global_imp, _ = save_shap_outputs(arr, feature_names, class_names, output_dir, top_n)
    print(f"Saved SHAP outputs to {output_dir}/")
    print("Top 10 features by mean |SHAP|:")
    print(global_imp.head(10).round(4))

    if beeswarm_class:
        k = 0 if arr.shape[2] == 1 else class_names.index(beeswarm_class)
        shap.summary_plot(arr[:, :, k], X_df, max_display=top_n, show=False)
        plt.title(f"SHAP summary - {beeswarm_class}")
        plt.tight_layout()
        safe = "".join(ch if ch.isalnum() else "_" for ch in beeswarm_class)
        plt.savefig(os.path.join(output_dir, f"shap_beeswarm_{safe}.png"), dpi=150)
        plt.close("all")

    return global_imp


if __name__ == "__main__":
    import argparse

    from data_preprocessing import get_class_names, preprocess

    parser = argparse.ArgumentParser(description="SHAP + gain explanations for the trained XGBoost model")
    parser.add_argument("--path", required=True)
    parser.add_argument("--binary", action="store_true")
    parser.add_argument("--output_dir", default="artifacts")
    parser.add_argument("--min_class_count", type=int, default=10)
    parser.add_argument("--per_class", type=int, default=200)
    parser.add_argument("--beeswarm_class", default=None, help="class to plot, e.g. 'DDoS' (or 'ATTACK' in binary mode)")
    args = parser.parse_args()

    _, X_test, _, y_test, _, label_encoder, feature_names = preprocess(
        args.path, binary=args.binary, min_class_count=args.min_class_count
    )
    model = joblib.load(os.path.join(args.output_dir, "XGBoost.joblib"))
    explain_xgboost(
        model, X_test, y_test, feature_names, get_class_names(label_encoder),
        output_dir=args.output_dir, per_class=args.per_class, beeswarm_class=args.beeswarm_class,
    )
