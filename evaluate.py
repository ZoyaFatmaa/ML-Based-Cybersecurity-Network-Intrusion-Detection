import os

import joblib
import matplotlib

matplotlib.use("Agg")  # no display needed, works on servers too
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _roc_auc(model, X_test, y_test, n_classes: int) -> float:
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)
            if n_classes == 2:
                return roc_auc_score(y_test, proba[:, 1])
            proba = proba / proba.sum(axis=1, keepdims=True)  # float32 rounding can make rows sum to 0.9999..., and sklearn rejects that
            return roc_auc_score(
                y_test, proba, multi_class="ovr", average="weighted", labels=list(range(n_classes))
            )
        if hasattr(model, "decision_function") and n_classes == 2:
            return roc_auc_score(y_test, model.decision_function(X_test))
    except ValueError as e:
        # a class is missing from y_test, so one-vs-rest AUC can't be computed
        print(f"  ROC-AUC skipped: {e}")
    return float("nan")


def evaluate_model(name: str, model, X_test, y_test, class_names: list):
    """Score one model, binary or multiclass. Multiclass precision/recall/F1 are weighted by class size."""
    n_classes = len(class_names)
    labels = list(range(n_classes))
    y_pred = model.predict(X_test)

    if n_classes == 2:
        avg, extra = "binary", {}
    else:
        avg, extra = "weighted", {"labels": labels}

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average=avg, zero_division=0, **extra),
        "Recall": recall_score(y_test, y_pred, average=avg, zero_division=0, **extra),
        "F1": f1_score(y_test, y_pred, average=avg, zero_division=0, **extra),
        # macro F1 treats every class equally, so poor rare-attack detection can't hide behind BENIGN
        "Macro_F1": f1_score(y_test, y_pred, labels=labels, average="macro", zero_division=0),
        "ROC_AUC": _roc_auc(model, X_test, y_test, n_classes),
    }

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    report = classification_report(
        y_test, y_pred, labels=labels, target_names=class_names, output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report).T
    return metrics, cm, report_df


def plot_confusion_matrix(cm, class_names: list, title: str, path: str):
    n = len(class_names)
    if n == 2:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap="Blues", values_format="d")
    else:
        # Show the share of each true class instead of raw counts. BENIGN outnumbers
        # some attacks by about 100,000 to 1, so a count matrix is just one dark cell.
        row_sums = cm.sum(axis=1, keepdims=True)
        norm = np.divide(cm, row_sums, out=np.zeros(cm.shape, dtype=float), where=row_sums > 0)
        fig, ax = plt.subplots(figsize=(max(8, 0.8 * n), max(6.5, 0.7 * n)))
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(n))
        ax.set_xticklabels(class_names, rotation=60, ha="right")
        ax.set_yticks(range(n))
        ax.set_yticklabels(class_names)
        for i in range(n):
            for j in range(n):
                if norm[i, j] >= 0.005:
                    ax.text(
                        j, i, f"{norm[i, j] * 100:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.5 else "black",
                    )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        fig.colorbar(im, ax=ax, label="Fraction of true class")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close("all")


def evaluate_all(models: dict, X_test, y_test, class_names: list, output_dir: str = "artifacts") -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    f1_by_model, recall_by_model = {}, {}
    support = None

    for name, model in models.items():
        metrics, cm, report_df = evaluate_model(name, model, X_test, y_test, class_names)
        rows.append(metrics)

        cm_path = os.path.join(output_dir, f"confusion_matrix_{name}.png")
        plot_confusion_matrix(cm, class_names, f"Confusion Matrix - {name}", cm_path)
        pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
            os.path.join(output_dir, f"confusion_matrix_{name}.csv")
        )
        print(f"Saved confusion matrix for {name} to {cm_path}")

        report_path = os.path.join(output_dir, f"classification_report_{name}.csv")
        report_df.to_csv(report_path)
        print(f"Saved per-class report for {name} to {report_path}")

        f1_by_model[name] = report_df.loc[class_names, "f1-score"]
        recall_by_model[name] = report_df.loc[class_names, "recall"]
        support = report_df.loc[class_names, "support"]

    results_df = pd.DataFrame(rows).set_index("Model")
    print("\nModel comparison:")
    print(results_df.round(4))

    results_csv = os.path.join(output_dir, "model_comparison.csv")
    results_df.to_csv(results_csv)
    print(f"Saved comparison table to {results_csv}")

    # per-class tables with all models side by side, to see which one catches the rare attacks
    for label, table in (("f1", f1_by_model), ("recall", recall_by_model)):
        df = pd.DataFrame(table)
        df.insert(0, "Support", support.astype(int))
        df.to_csv(os.path.join(output_dir, f"per_class_{label}.csv"))
        if label == "recall" and len(class_names) > 2:
            print("\nPer-class recall (share of each attack type actually caught):")
            print(df.round(3))

    plot_comparison(results_df, output_dir)

    return results_df


def plot_comparison(results_df: pd.DataFrame, output_dir: str):
    cols = ["Accuracy", "Precision", "Recall", "F1", "Macro_F1", "ROC_AUC"]
    plot_df = results_df[[c for c in cols if c in results_df.columns]]

    ax = plot_df.plot(kind="bar", figsize=(11, 6))
    ax.set_title("IDS Model Comparison (CICIDS2017)")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    plt.xticks(rotation=0)
    plt.tight_layout()

    chart_path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(chart_path)
    plt.close("all")
    print(f"Saved comparison chart to {chart_path}")


if __name__ == "__main__":
    import argparse

    from data_preprocessing import get_class_names, preprocess

    parser = argparse.ArgumentParser(description="Evaluate trained IDS models")
    parser.add_argument("--path", required=True)
    parser.add_argument("--binary", action="store_true")
    parser.add_argument("--output_dir", default="artifacts")
    parser.add_argument("--min_class_count", type=int, default=10)
    args = parser.parse_args()

    _, X_test, _, y_test, _, label_encoder, _ = preprocess(
        args.path, binary=args.binary, min_class_count=args.min_class_count
    )
    class_names = get_class_names(label_encoder)

    model_names = ["RandomForest", "XGBoost", "LogisticRegression"]
    models = {name: joblib.load(os.path.join(args.output_dir, f"{name}.joblib")) for name in model_names}

    evaluate_all(models, X_test, y_test, class_names, output_dir=args.output_dir)
