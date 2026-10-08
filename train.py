import os
import time

import joblib
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight

from models import get_models


def save_preprocessing_artifacts(output_dir: str, scaler, label_encoder, feature_names):
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(output_dir, "scaler.joblib"))
    if label_encoder is not None:
        joblib.dump(label_encoder, os.path.join(output_dir, "label_encoder.joblib"))
    joblib.dump(feature_names, os.path.join(output_dir, "feature_names.joblib"))


def train_models(X_train, y_train, output_dir: str = "artifacts", random_state: int = 42) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    n_classes = len(np.unique(y_train))
    models = get_models(random_state=random_state, n_classes=n_classes)
    fitted = {}

    # XGBoost has no class_weight option, so pass the same "balanced" weighting as sample weights.
    # RandomForest and LogisticRegression already use it, so the comparison stays fair.
    xgb_sample_weight = compute_sample_weight("balanced", y_train)

    for name, model in models.items():
        print(f"\nTraining {name} ({n_classes} classes) ...")
        start = time.time()
        if name == "XGBoost":
            model.fit(X_train, y_train, sample_weight=xgb_sample_weight)
        else:
            model.fit(X_train, y_train)
        elapsed = time.time() - start
        print(f"{name} trained in {elapsed:.2f}s")

        model_path = os.path.join(output_dir, f"{name}.joblib")
        joblib.dump(model, model_path)
        print(f"Saved {name} to {model_path}")

        fitted[name] = model

    return fitted


if __name__ == "__main__":
    import argparse

    from data_preprocessing import preprocess

    parser = argparse.ArgumentParser(description="Train IDS models")
    parser.add_argument("--path", required=True)
    parser.add_argument("--binary", action="store_true")
    parser.add_argument("--output_dir", default="artifacts")
    parser.add_argument("--min_class_count", type=int, default=10)
    args = parser.parse_args()

    X_train, X_test, y_train, y_test, scaler, label_encoder, feature_names = preprocess(
        args.path, binary=args.binary, min_class_count=args.min_class_count
    )

    save_preprocessing_artifacts(args.output_dir, scaler, label_encoder, feature_names)
    train_models(X_train, y_train, output_dir=args.output_dir)
