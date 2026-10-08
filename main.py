import argparse
import os

from data_preprocessing import get_class_names, preprocess
from evaluate import evaluate_all
from train import save_preprocessing_artifacts, train_models


def main():
    parser = argparse.ArgumentParser(description="Run the full IDS comparative study pipeline")
    parser.add_argument("--path", required=True)
    parser.add_argument("--binary", action="store_true", help="attack vs benign only, instead of one class per attack type")
    parser.add_argument("--output_dir", default="artifacts")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--min_class_count", type=int, default=10,
                        help="multiclass only: skip attack types with fewer rows than this")
    parser.add_argument("--skip_explain", action="store_true", help="skip the SHAP and feature importance step")
    parser.add_argument("--shap_per_class", type=int, default=200, help="how many test rows per class to run SHAP on")
    parser.add_argument("--shap_beeswarm_class", default=None, help="also save a SHAP beeswarm plot for this class")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Step 1: Preprocessing")
    X_train, X_test, y_train, y_test, scaler, label_encoder, feature_names = preprocess(
        args.path,
        binary=args.binary,
        test_size=args.test_size,
        random_state=args.random_state,
        min_class_count=args.min_class_count,
    )
    class_names = get_class_names(label_encoder)
    save_preprocessing_artifacts(args.output_dir, scaler, label_encoder, feature_names)

    print("Step 2: Training")
    fitted_models = train_models(X_train, y_train, output_dir=args.output_dir, random_state=args.random_state)

    print("Step 3: Evaluation")
    results_df = evaluate_all(fitted_models, X_test, y_test, class_names, output_dir=args.output_dir)

    print("\nBest model by weighted F1:", results_df["F1"].idxmax())
    print("Best model by macro F1   :", results_df["Macro_F1"].idxmax())

    if not args.skip_explain:
        print("\nStep 4: Explainability (XGBoost)")
        try:
            from explain import explain_xgboost

            explain_xgboost(
                fitted_models["XGBoost"], X_test, y_test, feature_names, class_names,
                output_dir=args.output_dir, per_class=args.shap_per_class,
                random_state=args.random_state, beeswarm_class=args.shap_beeswarm_class,
            )
        except ImportError as e:
            print(f"Skipping explainability: {e}")


if __name__ == "__main__":
    main()
