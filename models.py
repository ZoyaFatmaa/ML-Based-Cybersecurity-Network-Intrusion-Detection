from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def get_models(random_state: int = 42, n_classes: int = 2) -> dict:
    multiclass = n_classes > 2

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            n_jobs=-1,
            random_state=random_state,
            class_weight="balanced",
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            # "logloss" is binary only, multiclass needs "mlogloss"
            eval_metric="mlogloss" if multiclass else "logloss",
            importance_type="gain",
            n_jobs=-1,
            random_state=random_state,
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        ),
    }
    return models
