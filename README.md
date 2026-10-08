# IDS Comparative Study (CICIDS2017)

Trains and compares three classifiers (Random Forest, XGBoost, Logistic Regression) for network intrusion detection on the CICIDS2017 dataset, in **binary** (attack vs benign) or **multiclass** (per attack type) mode, and uses SHAP to explain the XGBoost model.

## Files

- `data_preprocessing.py` - loads, cleans, encodes and scales the dataset
- `models.py` - model definitions (RF, XGBoost, Logistic Regression)
- `train.py` - fits models and saves them to disk
- `evaluate.py` - metrics, per-class reports, confusion matrices, comparison chart
- `explain.py` - SHAP and gain-based feature importance for XGBoost
- `main.py` - runs the full pipeline end to end
- `requirements.txt`

## Setup

```
pip install -r requirements.txt
```

## Data

Download CICIDS2017 (the MachineLearningCSV version). Point `--path` at a single CSV or a directory of CSVs; they are concatenated automatically.

## Running

Multiclass (every attack type is its own class):

```
python main.py --path /path/to/cicids2017 --output_dir artifacts
```

Binary (all attacks collapsed into ATTACK):

```
python main.py --path /path/to/cicids2017 --binary --output_dir artifacts
```

Useful flags: `--min_class_count 10` (multiclass: drop attack types with fewer rows), `--skip_explain`, `--shap_per_class 200`, `--shap_beeswarm_class DDoS`.

Individual steps (each re-runs preprocessing with the same seed, so splits match):

```
python train.py    --path /path/to/cicids2017 --output_dir artifacts
python evaluate.py --path /path/to/cicids2017 --output_dir artifacts
python explain.py  --path /path/to/cicids2017 --output_dir artifacts
```

## Outputs (`artifacts/`)

- `*.joblib` - trained models, scaler, label encoder, feature names
- `model_comparison.csv` / `.png` - Accuracy, Precision, Recall, F1, Macro-F1, ROC-AUC
- `classification_report_<model>.csv` - per-class precision / recall / F1 / support
- `per_class_recall.csv`, `per_class_f1.csv` - all models side by side (which model catches rare attacks?)
- `confusion_matrix_<model>.png` / `.csv` - row-normalised for multiclass
- `xgb_gain_importance.*`, `shap_global_importance.*`, `shap_per_class_importance.csv`, `shap_class_heatmap.png`

## Results

> **Results pending.** The tables below are placeholders. Run the pipeline on CICIDS2017 and replace each `TBD` with the numbers from `artifacts/model_comparison.csv`.

Multiclass run (all attack types, single 80/20 stratified split, seed 42):

| Model | Accuracy | Weighted F1 | Macro F1 | ROC-AUC |
|---|---|---|---|---|
| Random Forest | TBD | TBD | TBD | TBD |
| XGBoost | TBD | TBD | TBD | TBD |
| Logistic Regression | TBD | TBD | TBD | TBD |

Per-class recall for the rarest attacks (from `artifacts/per_class_recall.csv`):

| Attack type | Test support | Random Forest | XGBoost | Logistic Regression |
|---|---|---|---|---|
| Heartbleed | TBD | TBD | TBD | TBD |
| Infiltration | TBD | TBD | TBD | TBD |
| Web Attack - SQL Injection | TBD | TBD | TBD | TBD |

### Plots

_Plots pending. Once the pipeline has run, delete this line and uncomment the images below._

**Model comparison**

<!-- ![Model comparison](artifacts/model_comparison.png) -->

**Confusion matrix (XGBoost)**

<!-- ![Confusion matrix](artifacts/confusion_matrix_XGBoost.png) -->

**SHAP: which features drive each attack class**

<!-- ![SHAP heatmap](artifacts/shap_class_heatmap.png) -->

## Metrics

- **Multiclass** Precision / Recall / F1 are *support-weighted*; ROC-AUC is one-vs-rest weighted. BENIGN is about 80% of CICIDS2017, so weighted scores are mostly driven by it. **Macro-F1** (every class counts equally) and the per-class recall table are better for judging rare attacks.
- **Binary** metrics use ATTACK as the positive class.

## Explainability

`explain.py` uses SHAP's `TreeExplainer` on the XGBoost model. SHAP is computed on a class-balanced test sample (up to 200 rows per class) so rare attacks show up; a plain random sample would be about 80% benign. Global importance is mean |SHAP| summed over classes; the heatmap shows which features drive each attack class.

## Notes

- `class_weight="balanced"` is used for RF and Logistic Regression; XGBoost gets the equivalent via balanced `sample_weight`, so the comparison is fair. Very large weights on tiny classes (e.g. Heartbleed) can make XGBoost unstable. If that happens, soften them, for example with square-root weights.
- Classes with fewer than `--min_class_count` rows are dropped in multiclass mode (a class needs at least 2 rows just to be split). Very rare classes still end up with tiny test sets, so be careful reading their scores (see the `Support` column).
- All models train on scaled features. Tree models don't need it, but it keeps the pipeline simple.
- Hyperparameters are just starting points and are not tuned. Use GridSearchCV or RandomizedSearchCV for a more careful comparison.
- CICIDS2017 is known to contain near-duplicate flows and some label errors, so a single random split can give scores that look better than they really are.
