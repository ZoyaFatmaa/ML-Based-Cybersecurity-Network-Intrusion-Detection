import glob
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

BINARY_CLASS_NAMES = ["BENIGN", "ATTACK"]


def load_data(path: str) -> pd.DataFrame:
    if os.path.isdir(path):
        csv_files = sorted(glob.glob(os.path.join(path, "*.csv")))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in directory: {path}")
        frames = []
        for f in csv_files:
            print(f"Loading {f} ...")
            frames.append(pd.read_csv(f, low_memory=False))
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.read_csv(path, low_memory=False)

    print(f"Loaded dataset with shape: {df.shape}")
    return df


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()
    return df


def clean_data(df: pd.DataFrame, label_col: str = "Label") -> pd.DataFrame:
    df = clean_column_names(df)

    # drop duplicates before the split, or the same flow can land in both train and test
    before = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {before - len(df)} duplicate rows")

    df = df.replace([np.inf, -np.inf], np.nan)

    before = len(df)
    df = df.dropna()
    print(f"Dropped {before - len(df)} rows with NaN/inf values")

    nunique = df.nunique()
    constant_cols = [c for c in nunique[nunique <= 1].index if c != label_col]
    if constant_cols:
        print(f"Dropping constant columns: {constant_cols}")
        df = df.drop(columns=constant_cols)

    return df


def normalize_labels(labels: pd.Series) -> pd.Series:
    """Strip whitespace and fix the broken dash in the Web Attack labels."""
    return labels.astype(str).str.replace("\ufffd", "-", regex=False).str.strip()


def filter_rare_classes(df: pd.DataFrame, label_col: str = "Label", min_count: int = 10) -> pd.DataFrame:
    """Drop attack types with too few rows. A class needs at least 2 rows to survive a stratified split."""
    counts = df[label_col].value_counts()
    rare = counts[counts < min_count]
    if len(rare):
        print(f"Dropping classes with fewer than {min_count} rows: {rare.to_dict()}")
        df = df[~df[label_col].isin(rare.index)]
    return df


def encode_labels(df: pd.DataFrame, label_col: str = "Label", binary: bool = True):
    df[label_col] = normalize_labels(df[label_col])

    if binary:
        df[label_col] = df[label_col].apply(lambda x: "BENIGN" if x.upper() == "BENIGN" else "ATTACK")
        df[label_col] = df[label_col].map({"BENIGN": 0, "ATTACK": 1})
        return df, None

    le = LabelEncoder()
    df[label_col] = le.fit_transform(df[label_col])
    return df, le


def get_class_names(label_encoder) -> list:
    """Class names in label-index order. Binary mode has no encoder, so use BENIGN/ATTACK."""
    if label_encoder is None:
        return list(BINARY_CLASS_NAMES)
    return [str(c) for c in label_encoder.classes_]


def preprocess(
    path: str,
    label_col: str = "Label",
    binary: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
    min_class_count: int = 10,
):
    df = load_data(path)
    df = clean_data(df, label_col=label_col)
    df[label_col] = normalize_labels(df[label_col])

    if not binary:
        df = filter_rare_classes(df, label_col=label_col, min_count=min_class_count)

    df, label_encoder = encode_labels(df, label_col=label_col, binary=binary)

    X = df.drop(columns=[label_col])
    y = df[label_col]

    X = X.select_dtypes(include=[np.number])
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # fit the scaler on train only so test data does not leak in
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    class_names = get_class_names(label_encoder)
    print(f"Train shape: {X_train_scaled.shape}, Test shape: {X_test_scaled.shape}")
    print(f"Number of classes: {len(class_names)}")
    balance = y_train.value_counts(normalize=True).sort_index()
    balance.index = [class_names[i] for i in balance.index]
    print(f"Class balance (train):\n{balance}")

    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, scaler, label_encoder, feature_names


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess CICIDS2017 dataset")
    parser.add_argument("--path", required=True)
    parser.add_argument("--binary", action="store_true")
    parser.add_argument("--min_class_count", type=int, default=10)
    args = parser.parse_args()

    preprocess(args.path, binary=args.binary, min_class_count=args.min_class_count)
