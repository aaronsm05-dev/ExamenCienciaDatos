from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    DATA_PROCESSED_DIR,
    DATA_RAW_PATH,
    DROP_DURATION,
    RANDOM_STATE,
    TEST_SIZE,
)


def add_pdays_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """Crea indicador de contacto previo y marca -1 en pdays como faltante"""
    data = df.copy()
    if "pdays" in data.columns:
        data["was_contacted_prev"] = (data["pdays"] != -1).astype(int)
        data["pdays"] = data["pdays"].replace(-1, np.nan)
    return data


def load_and_split_data(
    csv_path: Path = DATA_RAW_PATH,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    drop_duration: bool = DROP_DURATION,
):
    """Carga Bank Marketing y divide en train y test de forma estratificada"""
    df = pd.read_csv(csv_path, sep=";")
    df = add_pdays_indicator(df)

    if drop_duration and "duration" in df.columns:
        df = df.drop(columns=["duration"])

    y = df["y"].map({"no": 0, "yes": 1}).astype(int)
    X = df.drop(columns=["y"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    return X_train, X_test, y_train, y_test


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Construye el preprocesador (se ajusta únicamente con X_train)"""
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ],
        remainder="drop",
    )


def prepare_datasets(
    csv_path: Path = DATA_RAW_PATH,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    drop_duration: bool = DROP_DURATION,
    save_processed: bool = False,
    processed_dir: Path = DATA_PROCESSED_DIR,
):
    """Devuelve X_train/X_test procesados y opcionalmente guarda los artefactos"""
    X_train_raw, X_test_raw, y_train, y_test = load_and_split_data(
        csv_path=csv_path,
        test_size=test_size,
        random_state=random_state,
        drop_duration=drop_duration,
    )

    preprocessor = build_preprocessor(X_train_raw)
    X_train_proc = preprocessor.fit_transform(X_train_raw)
    X_test_proc = preprocessor.transform(X_test_raw)

    y_train_arr = y_train.to_numpy()
    y_test_arr = y_test.to_numpy()
    feature_names = preprocessor.get_feature_names_out()

    if save_processed:
        processed_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            processed_dir / "bank_marketing_prepared.npz",
            X_train=X_train_proc,
            X_test=X_test_proc,
            y_train=y_train_arr,
            y_test=y_test_arr,
            feature_names=feature_names,
        )
        joblib.dump(preprocessor, processed_dir / "preprocessor.joblib")

    return (
        X_train_proc,
        X_test_proc,
        y_train_arr,
        y_test_arr,
        feature_names,
        preprocessor,
    )


if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, cols, _ = prepare_datasets(save_processed=True)
    print(f"Train: {X_tr.shape}, Test: {X_te.shape}, Features: {len(cols)}")