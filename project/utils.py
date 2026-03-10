"""Utility helpers for training, persistence, and API validation."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config


def load_and_validate_dataset(data_path: Path | str = config.DATA_PATH) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load dataset, validate required columns, and return cleaned dataframe with EDA summary."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {path}")

    raw_df = pd.read_csv(path)
    required_columns = set(config.ORIGINAL_COLUMNS)
    missing_columns = sorted(required_columns - set(raw_df.columns))
    if missing_columns:
        raise ValueError(
            f"Dataset tidak memiliki kolom wajib berikut: {', '.join(missing_columns)}"
        )

    cleaned_df = raw_df[config.ORIGINAL_COLUMNS].rename(columns=config.COLUMN_NAME_MAPPING)

    for numeric_column in config.NUMERIC_FEATURE_COLUMNS + [config.TARGET_COLUMN]:
        cleaned_df[numeric_column] = pd.to_numeric(cleaned_df[numeric_column], errors="coerce")

    if cleaned_df[config.TARGET_COLUMN].isna().any():
        raise ValueError("Target column memiliki nilai non-numeric atau missing.")

    eda_summary = {
        "rows": int(cleaned_df.shape[0]),
        "columns": int(cleaned_df.shape[1]),
        "missing_values_total": int(cleaned_df.isna().sum().sum()),
        "target_min": float(cleaned_df[config.TARGET_COLUMN].min()),
        "target_max": float(cleaned_df[config.TARGET_COLUMN].max()),
        "target_mean": float(cleaned_df[config.TARGET_COLUMN].mean()),
    }
    return cleaned_df, eda_summary


def build_preprocessor() -> ColumnTransformer:
    """Build preprocessing pipeline for numeric and categorical features."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, config.NUMERIC_FEATURE_COLUMNS),
            ("categorical", categorical_pipeline, config.CATEGORICAL_FEATURE_COLUMNS),
        ]
    )


def get_candidate_models(random_state: int = config.RANDOM_STATE) -> dict[str, Any]:
    """Return baseline model candidates and optional XGBoost if available."""
    models: dict[str, Any] = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=random_state),
    }

    try:
        from xgboost import XGBRegressor  # type: ignore

        models["XGBRegressor"] = XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=random_state,
        )
    except Exception:
        pass

    return models


def compute_regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE, and R2 metrics."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_and_save_best_model(
    data_path: Path | str = config.DATA_PATH,
    model_path: Path | str = config.MODEL_PATH,
    metadata_path: Path | str = config.METADATA_PATH,
) -> dict[str, Any]:
    """Train model candidates, select best by RMSE, and persist model + metadata."""
    df, eda_summary = load_and_validate_dataset(data_path=data_path)
    features = df[config.FEATURE_COLUMNS]
    target = df[config.TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
    )

    model_metrics: dict[str, dict[str, float]] = {}
    trained_pipelines: dict[str, Pipeline] = {}
    for model_name, model in get_candidate_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", model),
            ]
        )
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        metrics = compute_regression_metrics(y_test, predictions)
        model_metrics[model_name] = metrics
        trained_pipelines[model_name] = pipeline

    selected_model_name = min(model_metrics, key=lambda name: model_metrics[name]["rmse"])
    best_pipeline = trained_pipelines[selected_model_name]

    model_path_obj = Path(model_path)
    metadata_path_obj = Path(metadata_path)
    model_path_obj.parent.mkdir(parents=True, exist_ok=True)
    metadata_path_obj.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_pipeline, model_path_obj)

    metadata = {
        "selected_model": selected_model_name,
        "selection_metric": "rmse",
        "metrics": model_metrics,
        "feature_columns": config.FEATURE_COLUMNS,
        "numeric_feature_columns": config.NUMERIC_FEATURE_COLUMNS,
        "categorical_feature_columns": config.CATEGORICAL_FEATURE_COLUMNS,
        "target_column": config.TARGET_COLUMN,
        "column_mapping": config.COLUMN_NAME_MAPPING,
        "data_path": str(Path(data_path)),
        "model_path": str(model_path_obj),
        "test_size": config.TEST_SIZE,
        "random_state": config.RANDOM_STATE,
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "eda_summary": eda_summary,
    }

    with metadata_path_obj.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    return metadata


def load_model_and_metadata(
    model_path: Path | str = config.MODEL_PATH,
    metadata_path: Path | str = config.METADATA_PATH,
) -> tuple[Any | None, dict[str, Any] | None]:
    """Load persisted model and metadata if available."""
    model_file = Path(model_path)
    metadata_file = Path(metadata_path)
    if not model_file.exists() or not metadata_file.exists():
        return None, None

    loaded_model = joblib.load(model_file)
    with metadata_file.open("r", encoding="utf-8") as file:
        loaded_metadata = json.load(file)
    return loaded_model, loaded_metadata


def validate_single_record(
    record: dict[str, Any],
    feature_columns: list[str] | None = None,
    numeric_feature_columns: list[str] | None = None,
    categorical_feature_columns: list[str] | None = None,
) -> tuple[bool, list[str], dict[str, Any]]:
    """Validate one prediction payload record and return a normalized record."""
    if not isinstance(record, dict):
        return False, ["Payload harus berupa object JSON."], {}

    expected_features = feature_columns or config.FEATURE_COLUMNS
    numeric_features = set(numeric_feature_columns or config.NUMERIC_FEATURE_COLUMNS)
    categorical_features = set(
        categorical_feature_columns or config.CATEGORICAL_FEATURE_COLUMNS
    )

    missing_fields = [column for column in expected_features if column not in record]
    errors: list[str] = []
    if missing_fields:
        errors.append(f"Field wajib hilang: {', '.join(missing_fields)}")

    unexpected_fields = [column for column in record if column not in expected_features]
    if unexpected_fields:
        errors.append(f"Field tidak dikenali: {', '.join(unexpected_fields)}")

    normalized_record: dict[str, Any] = {}
    for feature in expected_features:
        if feature not in record:
            continue
        value = record[feature]
        if feature in numeric_features:
            if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
                errors.append(f"Field '{feature}' harus bertipe numerik (int/float).")
                continue
            normalized_record[feature] = float(value)
        elif feature in categorical_features:
            if value is None or not isinstance(value, str) or not value.strip():
                errors.append(f"Field '{feature}' harus string non-kosong.")
                continue
            normalized_record[feature] = value.strip()
        else:
            normalized_record[feature] = value

    if errors:
        return False, errors, {}
    return True, [], normalized_record


def validate_batch_records(
    records: list[dict[str, Any]],
    feature_columns: list[str] | None = None,
    numeric_feature_columns: list[str] | None = None,
    categorical_feature_columns: list[str] | None = None,
) -> tuple[bool, list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate a list of records and return normalized records."""
    if not isinstance(records, list) or not records:
        return False, [{"error": "Payload batch harus berupa list non-kosong."}], []

    all_errors: list[dict[str, Any]] = []
    normalized_records: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        is_valid, errors, normalized = validate_single_record(
            record=record,
            feature_columns=feature_columns,
            numeric_feature_columns=numeric_feature_columns,
            categorical_feature_columns=categorical_feature_columns,
        )
        if not is_valid:
            all_errors.append({"index": idx, "errors": errors})
            continue
        normalized_records.append(normalized)

    if all_errors:
        return False, all_errors, []
    return True, [], normalized_records


def records_to_dataframe(records: list[dict[str, Any]], feature_columns: list[str]) -> pd.DataFrame:
    """Convert normalized records to DataFrame in the expected feature order."""
    return pd.DataFrame(records)[feature_columns]


def clip_probability(value: float) -> float:
    """Clip prediction into [0, 1] since target probability range is bounded."""
    return float(np.clip(value, 0.0, 1.0))
