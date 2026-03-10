"""Flask API server for recommendation scoring model."""

from __future__ import annotations

from typing import Any

from flask import Flask, jsonify, request
from flask_cors import CORS

import config
from utils import (
    clip_probability,
    load_model_and_metadata,
    records_to_dataframe,
    train_and_save_best_model,
    validate_batch_records,
    validate_single_record,
)


app = Flask(__name__)
CORS(app)

MODEL_STATE: dict[str, Any] = {"model": None, "metadata": None}


def refresh_loaded_artifacts() -> bool:
    """Reload model and metadata from disk into memory."""
    model, metadata = load_model_and_metadata()
    MODEL_STATE["model"] = model
    MODEL_STATE["metadata"] = metadata
    return model is not None and metadata is not None


def get_feature_schema() -> tuple[list[str], list[str], list[str]]:
    """Get feature schema from metadata if available, fallback to config constants."""
    metadata = MODEL_STATE.get("metadata") or {}
    feature_columns = metadata.get("feature_columns", config.FEATURE_COLUMNS)
    numeric_columns = metadata.get(
        "numeric_feature_columns", config.NUMERIC_FEATURE_COLUMNS
    )
    categorical_columns = metadata.get(
        "categorical_feature_columns", config.CATEGORICAL_FEATURE_COLUMNS
    )
    return feature_columns, numeric_columns, categorical_columns


def make_prediction_label(probability: float) -> str:
    """Return recommendation label based on threshold."""
    if probability >= config.RECOMMENDATION_THRESHOLD:
        return "recommended"
    return "not_recommended"


refresh_loaded_artifacts()


@app.get("/")
def root() -> Any:
    """Root endpoint."""
    return jsonify(
        {
            "message": "Recommendation scoring API is active.",
            "endpoints": ["/health", "/train", "/predict", "/predict_batch"],
        }
    )


@app.get("/health")
def health() -> Any:
    """Health endpoint with model status."""
    metadata = MODEL_STATE.get("metadata")
    return jsonify(
        {
            "status": "healthy",
            "model_loaded": MODEL_STATE.get("model") is not None,
            "model_path": str(config.MODEL_PATH),
            "metadata_path": str(config.METADATA_PATH),
            "training_timestamp": metadata.get("training_timestamp")
            if metadata
            else None,
        }
    )


@app.post("/train")
def train_endpoint() -> Any:
    """Train model from dataset and persist artifacts."""
    try:
        result = train_and_save_best_model()
        refresh_loaded_artifacts()
        return jsonify(
            {
                "message": "Training completed successfully.",
                "result": result,
            }
        )
    except Exception as exc:
        return (
            jsonify(
                {
                    "error": "Training failed.",
                    "details": str(exc),
                }
            ),
            500,
        )


@app.post("/predict")
def predict() -> Any:
    """Predict recommendation probability for one record."""
    if MODEL_STATE.get("model") is None:
        return (
            jsonify(
                {
                    "error": "Model belum tersedia. Jalankan endpoint /train terlebih dahulu."
                }
            ),
            503,
        )

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return (
            jsonify(
                {"error": "Payload tidak valid. Gunakan object JSON untuk satu record."}
            ),
            400,
        )

    feature_columns, numeric_columns, categorical_columns = get_feature_schema()
    is_valid, errors, normalized = validate_single_record(
        record=payload,
        feature_columns=feature_columns,
        numeric_feature_columns=numeric_columns,
        categorical_feature_columns=categorical_columns,
    )
    if not is_valid:
        return jsonify({"error": "Validasi input gagal.", "details": errors}), 400

    input_frame = records_to_dataframe([normalized], feature_columns=feature_columns)
    raw_prediction = float(MODEL_STATE["model"].predict(input_frame)[0])
    probability = clip_probability(raw_prediction)

    return jsonify(
        {
            "predicted_probability": round(probability, 6),
            "recommendation_label": make_prediction_label(probability),
        }
    )


@app.post("/predict_batch")
def predict_batch() -> Any:
    """Predict recommendation probability for multiple records."""
    if MODEL_STATE.get("model") is None:
        return (
            jsonify(
                {
                    "error": "Model belum tersedia. Jalankan endpoint /train terlebih dahulu."
                }
            ),
            503,
        )

    payload = request.get_json(silent=True)
    records = None
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict) and isinstance(payload.get("records"), list):
        records = payload["records"]

    if records is None:
        return (
            jsonify(
                {
                    "error": "Payload tidak valid. Gunakan list record atau {'records': [...]}."
                }
            ),
            400,
        )

    feature_columns, numeric_columns, categorical_columns = get_feature_schema()
    is_valid, errors, normalized_records = validate_batch_records(
        records=records,
        feature_columns=feature_columns,
        numeric_feature_columns=numeric_columns,
        categorical_feature_columns=categorical_columns,
    )
    if not is_valid:
        return jsonify({"error": "Validasi batch gagal.", "details": errors}), 400

    input_frame = records_to_dataframe(normalized_records, feature_columns=feature_columns)
    raw_predictions = MODEL_STATE["model"].predict(input_frame)
    probabilities = [clip_probability(float(prediction)) for prediction in raw_predictions]

    predictions = [
        {
            "index": index,
            "predicted_probability": round(probability, 6),
            "recommendation_label": make_prediction_label(probability),
        }
        for index, probability in enumerate(probabilities)
    ]

    return jsonify({"count": len(predictions), "predictions": predictions})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
