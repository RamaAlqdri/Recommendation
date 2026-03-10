"""Central configuration for the recommendation scoring project."""

from pathlib import Path
import re


def to_snake_case(column_name: str) -> str:
    """Convert a raw column name into snake_case."""
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", column_name).strip("_").lower()
    return re.sub(r"_+", "_", normalized)


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "content_based_recommendation_dataset.csv"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "best_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

ORIGINAL_COLUMNS = [
    "Number of clicks on similar products",
    "Number of similar products purchased so far",
    "Average rating given to similar products",
    "Gender",
    "Median purchasing price (in rupees)",
    "Rating of the product",
    "Brand of the product",
    "Customer review sentiment score (overall)",
    "Price of the product",
    "Holiday",
    "Season",
    "Geographical locations",
    "Probability for the product to be recommended to the person",
]

TARGET_COLUMN_ORIGINAL = "Probability for the product to be recommended to the person"
COLUMN_NAME_MAPPING = {column: to_snake_case(column) for column in ORIGINAL_COLUMNS}
TARGET_COLUMN = COLUMN_NAME_MAPPING[TARGET_COLUMN_ORIGINAL]
FEATURE_COLUMNS = [
    COLUMN_NAME_MAPPING[column]
    for column in ORIGINAL_COLUMNS
    if column != TARGET_COLUMN_ORIGINAL
]

NUMERIC_FEATURE_COLUMNS_ORIGINAL = [
    "Number of clicks on similar products",
    "Number of similar products purchased so far",
    "Average rating given to similar products",
    "Median purchasing price (in rupees)",
    "Rating of the product",
    "Customer review sentiment score (overall)",
    "Price of the product",
]
CATEGORICAL_FEATURE_COLUMNS_ORIGINAL = [
    "Gender",
    "Brand of the product",
    "Holiday",
    "Season",
    "Geographical locations",
]
NUMERIC_FEATURE_COLUMNS = [
    COLUMN_NAME_MAPPING[column] for column in NUMERIC_FEATURE_COLUMNS_ORIGINAL
]
CATEGORICAL_FEATURE_COLUMNS = [
    COLUMN_NAME_MAPPING[column] for column in CATEGORICAL_FEATURE_COLUMNS_ORIGINAL
]

TEST_SIZE = 0.2
RANDOM_STATE = 42
RECOMMENDATION_THRESHOLD = 0.7
