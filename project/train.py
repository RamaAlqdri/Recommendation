"""Training entrypoint for recommendation scoring model."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import config
from utils import train_and_save_best_model


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for custom artifact paths."""
    parser = argparse.ArgumentParser(
        description="Train recommendation scoring model and save artifacts."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=config.DATA_PATH,
        help="Path ke dataset CSV.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=config.MODEL_PATH,
        help="Path output file model joblib.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=config.METADATA_PATH,
        help="Path output metadata JSON.",
    )
    return parser.parse_args()


def main() -> int:
    """Execute training flow and print JSON result."""
    args = parse_args()
    try:
        logging.info("Memulai training model...")
        result = train_and_save_best_model(
            data_path=args.data_path,
            model_path=args.model_path,
            metadata_path=args.metadata_path,
        )
        logging.info("Training selesai. Model terbaik: %s", result["selected_model"])
        print(json.dumps(result, indent=2))
        return 0
    except Exception as exc:
        logging.exception("Training gagal: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
