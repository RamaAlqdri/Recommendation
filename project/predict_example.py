"""Simple client example for calling /predict endpoint."""

from __future__ import annotations

import json

import requests


API_URL = "http://127.0.0.1:5000/predict"

SAMPLE_PAYLOAD = {
    "number_of_clicks_on_similar_products": 12,
    "number_of_similar_products_purchased_so_far": 4,
    "average_rating_given_to_similar_products": 4.2,
    "gender": "male",
    "median_purchasing_price_in_rupees": 500,
    "rating_of_the_product": 4.5,
    "brand_of_the_product": "PUMA",
    "customer_review_sentiment_score_overall": 0.8,
    "price_of_the_product": 200,
    "holiday": "No",
    "season": "winter",
    "geographical_locations": "plains",
}


def main() -> None:
    """Send one prediction request and print response."""
    response = requests.post(API_URL, json=SAMPLE_PAYLOAD, timeout=30)
    print(f"Status code: {response.status_code}")
    try:
        print(json.dumps(response.json(), indent=2))
    except ValueError:
        print(response.text)


if __name__ == "__main__":
    main()
