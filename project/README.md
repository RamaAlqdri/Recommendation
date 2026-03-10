# E-Commerce Recommendation Scoring API (Supervised Learning)

Project ini membangun sistem rekomendasi produk e-commerce berbasis **supervised learning regresi** untuk memprediksi probabilitas `0-1`:

`Probability for the product to be recommended to the person`

## Problem Framing

Kasus ini adalah **recommendation scoring problem**, bukan collaborative filtering klasik, karena:

- Label target sudah tersedia sebagai probabilitas rekomendasi.
- Prediksi dilakukan dari kombinasi fitur user behavior, produk, dan konteks (musim, lokasi, holiday).
- Output model adalah skor probabilitas untuk ranking/keputusan rekomendasi.

## Struktur Project

```text
project/
├── app.py
├── train.py
├── predict_example.py
├── requirements.txt
├── README.md
├── config.py
├── utils.py
├── models/
│   ├── best_model.joblib
│   └── model_metadata.json
└── data/
    └── content_based_recommendation_dataset.csv
```

## Mapping Nama Kolom (Original -> Snake Case)

Mapping ini dipakai otomatis saat training dan disimpan ke metadata.

1. `Number of clicks on similar products` -> `number_of_clicks_on_similar_products`
2. `Number of similar products purchased so far` -> `number_of_similar_products_purchased_so_far`
3. `Average rating given to similar products` -> `average_rating_given_to_similar_products`
4. `Gender` -> `gender`
5. `Median purchasing price (in rupees)` -> `median_purchasing_price_in_rupees`
6. `Rating of the product` -> `rating_of_the_product`
7. `Brand of the product` -> `brand_of_the_product`
8. `Customer review sentiment score (overall)` -> `customer_review_sentiment_score_overall`
9. `Price of the product` -> `price_of_the_product`
10. `Holiday` -> `holiday`
11. `Season` -> `season`
12. `Geographical locations` -> `geographical_locations`
13. `Probability for the product to be recommended to the person` -> `probability_for_the_product_to_be_recommended_to_the_person`

## Fitur Utama

- Validasi dataset dan exploratory checks singkat (shape, missing value, statistik target).
- Preprocessing numerik dan kategorikal dengan `ColumnTransformer`.
- Baseline model:
  - `LinearRegression`
  - `RandomForestRegressor`
  - `GradientBoostingRegressor`
  - `XGBRegressor` otomatis ditambahkan jika `xgboost` terpasang.
- Evaluasi model dengan:
  - MAE
  - RMSE
  - R2
- Pemilihan model terbaik berdasarkan RMSE terendah.
- Penyimpanan artifact:
  - `models/best_model.joblib`
  - `models/model_metadata.json`
- Flask API:
  - `GET /`
  - `GET /health`
  - `POST /train`
  - `POST /predict`
  - `POST /predict_batch`

## Instalasi

```bash
cd project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Konfigurasi Git

File Git config yang sudah disiapkan:

- `.gitignore` di root repo (`Recommendation/.gitignore`)
- `.gitattributes` di root repo (`Recommendation/.gitattributes`)

Langkah setup Git (jalankan dari root repo `Recommendation/`):

```bash
# Isi identitas (sekali per machine atau gunakan --global)
git config user.name "Nama Anda"
git config user.email "email@anda.com"

# Jika belum init repo
git init

# Gunakan branch utama
git branch -M main

# Commit awal
git add .
git commit -m "Initialize recommendation scoring project"
```

## Training Model

```bash
python train.py
```

Optional custom path:

```bash
python train.py --data-path data/content_based_recommendation_dataset.csv --model-path models/best_model.joblib --metadata-path models/model_metadata.json
```

## Menjalankan Flask Server

```bash
python app.py
```

Server default: `http://127.0.0.1:5000`

## Contoh Request `POST /predict`

```bash
curl -X POST "http://127.0.0.1:5000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
    "geographical_locations": "plains"
  }'
```

Contoh response:

```json
{
  "predicted_probability": 0.812345,
  "recommendation_label": "recommended"
}
```

Rule label:

- `predicted_probability >= 0.7` -> `recommended`
- `< 0.7` -> `not_recommended`

## Contoh Request `POST /predict_batch`

```bash
curl -X POST "http://127.0.0.1:5000/predict_batch" \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
      {
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
        "geographical_locations": "plains"
      },
      {
        "number_of_clicks_on_similar_products": 1,
        "number_of_similar_products_purchased_so_far": 0,
        "average_rating_given_to_similar_products": 2.9,
        "gender": "female",
        "median_purchasing_price_in_rupees": 800,
        "rating_of_the_product": 3.1,
        "brand_of_the_product": "AmazonBasics",
        "customer_review_sentiment_score_overall": 0.2,
        "price_of_the_product": 950,
        "holiday": "Yes",
        "season": "summer",
        "geographical_locations": "coastal"
      }
    ]
  }'
```

## Catatan Produksi

- Endpoint prediksi aman untuk kategori baru karena encoder menggunakan `handle_unknown="ignore"`.
- Jika model belum dilatih, endpoint `/predict` dan `/predict_batch` mengembalikan HTTP `503` dengan pesan jelas.
- Metadata training (`model_metadata.json`) berisi model terpilih, semua metrik kandidat, fitur, target, mapping kolom, dan timestamp training.
