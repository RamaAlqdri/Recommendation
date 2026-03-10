# E-Commerce Recommendation Scoring API

Dokumentasi ini ditulis untuk pemula, dari nol sampai project bisa jalan end-to-end.

Project ini memprediksi:
- `Probability for the product to be recommended to the person` (nilai 0 sampai 1)

## 1. Apa Masalah Yang Diselesaikan

Ini adalah **supervised learning regression** untuk recommendation scoring.

Kenapa bukan collaborative filtering klasik:
1. Kita punya label target langsung (probabilitas rekomendasi).
2. Prediksi dibuat dari fitur behavior user, fitur produk, dan konteks.
3. Output model adalah skor probabilitas yang bisa dipakai untuk keputusan rekomendasi.

## 2. Arsitektur End-to-End

Alur project:
1. `train.py` membaca dataset CSV.
2. Nama kolom dibersihkan ke `snake_case`.
3. Data dipisah train/test.
4. Preprocessing numerik + kategorikal dijalankan lewat `ColumnTransformer`.
5. Beberapa model baseline dilatih dan dibandingkan.
6. Model terbaik dipilih berdasarkan RMSE.
7. Artifact disimpan ke `models/`.
8. `app.py` memuat model dan menyajikan endpoint prediksi.

## 3. Struktur Folder

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
├── data/
│   └── content_based_recommendation_dataset.csv
└── notebooks/
    └── optional_eda.ipynb
```

## 4. Penjelasan Setiap File

1. `config.py`
Berisi konfigurasi path, mapping kolom, daftar fitur numerik/kategorikal, random state, threshold label rekomendasi.

2. `utils.py`
Berisi fungsi reusable:
- load dan validasi dataset
- build preprocessor
- training multi model
- evaluasi metrik
- simpan/load model
- validasi input API single dan batch

3. `train.py`
Entrypoint training model. Script ini menjalankan training end-to-end dan menyimpan artifact.

4. `app.py`
Flask API server dengan endpoint health, train, predict single, predict batch.

5. `predict_example.py`
Contoh client sederhana untuk memanggil endpoint `/predict`.

6. `notebooks/optional_eda.ipynb`
Notebook belajar yang menjelaskan statistik preprocessing hingga hasil model.

## 5. Dataset dan Mapping Kolom

Kolom target:
- `Probability for the product to be recommended to the person`

Kolom API memakai format `snake_case`. Mapping utama:
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

## 6. Setup Dari Nol

Jalankan dari folder `project`.

1. Buat virtual environment:
```bash
python3 -m venv .venv
```

2. Aktifkan virtual environment:
```bash
source .venv/bin/activate
```

3. Install dependency:
```bash
pip install -r requirements.txt
```

## 7. Konfigurasi Git

File Git sudah disiapkan di root repo:
- `../.gitignore`
- `../.gitattributes`

Konfigurasi dasar Git:
```bash
git config user.name "Nama Anda"
git config user.email "email@anda.com"
git init
git branch -M main
git add .
git commit -m "Initialize recommendation scoring project"
```

## 8. Training Model

Perintah standar:
```bash
python train.py
```

Perintah custom path:
```bash
python train.py \
  --data-path data/content_based_recommendation_dataset.csv \
  --model-path models/best_model.joblib \
  --metadata-path models/model_metadata.json
```

Output training:
1. `models/best_model.joblib`
2. `models/model_metadata.json`

Metadata berisi:
1. model terpilih
2. metrik semua kandidat model
3. feature columns
4. target column
5. mapping kolom
6. timestamp training

## 9. Preprocessing dan Modeling Yang Dipakai

Preprocessing:
1. Fitur numerik:
- `SimpleImputer(strategy="median")`
- `StandardScaler()`
2. Fitur kategorikal:
- `SimpleImputer(strategy="most_frequent")`
- `OneHotEncoder(handle_unknown="ignore")`

Model baseline:
1. `LinearRegression`
2. `RandomForestRegressor`
3. `GradientBoostingRegressor`
4. `XGBRegressor` opsional jika dependency ada

Metrik evaluasi:
1. MAE
2. RMSE
3. R2

Kriteria model terbaik:
1. RMSE paling kecil

## 10. Menjalankan API

Jalankan server:
```bash
python app.py
```

Base URL default:
- `http://127.0.0.1:5000`

Endpoint:
1. `GET /`
2. `GET /health`
3. `POST /train`
4. `POST /predict`
5. `POST /predict_batch`

Catatan:
1. Jika model belum tersedia, endpoint prediksi mengembalikan HTTP `503`.
2. Input validation error mengembalikan HTTP `400`.
3. Kategori baru tidak membuat crash karena `handle_unknown="ignore"`.

## 11. Contoh Request API

Contoh `POST /predict`:
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
  "predicted_probability": 0.871233,
  "recommendation_label": "recommended"
}
```

Rule label:
1. jika `predicted_probability >= 0.7` maka `recommended`
2. jika `< 0.7` maka `not_recommended`

Contoh `POST /predict_batch`:
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

## 12. Menjalankan Notebook Belajar

Notebook untuk belajar detail preprocessing dan hasil model:
- `notebooks/optional_eda.ipynb`

Langkah menjalankan:
1. Pastikan dependency sudah terinstall.
2. Install jupyter jika belum ada:
```bash
pip install notebook
```
3. Jalankan:
```bash
jupyter notebook notebooks/optional_eda.ipynb
```
4. Jalankan cell dari atas ke bawah (`Run All`).

## 13. Troubleshooting Cepat

1. Error `Dataset tidak ditemukan`
Periksa path file ada di `data/content_based_recommendation_dataset.csv`.

2. Error endpoint predict `Model belum tersedia`
Jalankan training dulu via `python train.py` atau `POST /train`.

3. Error validasi input `Field wajib hilang`
Pastikan semua field fitur dikirim lengkap dengan nama `snake_case`.

4. Error `ModuleNotFoundError`
Pastikan virtual environment aktif dan dependency terinstall.

## 14. Roadmap Belajar Selanjutnya

1. Tambahkan cross-validation.
2. Lakukan hyperparameter tuning.
3. Tambahkan monitoring performa model.
4. Tambahkan test otomatis untuk endpoint API.
