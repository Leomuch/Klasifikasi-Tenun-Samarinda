"""
PENGUJIAN WAKTU PROSES SISTEM (poin 8c).

Mengukur waktu rata-rata per citra untuk tiga tahap:
    1. Ekstraksi fitur VGG16 (Block4 + GAP)
    2. Klasifikasi SVM (predict + predict_proba)
    3. Retrieval (Euclidean distance ke seluruh galeri + urutkan Top-5)

Model, feature extractor, dan database retrieval dimuat SEKALI di awal
supaya angka yang diukur murni waktu proses per citra, bukan waktu loading.

Jalankan dari root repo (folder yang berisi config.py):
    python timing_test.py

Prasyarat: sudah menjalankan split_dataset, retrieval_db.py, dan train.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from vgg16 import load_feature_extractor, load_and_preprocess_image
from classify_image import load_svm_model

MODEL_DIR = Path("models")
SPLIT_PATH = MODEL_DIR / "split_dataset.npz"
DB_PATH = MODEL_DIR / "retrieval_database.npz"
TOP_K = 5


def main():
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"{SPLIT_PATH} tidak ada. Jalankan split_dataset dulu."
        )

    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"{DB_PATH} tidak ada. Jalankan retrieval_db.py dulu."
        )

    print("Memuat feature extractor, model SVM, dan database retrieval...")

    feature_model = load_feature_extractor()
    svm_model = load_svm_model(MODEL_DIR)

    db = np.load(DB_PATH, allow_pickle=True)
    db_features = db["features"]

    split = np.load(SPLIT_PATH, allow_pickle=True)
    paths_test = [Path(p) for p in split["paths_test"]]

    print(f"Jumlah citra uji : {len(paths_test)}")
    print(f"Dimensi fitur    : {db_features.shape[1]}")

    # ------------------------------------------------------------
    # WARMUP: inferensi pertama TensorFlow selalu lebih lambat
    # (tidak ikut dihitung)
    # ------------------------------------------------------------
    print("\nWarmup...")

    for p in paths_test[:2]:
        arr = load_and_preprocess_image(p)
        feat = feature_model.predict(arr[None, ...], verbose=0)
        _ = svm_model.predict(feat.reshape(1, -1))

    # ------------------------------------------------------------
    # UKUR WAKTU (dalam milidetik)
    # ------------------------------------------------------------
    t_extract = []
    t_classify = []
    t_retrieve = []

    print("Mengukur waktu proses per citra...\n")

    for p in paths_test:

        # 1. Ekstraksi fitur VGG16
        t0 = time.perf_counter()
        arr = load_and_preprocess_image(p)
        feat = feature_model.predict(arr[None, ...], verbose=0)
        t1 = time.perf_counter()

        q = feat.reshape(1, -1)

        # 2. Klasifikasi SVM
        t2 = time.perf_counter()
        _ = svm_model.predict(q)
        _ = svm_model.predict_proba(q)
        t3 = time.perf_counter()

        # 3. Retrieval: jarak Euclidean ke seluruh galeri + urutkan Top-K
        t4 = time.perf_counter()
        distances = np.linalg.norm(db_features - q, axis=1)
        _ = np.argsort(distances)[:TOP_K]
        t5 = time.perf_counter()

        t_extract.append((t1 - t0) * 1000.0)
        t_classify.append((t3 - t2) * 1000.0)
        t_retrieve.append((t5 - t4) * 1000.0)

    # ------------------------------------------------------------
    # RINGKASAN
    # ------------------------------------------------------------
    def stat(name, arr):
        arr = np.array(arr)
        return {
            "Tahap": name,
            "Rata-rata (ms)": round(float(arr.mean()), 3),
            "Std (ms)": round(float(arr.std()), 3),
            "Min (ms)": round(float(arr.min()), 3),
            "Maks (ms)": round(float(arr.max()), 3),
        }

    total = (
        np.array(t_extract)
        + np.array(t_classify)
        + np.array(t_retrieve)
    )

    rows = [
        stat("1. Ekstraksi fitur VGG16", t_extract),
        stat("2. Klasifikasi SVM", t_classify),
        stat("3. Retrieval (Top-5)", t_retrieve),
        stat("TOTAL per citra", total),
    ]

    df = pd.DataFrame(rows)

    print("=" * 70)
    print("HASIL PENGUJIAN WAKTU PROSES")
    print("=" * 70)
    print(df.to_string(index=False))
    print("=" * 70)
    print(f"Rata-rata total per citra : {total.mean():.2f} ms")
    print(f"Throughput                : {1000.0 / total.mean():.2f} citra/detik")
    print(
        "\nCatatan: waktu bergantung hardware. "
        "Sebutkan CPU/GPU, RAM, dan apakah TensorFlow memakai GPU di skripsi."
    )

    out = MODEL_DIR / "timing_report.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"\nTabel disimpan di: {out}")


if __name__ == "__main__":
    main()
