import argparse
from pathlib import Path

import numpy as np

from config import (
    CLASS_ORDER,
    DISPLAY_NAMES,
)

from classify_image import (
    load_feature_extractor,
    load_svm_model,
    load_class_indices,
    extract_single_feature,
    predict_feature
)


def load_feature_database(model_dir: Path):
    db_path = model_dir / "retrieval_database.npz"

    if not db_path.exists():
        raise FileNotFoundError(
            f"Database retrieval tidak ditemukan: {db_path}. "
            "Jalankan retrieval_db.py terlebih dahulu."
        )

    db = np.load(db_path, allow_pickle=True)

    features = db["features"]
    labels = db["labels"]
    paths = db["paths"]

    print("\n" + "=" * 70)
    print("INFORMASI DATABASE RETRIEVAL")
    print("=" * 70)

    print(f"Path database     : {db_path}")
    print(f"Shape features    : {features.shape}")
    print(f"Shape labels      : {labels.shape}")
    print(f"Shape paths       : {paths.shape}")
    print(f"Jumlah citra      : {len(labels)}")
    print(f"Jumlah fitur      : {features.shape[1]}")

    # Distribusi kelas
    print("\nDistribusi kelas database:")

    unique_labels, counts = np.unique(
        labels,
        return_counts=True
    )

    for label, count in zip(unique_labels, counts):
        label_index = int(label)

        class_name = CLASS_ORDER[label_index]

        label_name = DISPLAY_NAMES.get(
            class_name,
            class_name
        )

        print(
            f"- {label_name:25s}: "
            f"{count:4d} citra"
        )

    print("=" * 70)

    return {
        "features": features,
        "labels": labels,
        "paths": paths
    }


def retrieve_similar(
    query_feature,
    model_dir: Path,
    top_k=5
):
    """
    Mencari citra paling mirip dari SELURUH kelas.

    Retrieval tidak menggunakan hasil prediksi SVM
    sebagai filter.
    """

    feature_db = load_feature_database(model_dir)

    db_features = feature_db["features"]
    db_labels = feature_db["labels"]
    db_paths = feature_db["paths"]

    print("\n" + "=" * 70)
    print("INFORMASI QUERY FEATURE")
    print("=" * 70)

    print(
        f"Shape query sebelum reshape : "
        f"{query_feature.shape}"
    )

    # Ubah menjadi (1, jumlah_fitur)
    query_feature = query_feature.reshape(1, -1)

    print(
        f"Shape query setelah reshape  : "
        f"{query_feature.shape}"
    )

    print(
        f"Jumlah fitur database        : "
        f"{db_features.shape[1]}"
    )

    print(
        f"Jumlah fitur query           : "
        f"{query_feature.shape[1]}"
    )

    print("=" * 70)

    # ==========================================================
    # VALIDASI DIMENSI
    # ==========================================================

    if db_features.ndim != 2:
        raise ValueError(
            "\nDatabase retrieval memiliki bentuk yang tidak valid!\n"
            f"Shape database: {db_features.shape}\n"
            "Database harus berbentuk (jumlah_citra, jumlah_fitur)."
        )

    if query_feature.ndim != 2:
        raise ValueError(
            "\nQuery feature memiliki bentuk yang tidak valid!\n"
            f"Shape query: {query_feature.shape}"
        )

    if db_features.shape[1] != query_feature.shape[1]:
        raise ValueError(
            "\n"
            + "=" * 70
            + "\n"
            "DIMENSI FITUR TIDAK COCOK!\n"
            + "=" * 70
            + "\n"
            f"Database : {db_features.shape}\n"
            f"Query    : {query_feature.shape}\n\n"
            f"Database menggunakan "
            f"{db_features.shape[1]} fitur.\n"
            f"Query menggunakan "
            f"{query_feature.shape[1]} fitur.\n\n"
            "Pastikan database retrieval dan feature extractor "
            "menggunakan konfigurasi VGG16 yang sama.\n"
            "Untuk penelitian Anda, targetnya adalah:\n"
            "Database : (jumlah_citra, 512)\n"
            "Query    : (1, 512)\n"
            + "=" * 70
        )

    # ==========================================================
    # INFORMASI FITUR
    # ==========================================================

    print("\nDimensi fitur VALID.")

    if db_features.shape[1] == 512:
        print(
            "✓ Fitur menggunakan 512 dimensi "
            "(Block4 + Global Average Pooling)."
        )
    else:
        print(
            "⚠️ Jumlah fitur bukan 512."
        )

    # ==========================================================
    # EUCLIDEAN DISTANCE
    # ==========================================================

    print("\nMenghitung Euclidean Distance...")

    distance = np.linalg.norm(
        db_features - query_feature,
        axis=1
    )

    print(
        f"Distance minimum : "
        f"{np.min(distance):.4f}"
    )

    print(
        f"Distance maksimum : "
        f"{np.max(distance):.4f}"
    )

    # ==========================================================
    # SEMUA KELAS
    # ==========================================================

    candidate_indices = np.arange(
        len(db_labels)
    )

    if len(candidate_indices) == 0:
        return []

    # Urutkan berdasarkan distance terkecil
    candidate_distance = distance[
        candidate_indices
    ]

    sorted_local_indices = np.argsort(
        candidate_distance
    )[:top_k]

    top_indices = candidate_indices[
        sorted_local_indices
    ]

    # ==========================================================
    # HASIL RETRIEVAL
    # ==========================================================

    results = []

    for rank, idx in enumerate(
        top_indices,
        start=1
    ):

        label_index = int(
            db_labels[idx]
        )

        class_name = CLASS_ORDER[
            label_index
        ]

        label_name = DISPLAY_NAMES.get(
            class_name,
            class_name
        )

        results.append({
            "rank": rank,
            "path": str(db_paths[idx]),
            "label_index": label_index,
            "label_name": label_name,
            "distance": float(
                distance[idx]
            )
        })

    return results


def classify_and_retrieve(
    image_path: Path,
    model_dir: Path,
    top_k=5
):
    """
    1. Ekstraksi fitur query menggunakan VGG16
    2. Klasifikasi menggunakan SVM
    3. Retrieval dari SELURUH kelas

    Hasil klasifikasi SVM tidak digunakan
    untuk membatasi database retrieval.
    """

    print("\n" + "=" * 70)
    print("MEMUAT FEATURE EXTRACTOR")
    print("=" * 70)

    feature_model = load_feature_extractor()

    print(
        f"Feature extractor output shape : "
        f"{feature_model.output_shape}"
    )

    print("=" * 70)

    svm_model = load_svm_model(
        model_dir
    )

    class_indices = load_class_indices(
        model_dir
    )

    # ==========================================================
    # EKSTRAKSI QUERY
    # ==========================================================

    print("\n" + "=" * 70)
    print("EKSTRAKSI FITUR QUERY")
    print("=" * 70)

    query_feature = extract_single_feature(
        feature_model=feature_model,
        image_path=image_path
    )

    print(
        f"Query feature shape : "
        f"{query_feature.shape}"
    )

    print("=" * 70)

    # ==========================================================
    # KLASIFIKASI SVM
    # ==========================================================

    print("\nMelakukan klasifikasi SVM...")

    prediction = predict_feature(
        svm_model=svm_model,
        query_feature=query_feature,
        class_indices=class_indices
    )

    # ==========================================================
    # RETRIEVAL SEMUA KELAS
    # ==========================================================

    print("\nMelakukan retrieval dari seluruh kelas...")

    results = retrieve_similar(
        query_feature=query_feature,
        model_dir=model_dir,
        top_k=top_k
    )

    return prediction, results


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Klasifikasi citra lalu retrieval "
            "Top-K citra paling mirip dari semua kelas."
        )
    )

    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path citra query."
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        default="models",
        help="Folder model."
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Jumlah hasil retrieval."
    )

    args = parser.parse_args()

    image_path = Path(
        args.image_path
    )

    model_dir = Path(
        args.model_dir
    )

    if not image_path.exists():
        raise FileNotFoundError(
            f"Citra tidak ditemukan: "
            f"{image_path}"
        )

    prediction, results = classify_and_retrieve(
        image_path=image_path,
        model_dir=model_dir,
        top_k=args.top_k
    )

    # ==========================================================
    # HASIL KLASIFIKASI
    # ==========================================================

    print("\n" + "=" * 70)
    print("HASIL KLASIFIKASI")
    print("=" * 70)

    print(
        f"Prediksi motif : "
        f"{prediction['label_name']}"
    )

    print(
        f"Confidence     : "
        f"{prediction['confidence']:.2f}%"
    )

    # ==========================================================
    # PROBABILITAS
    # ==========================================================

    if prediction["probability_dict"]:

        print("\nProbabilitas per kelas:")

        for label_name, prob in (
            prediction[
                "probability_dict"
            ].items()
        ):

            print(
                f"- {label_name:25s}: "
                f"{prob:.2f}%"
            )

    # ==========================================================
    # HASIL RETRIEVAL
    # ==========================================================

    print("\n" + "=" * 70)

    print(
        f"TOP-{args.top_k} "
        "CITRA PALING MIRIP (SEMUA KELAS)"
    )

    print("=" * 70)

    if not results:

        print(
            "Tidak ada hasil retrieval."
        )

    else:

        for item in results:

            print(
                f"{item['rank']}. "
                f"{item['label_name']} | "
                f"distance="
                f"{item['distance']:.4f} | "
                f"{item['path']}"
            )


if __name__ == "__main__":
    main()