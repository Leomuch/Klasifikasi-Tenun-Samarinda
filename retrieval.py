import argparse
from pathlib import Path

import numpy as np

from config import (
    CLASS_ORDER,
    DISPLAY_NAMES,
    VGG16_FEATURES_FILENAME,
)

from classify_image import (
    load_feature_extractor,
    load_svm_model,
    load_class_indices,
    extract_single_feature,
    predict_feature
)


def load_feature_database(model_dir: Path):
    db_path = model_dir / "retrieval_database.npz" # VGG16_FEATURES_FILENAME

    if not db_path.exists():
        raise FileNotFoundError(
            f"Database retrieval tidak ditemukan: {db_path}. "
            "Jalankan vgg16.py terlebih dahulu."
        )

    db = np.load(db_path, allow_pickle=True)

    return {
        "features": db["features"],
        "labels": db["labels"],
        "paths": db["paths"]
    }


def retrieve_similar(
    query_feature,
    model_dir: Path,
    predicted_label=None,
    top_k=5
):
    feature_db = load_feature_database(model_dir)

    db_features = feature_db["features"]
    db_labels = feature_db["labels"]
    db_paths = feature_db["paths"]

    query_feature = query_feature.reshape(1, -1)

    distance = np.linalg.norm(
        db_features - query_feature,
        axis=1
    )

    if predicted_label is not None:
        candidate_indices = np.where(db_labels == predicted_label)[0]
    else:
        candidate_indices = np.arange(len(db_labels))

    if len(candidate_indices) == 0:
        return []

    candidate_distance = distance[candidate_indices]
    sorted_local_indices = np.argsort(candidate_distance)[:top_k]
    top_indices = candidate_indices[sorted_local_indices]

    results = []

    for rank, idx in enumerate(top_indices, start=1):
        label_index = int(db_labels[idx])
        class_name = CLASS_ORDER[label_index]
        label_name = DISPLAY_NAMES.get(class_name, class_name)

        results.append({
            "rank": rank,
            "path": str(db_paths[idx]),
            "label_index": label_index,
            "label_name": label_name,
            "distance": float(distance[idx])
        })

    return results


def classify_and_retrieve(
    image_path: Path,
    model_dir: Path,
    top_k=5,
    same_class_only=True
):
    feature_model = load_feature_extractor()
    svm_model = load_svm_model(model_dir)
    class_indices = load_class_indices(model_dir)

    query_feature = extract_single_feature(feature_model, image_path)

    prediction = predict_feature(
        svm_model=svm_model,
        query_feature=query_feature,
        class_indices=class_indices
    )

    predicted_label = prediction["predicted_label"]

    retrieval_label = predicted_label if same_class_only else None

    results = retrieve_similar(
        query_feature=query_feature,
        model_dir=model_dir,
        predicted_label=retrieval_label,
        top_k=top_k
    )

    return prediction, results


def main():
    parser = argparse.ArgumentParser(
        description="Klasifikasi citra lalu retrieval Top-K citra paling mirip."
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
    parser.add_argument(
        "--all_classes",
        action="store_true",
        help="Jika dipakai, retrieval dilakukan dari semua kelas."
    )

    args = parser.parse_args()

    image_path = Path(args.image_path)
    model_dir = Path(args.model_dir)

    if not image_path.exists():
        raise FileNotFoundError(f"Citra tidak ditemukan: {image_path}")

    prediction, results = classify_and_retrieve(
        image_path=image_path,
        model_dir=model_dir,
        top_k=args.top_k,
        same_class_only=not args.all_classes
    )

    print("=" * 70)
    print("HASIL KLASIFIKASI")
    print("=" * 70)
    print(f"Prediksi motif : {prediction['label_name']}")
    print(f"Confidence     : {prediction['confidence']:.2f}%")

    print("\n" + "=" * 70)
    print(f"TOP-{args.top_k} CITRA PALING MIRIP")
    print("=" * 70)

    for item in results:
        print(
            f"{item['rank']}. "
            f"{item['label_name']} | "
            f"distance={item['distance']:.4f} | "
            f"{item['path']}"
        )


if __name__ == "__main__":
    main()