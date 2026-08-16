from pathlib import Path
import numpy as np

from vgg16 import load_feature_extractor

from classify_image import (
    load_svm_model,
    load_class_indices,
    extract_single_feature,
    predict_feature
)

from retrieval import retrieve_similar


MODEL_DIR = Path("models")
TOP_K = 5


print("Memuat model...")

feature_model = load_feature_extractor()
svm_model = load_svm_model(MODEL_DIR)
class_indices = load_class_indices(MODEL_DIR)

print("Model berhasil dimuat.")


data = np.load(
    MODEL_DIR / "split_dataset.npz",
    allow_pickle=True
)

paths_test = data["paths_test"]
y_test = data["y_test"]


precision_scores = []


for image_path, true_label in zip(paths_test, y_test):

    image_path = Path(str(image_path))

    query_feature = extract_single_feature(
        feature_model,
        image_path
    )

    prediction = predict_feature(
        svm_model=svm_model,
        query_feature=query_feature,
        class_indices=class_indices
    )

    results = retrieve_similar(
        query_feature=query_feature,
        model_dir=MODEL_DIR,
        top_k=TOP_K
    )

    relevant = 0

    for item in results:

        if item["label_index"] == int(true_label):
            relevant += 1

    precision_at_5 = relevant / TOP_K

    precision_scores.append(
        precision_at_5
    )


mean_precision = np.mean(
    precision_scores
)


print("\n" + "=" * 60)
print("EVALUASI RETRIEVAL")
print("=" * 60)

print(
    f"Jumlah query      : "
    f"{len(precision_scores)}"
)

print(
    f"Top-K             : "
    f"{TOP_K}"
)

print(
    f"Mean Precision@5  : "
    f"{mean_precision:.4f}"
)