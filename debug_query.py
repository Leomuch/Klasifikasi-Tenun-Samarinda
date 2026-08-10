from pathlib import Path

import joblib
import numpy as np

from classify_image import (
    load_feature_extractor,
    extract_single_feature
)

from config import (
    CLASS_ORDER,
    DISPLAY_NAMES
)


MODEL_DIR = Path("models")

QUERY_PATH = Path(
    r"D:\Desktop\tes sistem\hatta888.jpg"
)


# ============================================================
# LOAD MODEL
# ============================================================

feature_model = load_feature_extractor()

svm_model = joblib.load(
    MODEL_DIR / "svm_cnn_model.pkl"
)


# ============================================================
# LOAD TRAINING DATA
# ============================================================

data = np.load(
    MODEL_DIR / "split_dataset.npz",
    allow_pickle=True
)

X_train = data["X_train"]
y_train = data["y_train"]

print("=" * 70)
print("DATA TRAIN")
print("=" * 70)

print("X_train :", X_train.shape)
print("y_train :", y_train.shape)


# ============================================================
# EXTRACT QUERY
# ============================================================

query = extract_single_feature(
    feature_model,
    QUERY_PATH
)

query = query.reshape(1, -1)


print("\n" + "=" * 70)
print("QUERY")
print("=" * 70)

print("Query shape:", query.shape)


# ============================================================
# STANDARD SCALER
# ============================================================

scaler = svm_model.named_steps["scaler"]
svm = svm_model.named_steps["svm"]


X_train_scaled = scaler.transform(
    X_train
)

query_scaled = scaler.transform(
    query
)


# ============================================================
# DISTANCE RAW FEATURE
# ============================================================

raw_distances = np.linalg.norm(
    X_train - query,
    axis=1
)


# ============================================================
# DISTANCE STANDARDIZED FEATURE
# ============================================================

scaled_distances = np.linalg.norm(
    X_train_scaled - query_scaled,
    axis=1
)


# ============================================================
# ANALISIS SETIAP KELAS
# ============================================================

print("\n" + "=" * 70)
print("DISTANCE QUERY TERHADAP SETIAP KELAS")
print("=" * 70)


for class_index in sorted(
    np.unique(y_train)
):

    mask = (
        y_train == class_index
    )

    raw = raw_distances[mask]
    scaled = scaled_distances[mask]

    class_name = CLASS_ORDER[
        int(class_index)
    ]

    display_name = DISPLAY_NAMES.get(
        class_name,
        class_name
    )

    print(
        f"\n{display_name}"
    )

    print(
        f"Jumlah        : {len(raw)}"
    )

    print(
        f"Raw minimum   : {np.min(raw):.4f}"
    )

    print(
        f"Raw mean      : {np.mean(raw):.4f}"
    )

    print(
        f"Scaled minimum: {np.min(scaled):.4f}"
    )

    print(
        f"Scaled mean   : {np.mean(scaled):.4f}"
    )


# ============================================================
# 10 NEAREST RAW
# ============================================================

print("\n" + "=" * 70)
print("10 NEAREST DATASET - RAW FEATURE")
print("=" * 70)


nearest = np.argsort(
    raw_distances
)[:10]


for rank, idx in enumerate(
    nearest,
    start=1
):

    label_index = int(
        y_train[idx]
    )

    class_name = CLASS_ORDER[
        label_index
    ]

    display_name = DISPLAY_NAMES.get(
        class_name,
        class_name
    )

    print(
        f"{rank:2d}. "
        f"{display_name:25s} "
        f"distance={raw_distances[idx]:.4f}"
    )


# ============================================================
# 10 NEAREST STANDARDIZED
# ============================================================

print("\n" + "=" * 70)
print("10 NEAREST DATASET - STANDARDIZED FEATURE")
print("=" * 70)


nearest_scaled = np.argsort(
    scaled_distances
)[:10]


for rank, idx in enumerate(
    nearest_scaled,
    start=1
):

    label_index = int(
        y_train[idx]
    )

    class_name = CLASS_ORDER[
        label_index
    ]

    display_name = DISPLAY_NAMES.get(
        class_name,
        class_name
    )

    print(
        f"{rank:2d}. "
        f"{display_name:25s} "
        f"distance={scaled_distances[idx]:.4f}"
    )


# ============================================================
# SVM DECISION
# ============================================================

print("\n" + "=" * 70)
print("SVM DECISION FUNCTION")
print("=" * 70)


decision = svm.decision_function(
    query_scaled
)[0]


for class_index, score in zip(
    svm.classes_,
    decision
):

    class_index = int(
        class_index
    )

    class_name = CLASS_ORDER[
        class_index
    ]

    display_name = DISPLAY_NAMES.get(
        class_name,
        class_name
    )

    print(
        f"{display_name:25s}: "
        f"{score:.6f}"
    )


# ============================================================
# MODEL PARAMETERS
# ============================================================

print("\n" + "=" * 70)
print("SVM PARAMETERS")
print("=" * 70)

print("Kernel :", svm.kernel)
print("C      :", svm.C)
print(
    "Class weight:",
    svm.class_weight
)

print("=" * 70)