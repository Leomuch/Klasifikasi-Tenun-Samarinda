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

print("Query shape :", query.shape)

# ============================================================
# RAW EUCLIDEAN DISTANCE
# ============================================================

raw_distances = np.linalg.norm(
    X_train - query,
    axis=1
)

# ============================================================
# DISTANCE PER CLASS
# ============================================================

print("\n" + "=" * 70)
print("DISTANCE QUERY TERHADAP SETIAP KELAS")
print("=" * 70)

for class_index in sorted(np.unique(y_train)):

    mask = y_train == class_index

    distances = raw_distances[mask]

    class_name = CLASS_ORDER[
        int(class_index)
    ]

    display_name = DISPLAY_NAMES.get(
        class_name,
        class_name
    )

    print(f"\n{display_name}")

    print(
        f"Jumlah  : {len(distances)}"
    )

    print(
        f"Minimum : {np.min(distances):.6f}"
    )

    print(
        f"Maximum : {np.max(distances):.6f}"
    )

    print(
        f"Mean    : {np.mean(distances):.6f}"
    )

    print(
        f"Median  : {np.median(distances):.6f}"
    )

# ============================================================
# TOP 20 NEAREST NEIGHBOR
# ============================================================

print("\n" + "=" * 70)
print("20 NEAREST TRAINING DATA")
print("=" * 70)

nearest = np.argsort(
    raw_distances
)[:20]

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
        f"distance={raw_distances[idx]:.6f}"
    )

# ============================================================
# TOP-K DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("DISTRIBUSI NEAREST NEIGHBOR")
print("=" * 70)

for k in [5, 10, 20]:

    top_indices = nearest[:k]

    top_labels = y_train[
        top_indices
    ]

    print(f"\nTop-{k}:")

    for class_index in sorted(
        np.unique(y_train)
    ):

        count = np.sum(
            top_labels == class_index
        )

        class_name = CLASS_ORDER[
            int(class_index)
        ]

        display_name = DISPLAY_NAMES.get(
            class_name,
            class_name
        )

        percentage = (
            count / k
        ) * 100

        print(
            f"- {display_name:25s}: "
            f"{count:2d}/{k} "
            f"({percentage:.1f}%)"
        )

# ============================================================
# SVM DECISION FUNCTION
# ============================================================

print("\n" + "=" * 70)
print("SVM DECISION FUNCTION")
print("=" * 70)

# Karena model tanpa StandardScaler,
# query langsung digunakan.

svm = svm_model

if hasattr(
    svm_model,
    "named_steps"
):

    if "svm" in svm_model.named_steps:
        svm = svm_model.named_steps["svm"]

decision = svm.decision_function(
    query
)

decision = np.asarray(
    decision
)

if decision.ndim == 2:
    decision = decision[0]

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

print("\n" + "=" * 70)

print(
    "SVM prediction :",
    svm_model.predict(query)
)