import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from PIL import Image, ImageOps

from tensorflow.keras.applications.vgg16 import preprocess_input
from vgg16 import load_feature_extractor

from config import (
    CLASS_ORDER,
    DISPLAY_NAMES,
    IMG_SIZE,
    PREPROCESS_MODE,
    MODEL_FILENAME,
    CLASS_INDEX_FILENAME
)


def load_svm_model(model_dir: Path):
    model_path = model_dir / MODEL_FILENAME

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model SVM tidak ditemukan: {model_path}. "
            "Jalankan train_model.py terlebih dahulu."
        )

    return joblib.load(model_path)


def load_class_indices(model_dir: Path):
    class_index_path = model_dir / CLASS_INDEX_FILENAME

    if class_index_path.exists():
        with open(class_index_path, "r", encoding="utf-8") as f:
            return json.load(f)

    class_indices = {}

    for i, class_name in enumerate(CLASS_ORDER):
        class_indices[str(i)] = {
            "folder_name": class_name,
            "display_name": DISPLAY_NAMES.get(
                class_name,
                class_name
            )
        }

    return class_indices


def load_and_preprocess_image(image_path: Path):

    with Image.open(image_path) as img:

        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")

        img = ImageOps.fit(
            img,
            IMG_SIZE,
            method=Image.Resampling.LANCZOS
        )

        arr = np.asarray(
            img,
            dtype=np.float32
        )

    if PREPROCESS_MODE == "vgg16":

        arr = preprocess_input(arr)

    elif PREPROCESS_MODE == "rescale":

        arr = arr / 255.0

    else:

        raise ValueError(
            "PREPROCESS_MODE harus "
            "'vgg16' atau 'rescale'."
        )

    return np.expand_dims(
        arr,
        axis=0
    )


def extract_single_feature(
    feature_model,
    image_path: Path
):

    batch = load_and_preprocess_image(
        image_path
    )

    feature = feature_model.predict(
        batch,
        verbose=0
    )

    feature = feature.astype(
        np.float32
    )

    print(
        f"\nQuery feature shape : "
        f"{feature.shape}"
    )

    return feature


def predict_feature(
    svm_model,
    query_feature,
    class_indices
):

    # ==========================================================
    # VALIDASI DIMENSI
    # ==========================================================

    print("\n" + "=" * 70)
    print("ANALISIS SVM")
    print("=" * 70)

    print(
        f"Query feature shape : "
        f"{query_feature.shape}"
    )

    # Pipeline SVM biasanya menerima:
    # (jumlah_citra, jumlah_fitur)

    if query_feature.ndim != 2:
        query_feature = query_feature.reshape(
            1,
            -1
        )

    print(
        f"Feature dimension   : "
        f"{query_feature.shape[1]}"
    )

    if query_feature.shape[1] != 512:
        print(
            "WARNING: Query tidak memiliki "
            "512 fitur."
        )
    else:
        print(
            "✓ Query menggunakan "
            "512 fitur (Block4 + GAP)."
        )

    # ==========================================================
    # PREDIKSI SVM
    # ==========================================================

    predicted_label = int(
        svm_model.predict(
            query_feature
        )[0]
    )

    label_info = class_indices.get(
        str(predicted_label),
        {}
    )

    label_name = label_info.get(
        "display_name",
        str(predicted_label)
    )

    print(
        f"\nPredicted label     : "
        f"{predicted_label}"
    )

    print(
        f"Predicted class     : "
        f"{label_name}"
    )

    # ==========================================================
    # DECISION FUNCTION
    # ==========================================================

    decision_dict = {}

    if hasattr(
        svm_model,
        "decision_function"
    ):

        decision_scores = (
            svm_model
            .decision_function(query_feature)
        )

        decision_scores = np.asarray(
            decision_scores
        )

        # Untuk multiclass SVM
        if decision_scores.ndim == 2:

            decision_scores = (
                decision_scores[0]
            )

        print(
            "\nDecision Function SVM:"
        )

        for class_index, score in zip(
            svm_model.classes_,
            decision_scores
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

            decision_dict[
                display_name
            ] = float(score)

            print(
                f"- {display_name:25s}: "
                f"{score:.6f}"
            )

        # Kelas dengan decision score terbesar
        best_decision_index = int(
            np.argmax(decision_scores)
        )

        best_decision_label = int(
            svm_model.classes_[
                best_decision_index
            ]
        )

        best_decision_name = DISPLAY_NAMES.get(
            CLASS_ORDER[
                best_decision_label
            ],
            CLASS_ORDER[
                best_decision_label
            ]
        )

        print(
            "\nKelas dengan "
            "decision score tertinggi:"
        )

        print(
            f"→ {best_decision_name}"
        )

    # ==========================================================
    # PROBABILITAS SVC
    # ==========================================================

    probability_dict = {}
    confidence = 0.0

    if hasattr(
        svm_model,
        "predict_proba"
    ):

        probabilities = (
            svm_model
            .predict_proba(
                query_feature
            )[0]
        )

        print(
            "\nProbabilitas SVC:"
        )

        for class_index, prob in zip(
            svm_model.classes_,
            probabilities
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

            probability = float(
                prob * 100
            )

            probability_dict[
                display_name
            ] = probability

            print(
                f"- {display_name:25s}: "
                f"{probability:.2f}%"
            )

            if class_index == predicted_label:

                confidence = probability

    # ==========================================================
    # INFO MODEL
    # ==========================================================

    if hasattr(
        svm_model,
        "named_steps"
    ):

        print(
            "\nKonfigurasi Pipeline:"
        )

        if "scaler" in svm_model.named_steps:

            print(
                "- StandardScaler : AKTIF"
            )

        if "svm" in svm_model.named_steps:

            svm = svm_model.named_steps[
                "svm"
            ]

            print(
                f"- SVM kernel     : "
                f"{svm.kernel}"
            )

            print(
                f"- SVM C          : "
                f"{svm.C}"
            )

            print(
                f"- Probability    : "
                f"{svm.probability}"
            )

            print(
                f"- Class weight   : "
                f"{svm.class_weight}"
            )

    print(
        "\n" + "=" * 70
    )

    return {
        "predicted_label": predicted_label,
        "label_name": label_name,
        "confidence": confidence,
        "probability_dict": probability_dict,
        "decision_dict": decision_dict
    }


def classify_image(
    image_path: Path,
    model_dir: Path
):

    feature_model = (
        load_feature_extractor()
    )

    svm_model = load_svm_model(
        model_dir
    )

    class_indices = load_class_indices(
        model_dir
    )

    query_feature = extract_single_feature(
        feature_model=feature_model,
        image_path=image_path
    )

    prediction = predict_feature(
        svm_model=svm_model,
        query_feature=query_feature,
        class_indices=class_indices
    )

    return prediction, query_feature


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Klasifikasi satu citra "
            "motif Tenun Samarinda."
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

    prediction, _ = classify_image(
        image_path,
        model_dir
    )

    print("=" * 70)
    print(
        "HASIL KLASIFIKASI CITRA"
    )
    print("=" * 70)

    print(
        f"Prediksi motif : "
        f"{prediction['label_name']}"
    )

    print(
        f"Confidence     : "
        f"{prediction['confidence']:.2f}%"
    )

    if prediction[
        "probability_dict"
    ]:

        print(
            "\nProbabilitas per kelas:"
        )

        for label_name, prob in (
            prediction[
                "probability_dict"
            ].items()
        ):

            print(
                f"- {label_name:25s}: "
                f"{prob:.2f}%"
            )


if __name__ == "__main__":
    main()