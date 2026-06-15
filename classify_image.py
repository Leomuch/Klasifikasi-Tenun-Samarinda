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
            "display_name": DISPLAY_NAMES.get(class_name, class_name)
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

        arr = np.asarray(img, dtype=np.float32)

    if PREPROCESS_MODE == "vgg16":
        arr = preprocess_input(arr)
    elif PREPROCESS_MODE == "rescale":
        arr = arr / 255.0
    else:
        raise ValueError("PREPROCESS_MODE harus 'vgg16' atau 'rescale'.")

    return np.expand_dims(arr, axis=0)


def extract_single_feature(feature_model, image_path: Path):
    batch = load_and_preprocess_image(image_path)
    feature = feature_model.predict(batch, verbose=0)
    return feature.astype(np.float32)


def predict_feature(svm_model, query_feature, class_indices):
    predicted_label = int(svm_model.predict(query_feature)[0])

    label_info = class_indices.get(str(predicted_label), {})
    label_name = label_info.get("display_name", str(predicted_label))

    probability_dict = {}
    confidence = 0.0

    if hasattr(svm_model, "predict_proba"):
        probabilities = svm_model.predict_proba(query_feature)[0]

        for class_index, prob in zip(svm_model.classes_, probabilities):
            class_index = int(class_index)
            class_name = CLASS_ORDER[class_index]
            display_name = DISPLAY_NAMES.get(class_name, class_name)

            probability_dict[display_name] = float(prob * 100)

            if class_index == predicted_label:
                confidence = float(prob * 100)

    return {
        "predicted_label": predicted_label,
        "label_name": label_name,
        "confidence": confidence,
        "probability_dict": probability_dict
    }


def classify_image(image_path: Path, model_dir: Path):
    feature_model = load_feature_extractor()
    svm_model = load_svm_model(model_dir)
    class_indices = load_class_indices(model_dir)

    query_feature = extract_single_feature(feature_model, image_path)

    prediction = predict_feature(
        svm_model=svm_model,
        query_feature=query_feature,
        class_indices=class_indices
    )

    return prediction, query_feature


def main():
    parser = argparse.ArgumentParser(
        description="Klasifikasi satu citra motif Tenun Samarinda."
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

    image_path = Path(args.image_path)
    model_dir = Path(args.model_dir)

    if not image_path.exists():
        raise FileNotFoundError(f"Citra tidak ditemukan: {image_path}")

    prediction, _ = classify_image(image_path, model_dir)

    print("=" * 70)
    print("HASIL KLASIFIKASI CITRA")
    print("=" * 70)
    print(f"Prediksi motif : {prediction['label_name']}")
    print(f"Confidence     : {prediction['confidence']:.2f}%")

    if prediction["probability_dict"]:
        print("\nProbabilitas per kelas:")
        for label_name, prob in prediction["probability_dict"].items():
            print(f"- {label_name:20s}: {prob:.2f}%")


if __name__ == "__main__":
    main()