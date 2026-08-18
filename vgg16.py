import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Concatenate

from config import (
    CLASS_ORDER,
    DISPLAY_NAMES,
    IMAGE_EXTENSIONS,
    IMG_SIZE,
    BATCH_SIZE,
    PREPROCESS_MODE,
    VGG16_FEATURES_FILENAME,
    CLASS_INDEX_FILENAME
)

def collect_images(dataset_dir: Path):
    image_paths = []
    labels = []

    for class_index, class_name in enumerate(CLASS_ORDER):
        folder = dataset_dir / class_name

        if not folder.exists():
            continue

        paths = sorted([
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ])

        for path in paths:
            image_paths.append(path)
            labels.append(class_index)

    return image_paths, np.array(labels, dtype=np.int64)


def load_feature_extractor():
    """
    Memuat VGG16 pretrained ImageNet.

    KONFIGURASI AKTIF:
    Block3 + Global Average Pooling (GAP)

    Alternatif yang tersedia tetapi dikomentari:
    - Block4 + GAP
    - Block4 + Global Max Pooling
    - Block4 + Average Pooling + Max Pooling
    - Block5 + GAP
    - FC1
    - FC2
    """

    # ============================================================
    # BLOCK3 + GLOBAL AVERAGE POOLING
    # ============================================================
    base_model = VGG16(
        weights="imagenet",
        include_top=False
    )

    x = base_model.get_layer("block3_pool").output
    x = GlobalAveragePooling2D()(x)

    feature_model = Model(
        inputs=base_model.input,
        outputs=x
    )

    return feature_model

    # ============================================================
    # BLOCK4 + GLOBAL AVERAGE POOLING
    # ============================================================

    # base_model = VGG16(
    #     weights="imagenet",
    #     include_top=False
    # )
    
    # x = base_model.get_layer("block4_pool").output
    # x = GlobalAveragePooling2D()(x)
    
    # feature_model = Model(
    #     inputs=base_model.input,
    #     outputs=x
    # )
    
    # return feature_model

    # ============================================================
    # BLOCK4 + GLOBAL MAX POOLING
    # ============================================================

    # base_model = VGG16(
    #     weights="imagenet",
    #     include_top=False
    # )
    #
    # x = base_model.get_layer("block4_pool").output
    # x = GlobalMaxPooling2D()(x)
    #
    # feature_model = Model(
    #     inputs=base_model.input,
    #     outputs=x
    # )
    #
    # return feature_model

    # ============================================================
    # BLOCK4 + AVERAGE + MAX POOLING
    # ============================================================

    # base_model = VGG16(
    #     weights="imagenet",
    #     include_top=False
    # )
    #
    # x = base_model.get_layer("block4_pool").output
    #
    # avg_pool = GlobalAveragePooling2D()(x)
    # max_pool = GlobalMaxPooling2D()(x)
    #
    # x = Concatenate()([
    #     avg_pool,
    #     max_pool
    # ])
    #
    # feature_model = Model(
    #     inputs=base_model.input,
    #     outputs=x
    # )
    #
    # return feature_model

    # ============================================================
    # BLOCK5 + GLOBAL AVERAGE POOLING
    # ============================================================

    # base_model = VGG16(
    #     weights="imagenet",
    #     include_top=False
    # )
    #
    # x = base_model.get_layer("block5_pool").output
    # x = GlobalAveragePooling2D()(x)
    #
    # feature_model = Model(
    #     inputs=base_model.input,
    #     outputs=x
    # )
    #
    # return feature_model

    # ============================================================
    # Fully Connected 1
    # ============================================================

    # base_model = VGG16(
    #     weights="imagenet",
    #     include_top=True
    # )
    #
    # x = base_model.get_layer("fc1").output
    #
    # feature_model = Model(
    #     inputs=base_model.input,
    #     outputs=x
    # )
    #
    # return feature_model

    # ============================================================
    # Fully Connected 2
    # ============================================================

    # base_model = VGG16(
    #     weights="imagenet",
    #     include_top=True
    # )
    #
    # x = base_model.get_layer("fc2").output
    #
    # feature_model = Model(
    #     inputs=base_model.input,
    #     outputs=x
    # )
    #
    # return feature_model

def lihat_arsitektur_vgg16(feature_model, model_dir: Path):
    """
    Menampilkan dan menyimpan informasi arsitektur VGG16
    yang digunakan sebagai feature extractor.
    """
    print("\n" + "=" * 70)
    print("ARSITEKTUR VGG16 SEBAGAI FEATURE EXTRACTOR")
    print("=" * 70)
    print(f"Input shape  : {feature_model.input_shape}")
    print("Output layer : block3")
    print(f"Output shape : {feature_model.output_shape}")
    print("Keterangan   : Setiap citra menghasilkan 512 fitur.")
    print("=" * 70)

    # Simpan summary VGG16 ke file txt
    summary_path = model_dir / "vgg16_summary.txt"

    with open(summary_path, "w", encoding="utf-8") as f:
        feature_model.summary(print_fn=lambda x: f.write(x + "\n"))

    print(f"Summary VGG16 disimpan di: {summary_path}")

    # Simpan daftar layer VGG16 ke CSV
    rows = []

    for i, layer in enumerate(feature_model.layers):
        rows.append({
            "no": i + 1,
            "nama_layer": layer.name,
            "tipe_layer": layer.__class__.__name__,
            "output_shape": str(layer.output.shape),
            "jumlah_parameter": layer.count_params()
        })

    df_layers = pd.DataFrame(rows)

    layer_csv_path = model_dir / "vgg16_layers.csv"
    df_layers.to_csv(layer_csv_path, index=False)

    print(f"Daftar layer VGG16 disimpan di: {layer_csv_path}")


def load_and_preprocess_image(path: Path):
    with Image.open(path) as img:
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

    return arr


def extract_features(feature_model, image_paths, batch_size=BATCH_SIZE):
    all_features = []

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start:start + batch_size]

        batch_images = np.stack([
            load_and_preprocess_image(path)
            for path in batch_paths
        ], axis=0)

        features = feature_model.predict(batch_images, verbose=0)
        all_features.append(features)

        done = min(start + batch_size, len(image_paths))
        print(f"Ekstraksi fitur VGG16: {done}/{len(image_paths)} citra")

    return np.vstack(all_features).astype(np.float32)


def save_class_indices(model_dir: Path):
    class_indices = {}

    for i, class_name in enumerate(CLASS_ORDER):
        class_indices[str(i)] = {
            "folder_name": class_name,
            "display_name": DISPLAY_NAMES.get(class_name, class_name)
        }

    output_path = model_dir / CLASS_INDEX_FILENAME

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(class_indices, f, indent=2, ensure_ascii=False)

    print(f"Class indices disimpan di: {output_path}")


def save_vgg16_features(model_dir: Path, features, labels, image_paths):
    """
    Menyimpan satu file fitur VGG16 untuk dua kebutuhan:
    - features    : dipakai untuk data splitting dan training SVM
    - labels      : label numerik kelas
    - paths       : path citra dataset
    """

    output_path = model_dir / VGG16_FEATURES_FILENAME

    np.savez_compressed(
        output_path,
        features=features,
        labels=labels,
        paths=np.array([str(p) for p in image_paths], dtype=object)
    )

    print(f"Fitur VGG16 disimpan di: {output_path}")

def lihat_output_ekstraksi_fitur(model_dir: Path):
    """
    Menampilkan informasi hasil ekstraksi fitur VGG16
    dari file .npz yang sudah disimpan.
    """
    feature_path = model_dir / VGG16_FEATURES_FILENAME

    if not feature_path.exists():
        print(f"File fitur tidak ditemukan: {feature_path}")
        return

    data = np.load(feature_path, allow_pickle=True)

    features = data["features"]
    labels = data["labels"]
    paths = data["paths"]

    print("\n" + "=" * 70)
    print("OUTPUT EKSTRAKSI FITUR VGG16")
    print("=" * 70)
    print(f"Shape features    : {features.shape}")
    print(f"Shape labels      : {labels.shape}")
    print(f"Shape paths       : {paths.shape}")
    print("-" * 70)
    print(f"Contoh file pertama       : {paths[0]}")
    print(f"Label file pertama        : {labels[0]}")
    print(f"Jumlah fitur per citra    : {features.shape[1]}")
    print(f"10 fitur pertama citra 1  : {features[0][:10]}")
    print("=" * 70)

    # Buat dataframe preview fitur
    rows = []

    for i in range(len(paths)):
        image_path = Path(str(paths[i]))
        label_index = int(labels[i])
        class_folder = CLASS_ORDER[label_index]
        class_name = DISPLAY_NAMES.get(class_folder, class_folder)

        row = {
            "no": i + 1,
            "nama_file": image_path.name,
            "kelas": class_name,
            "label": label_index,
            "path": str(image_path),
            "jumlah_fitur": features.shape[1],
            "mean_fitur": float(np.mean(features[i])),
            "std_fitur": float(np.std(features[i])),
            "min_fitur": float(np.min(features[i])),
            "max_fitur": float(np.max(features[i])),
            "euclidean_norm": float(np.linalg.norm(features[i]))
        }

        # Tampilkan 20 fitur pertama saja sebagai preview
        for j in range(20):
            row[f"fitur_{j+1:04d}"] = float(features[i][j])

        rows.append(row)

    df_preview = pd.DataFrame(rows)

    preview_csv_path = model_dir / "preview_fitur_vgg16.csv"
    df_preview.to_csv(preview_csv_path, index=False)

    print(f"Preview dataframe fitur disimpan di: {preview_csv_path}")


def print_summary(labels):
    print("=" * 70)
    print("RINGKASAN FITUR DATASET")
    print("=" * 70)

    for i, class_name in enumerate(CLASS_ORDER):
        count = int(np.sum(labels == i))
        display_name = DISPLAY_NAMES.get(class_name, class_name)
        print(f"{display_name:20s}: {count:4d} citra")

    print("-" * 70)
    print(f"Total: {len(labels)} citra")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Ekstraksi fitur VGG16 Block4 512 dimensi."
    )

    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="data/processed",
        help="Folder dataset hasil preprocessing."
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        default="models",
        help="Folder output model dan fitur."
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size ekstraksi fitur."
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    image_paths, labels = collect_images(dataset_dir)

    if len(image_paths) == 0:
        raise FileNotFoundError(
            f"Tidak ada citra ditemukan di {dataset_dir}."
        )

    print_summary(labels)

    print("\nMemuat VGG16 pretrained ImageNet...")
    feature_model = load_feature_extractor()
    lihat_arsitektur_vgg16(feature_model, model_dir)

    print("\nMulai ekstraksi fitur VGG16 layer Block4...")
    features = extract_features(
        feature_model=feature_model,
        image_paths=image_paths,
        batch_size=args.batch_size
    )

    save_vgg16_features(
        model_dir=model_dir,
        features=features,
        labels=labels,
        image_paths=image_paths
    )

    save_class_indices(model_dir)

    print("\nEkstraksi fitur VGG16 selesai.")


if __name__ == "__main__":
    main()