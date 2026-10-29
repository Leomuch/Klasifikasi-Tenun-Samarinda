
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from tensorflow.keras.applications.vgg16 import (
    VGG16,
    preprocess_input,
)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    Concatenate,
)

from config import (
    CLASS_ORDER,
    DISPLAY_NAMES,
    IMAGE_EXTENSIONS,
    IMG_SIZE,
    BATCH_SIZE,
    VGG16_FEATURES_FILENAME,
    CLASS_INDEX_FILENAME,
)


# ==============================================================
# KONFIGURASI EKSPERIMEN VGG16
# ==============================================================

# KONFIGURASI AKTIF
FEATURE_MODE = "block4_gap"

# Konfigurasi alternatif: aktifkan satu per satu
# FEATURE_MODE = "block4_gmp"
# FEATURE_MODE = "block4_avg_max"
# FEATURE_MODE = "block5_gap"
# FEATURE_MODE = "block5_gmp"
# FEATURE_MODE = "block5_avg_max"
# FEATURE_MODE = "fc1"
# FEATURE_MODE = "fc2"

FEATURE_DIMENSIONS = {
    "block4_gap": 512,
    "block4_gmp": 512,
    "block4_avg_max": 1024,
    "block5_gap": 512,
    "block5_gmp": 512,
    "block5_avg_max": 1024,
    "fc1": 4096,
    "fc2": 4096,
}

FEATURE_DESCRIPTIONS = {
    "block4_gap": "Block4 + GlobalAveragePooling2D (GAP)",
    "block4_gmp": "Block4 + GlobalMaxPooling2D (GMP)",
    "block4_avg_max": "Block4 + GAP + GMP (Concatenate)",
    "block5_gap": "Block5 + GlobalAveragePooling2D (GAP)",
    "block5_gmp": "Block5 + GlobalMaxPooling2D (GMP)",
    "block5_avg_max": "Block5 + GAP + GMP (Concatenate)",
    "fc1": "VGG16 FC1",
    "fc2": "VGG16 FC2",
}


# ==============================================================
# MENGUMPULKAN DATASET
# ==============================================================

def collect_images(dataset_dir: Path):
    image_paths = []
    labels = []

    for class_index, class_name in enumerate(CLASS_ORDER):
        folder = dataset_dir / class_name

        if not folder.exists():
            print(f"[WARNING] Folder kelas tidak ditemukan: {folder}")
            continue

        paths = sorted([
            p for p in folder.rglob("*")
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
        ])

        for path in paths:
            image_paths.append(path)
            labels.append(class_index)

    return image_paths, np.array(labels, dtype=np.int64)


# ==============================================================
# MEMUAT VGG16 FEATURE EXTRACTOR
# ==============================================================

def load_feature_extractor():
    """
    Memuat VGG16 pretrained ImageNet sebagai feature extractor.

    Konfigurasi aktif:
        Block4 + GAP = 512 fitur.

    Alternatif:
        Block4 + GMP = 512 fitur.
        Block4 + GAP + GMP = 1024 fitur.
        Block5 + GAP = 512 fitur.
        Block5 + GMP = 512 fitur.
        Block5 + GAP + GMP = 1024 fitur.
        FC1 = 4096 fitur.
        FC2 = 4096 fitur.

    Bobot VGG16 tidak dilatih ulang atau di-fine-tuning.
    """

    if FEATURE_MODE not in FEATURE_DIMENSIONS:
        raise ValueError(
            f"Konfigurasi fitur tidak dikenal: {FEATURE_MODE}"
        )

    # ----------------------------------------------------------
    # EKSPERIMEN FC1 DAN FC2
    # ----------------------------------------------------------
    # FC1 dan FC2 menggunakan VGG16 dengan include_top=True.
    # Output FC1/FC2 memiliki 4096 fitur.

    if FEATURE_MODE in ("fc1", "fc2"):
        base_model = VGG16(
            weights="imagenet",
            include_top=True,
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        )

        output = base_model.get_layer(FEATURE_MODE).output

        feature_model = Model(
            inputs=base_model.input,
            outputs=output,
            name=f"vgg16_{FEATURE_MODE}",
        )

    else:
        # ------------------------------------------------------
        # EKSPERIMEN BLOCK4 DAN BLOCK5
        # ------------------------------------------------------

        base_model = VGG16(
            weights="imagenet",
            include_top=False,
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        )

        if FEATURE_MODE.startswith("block4"):
            pooling_layer = "block4_pool"
        else:
            pooling_layer = "block5_pool"

        x = base_model.get_layer(pooling_layer).output

        # ------------------------------------------------------
        # BLOCK4 + GAP (AKTIF)
        # ------------------------------------------------------
        if FEATURE_MODE == "block4_gap":
            x = GlobalAveragePooling2D(name="global_average_pooling2d")(x)

        # ------------------------------------------------------
        # ALTERNATIF: BLOCK4 + GMP
        # ------------------------------------------------------
        elif FEATURE_MODE == "block4_gmp":
            x = GlobalMaxPooling2D(name="global_max_pooling2d")(x)

        # ------------------------------------------------------
        # ALTERNATIF: BLOCK4 + AVG + MAX
        # ------------------------------------------------------
        elif FEATURE_MODE == "block4_avg_max":
            avg_features = GlobalAveragePooling2D(
                name="global_average_pooling2d"
            )(x)

            max_features = GlobalMaxPooling2D(
                name="global_max_pooling2d"
            )(x)

            x = Concatenate(name="average_max_concatenate")(
                [avg_features, max_features]
            )

        # ------------------------------------------------------
        # ALTERNATIF: BLOCK5 + GAP
        # ------------------------------------------------------
        elif FEATURE_MODE == "block5_gap":
            x = GlobalAveragePooling2D(name="global_average_pooling2d")(x)

        # ------------------------------------------------------
        # ALTERNATIF: BLOCK5 + GMP
        # ------------------------------------------------------
        elif FEATURE_MODE == "block5_gmp":
            x = GlobalMaxPooling2D(name="global_max_pooling2d")(x)

        # ------------------------------------------------------
        # ALTERNATIF: BLOCK5 + AVG + MAX
        # ------------------------------------------------------
        elif FEATURE_MODE == "block5_avg_max":
            avg_features = GlobalAveragePooling2D(
                name="global_average_pooling2d"
            )(x)

            max_features = GlobalMaxPooling2D(
                name="global_max_pooling2d"
            )(x)

            x = Concatenate(name="average_max_concatenate")(
                [avg_features, max_features]
            )

        feature_model = Model(
            inputs=base_model.input,
            outputs=x,
            name=f"vgg16_{FEATURE_MODE}",
        )

    # Membekukan seluruh layer dasar VGG16.
    for layer in base_model.layers:
        layer.trainable = False

    actual_dim = int(feature_model.output_shape[-1])
    expected_dim = FEATURE_DIMENSIONS[FEATURE_MODE]

    if actual_dim != expected_dim:
        raise ValueError(
            f"Dimensi fitur {FEATURE_MODE} tidak sesuai. "
            f"Diharapkan {expected_dim}, diperoleh {actual_dim}."
        )

    return feature_model


# ==============================================================
# INFORMASI ARSITEKTUR VGG16
# ==============================================================

def lihat_arsitektur_vgg16(feature_model, model_dir: Path):
    print("\n" + "=" * 70)
    print("ARSITEKTUR VGG16 SEBAGAI FEATURE EXTRACTOR")
    print("=" * 70)
    print(f"Konfigurasi  : {FEATURE_MODE}")
    print(f"Deskripsi    : {FEATURE_DESCRIPTIONS[FEATURE_MODE]}")
    print(f"Input shape  : {feature_model.input_shape}")
    print(f"Output shape : {feature_model.output_shape}")
    print("Preprocessing: VGG16 ImageNet preprocess_input")
    print("Fine-tuning  : Tidak dilakukan")
    print(f"Jumlah fitur : {feature_model.output_shape[-1]}")
    print("=" * 70)

    # Simpan summary arsitektur.
    summary_path = model_dir / "vgg16_summary.txt"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Konfigurasi: {FEATURE_MODE}\n")
        f.write(f"Deskripsi: {FEATURE_DESCRIPTIONS[FEATURE_MODE]}\n\n")

        feature_model.summary(
            print_fn=lambda line: f.write(line + "\n")
        )

    print(f"Summary VGG16 disimpan di: {summary_path}")

    # Simpan daftar layer ke CSV.
    rows = []

    for i, layer in enumerate(feature_model.layers):
        rows.append({
            "no": i + 1,
            "nama_layer": layer.name,
            "tipe_layer": layer.__class__.__name__,
            "output_shape": str(layer.output.shape),
            "jumlah_parameter": layer.count_params(),
            "trainable": layer.trainable,
        })

    df_layers = pd.DataFrame(rows)
    layer_csv_path = model_dir / "vgg16_layers.csv"

    df_layers.to_csv(layer_csv_path, index=False)

    print(f"Daftar layer VGG16 disimpan di: {layer_csv_path}")


# ==============================================================
# LOAD DAN PREPROCESS CITRA UNTUK VGG16
# ==============================================================

def load_and_preprocess_image(path: Path):
    """
    Tahapan:
    1. Koreksi orientasi EXIF.
    2. Konversi RGB.
    3. Crop dan resize.
    4. Konversi ke float32.
    5. Preprocessing resmi VGG16 ImageNet.

    preprocess_input mengubah RGB menjadi BGR dan mengurangi
    mean ImageNet. Piksel tidak dibagi dengan 255.
    """

    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")

        img = ImageOps.fit(
            img,
            IMG_SIZE,
            method=Image.Resampling.LANCZOS,
        )

        arr = np.asarray(img, dtype=np.float32)

    arr = preprocess_input(arr)

    return arr


# ==============================================================
# EKSTRAKSI FITUR VGG16
# ==============================================================

def extract_features(
    feature_model,
    image_paths,
    batch_size=BATCH_SIZE,
):
    if batch_size < 1:
        raise ValueError("Batch size harus lebih besar dari nol.")

    if len(image_paths) == 0:
        raise ValueError("Tidak ada citra untuk diekstraksi.")

    all_features = []

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start:start + batch_size]

        batch_images = np.stack([
            load_and_preprocess_image(path)
            for path in batch_paths
        ], axis=0)

        features = feature_model.predict(
            batch_images,
            verbose=0,
        )

        all_features.append(features.astype(np.float32))

        done = min(start + batch_size, len(image_paths))

        print(
            f"Ekstraksi fitur VGG16 ({FEATURE_MODE}): "
            f"{done}/{len(image_paths)} citra"
        )

    return np.vstack(all_features).astype(np.float32)


# ==============================================================
# MENYIMPAN INDEKS KELAS
# ==============================================================

def save_class_indices(model_dir: Path):
    class_indices = {}

    for i, class_name in enumerate(CLASS_ORDER):
        class_indices[str(i)] = {
            "folder_name": class_name,
            "display_name": DISPLAY_NAMES.get(
                class_name,
                class_name,
            ),
        }

    output_path = model_dir / CLASS_INDEX_FILENAME

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            class_indices,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Class indices disimpan di: {output_path}")


# ==============================================================
# MENYIMPAN HASIL EKSTRAKSI FITUR
# ==============================================================

def save_vgg16_features(
    model_dir: Path,
    features,
    labels,
    image_paths,
):
    output_path = model_dir / VGG16_FEATURES_FILENAME

    np.savez_compressed(
        output_path,
        features=features,
        labels=labels,
        paths=np.array(
            [str(p) for p in image_paths],
            dtype=object,
        ),
    )

    print(f"Fitur VGG16 disimpan di: {output_path}")


# ==============================================================
# MENAMPILKAN OUTPUT EKSTRAKSI FITUR
# ==============================================================

def lihat_output_ekstraksi_fitur(model_dir: Path):
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
    print(f"Konfigurasi       : {FEATURE_MODE}")
    print(f"Shape features    : {features.shape}")
    print(f"Shape labels      : {labels.shape}")
    print(f"Shape paths       : {paths.shape}")
    print("-" * 70)

    if len(paths) > 0:
        print(f"Contoh file pertama      : {paths[0]}")
        print(f"Label file pertama       : {labels[0]}")
        print(f"Jumlah fitur per citra   : {features.shape[1]}")
        print(f"10 fitur pertama citra 1 : {features[0][:10]}")

    print("=" * 70)

    # Membuat preview fitur dalam bentuk DataFrame.
    rows = []

    for i in range(len(paths)):
        image_path = Path(str(paths[i]))
        label_index = int(labels[i])
        class_folder = CLASS_ORDER[label_index]

        class_name = DISPLAY_NAMES.get(
            class_folder,
            class_folder,
        )

        row = {
            "no": i + 1,
            "nama_file": image_path.name,
            "kelas": class_name,
            "label": label_index,
            "path": str(image_path),
            "konfigurasi": FEATURE_MODE,
            "jumlah_fitur": features.shape[1],
            "mean_fitur": float(np.mean(features[i])),
            "std_fitur": float(np.std(features[i])),
            "min_fitur": float(np.min(features[i])),
            "max_fitur": float(np.max(features[i])),
            "euclidean_norm": float(np.linalg.norm(features[i])),
        }

        # Simpan maksimal 20 fitur pertama sebagai preview.
        for j in range(min(20, features.shape[1])):
            row[f"fitur_{j + 1:04d}"] = float(features[i][j])

        rows.append(row)

    df_preview = pd.DataFrame(rows)

    preview_csv_path = model_dir / "preview_fitur_vgg16.csv"

    df_preview.to_csv(preview_csv_path, index=False)

    print(f"Preview fitur disimpan di: {preview_csv_path}")


# ==============================================================
# RINGKASAN DATASET
# ==============================================================

def print_summary(labels):
    print("=" * 70)
    print("RINGKASAN DATASET")
    print("=" * 70)

    for i, class_name in enumerate(CLASS_ORDER):
        count = int(np.sum(labels == i))

        display_name = DISPLAY_NAMES.get(
            class_name,
            class_name,
        )

        print(f"{display_name:25s}: {count:4d} citra")

    print("-" * 70)
    print(f"Total: {len(labels)} citra")
    print("=" * 70)


# ==============================================================
# MAIN PROGRAM
# ==============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Ekstraksi fitur VGG16 pretrained ImageNet. "
            f"Konfigurasi: {FEATURE_MODE}; "
            f"dimensi: {FEATURE_DIMENSIONS[FEATURE_MODE]}."
        )
    )

    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="data/processed",
        help="Folder dataset hasil preprocessing.",
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        default="models",
        help="Folder output model dan fitur.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size ekstraksi fitur.",
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    model_dir = Path(args.model_dir)

    model_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    # KUMPULKAN DATASET
    # ----------------------------------------------------------

    image_paths, labels = collect_images(dataset_dir)

    if len(image_paths) == 0:
        raise FileNotFoundError(
            f"Tidak ada citra ditemukan di {dataset_dir.resolve()}."
        )

    if len(image_paths) != len(labels):
        raise ValueError(
            "Jumlah citra tidak sama dengan jumlah label."
        )

    print_summary(labels)

    # ----------------------------------------------------------
    # LOAD FEATURE EXTRACTOR
    # ----------------------------------------------------------

    print("\nMemuat VGG16 pretrained ImageNet...")
    print(f"Konfigurasi eksperimen: {FEATURE_MODE}")

    feature_model = load_feature_extractor()

    lihat_arsitektur_vgg16(
        feature_model,
        model_dir,
    )

    # ----------------------------------------------------------
    # EKSTRAKSI FITUR
    # ----------------------------------------------------------

    print(
        f"\nMulai ekstraksi fitur VGG16: "
        f"{FEATURE_DESCRIPTIONS[FEATURE_MODE]}..."
    )
    print("Preprocessing: VGG16 ImageNet preprocess_input")
    print("Pembagian piksel dengan 255: Tidak dilakukan")

    features = extract_features(
        feature_model=feature_model,
        image_paths=image_paths,
        batch_size=args.batch_size,
    )

    # ----------------------------------------------------------
    # VALIDASI DIMENSI FITUR
    # ----------------------------------------------------------

    expected_features = FEATURE_DIMENSIONS[FEATURE_MODE]

    if features.shape[1] != expected_features:
        raise ValueError(
            f"Dimensi fitur {FEATURE_MODE} tidak sesuai. "
            f"Diharapkan {expected_features}, "
            f"tetapi diperoleh {features.shape[1]}."
        )

    print("\n" + "=" * 70)
    print("VALIDASI HASIL EKSTRAKSI")
    print("=" * 70)
    print(f"Konfigurasi : {FEATURE_MODE}")
    print(f"Jumlah citra: {features.shape[0]}")
    print(f"Dimensi fitur: {features.shape[1]}")
    print(f"Tipe data   : {features.dtype}")
    print("=" * 70)

    # ----------------------------------------------------------
    # SIMPAN FITUR DAN LABEL
    # ----------------------------------------------------------

    save_vgg16_features(
        model_dir=model_dir,
        features=features,
        labels=labels,
        image_paths=image_paths,
    )

    # ----------------------------------------------------------
    # SIMPAN INDEKS KELAS
    # ----------------------------------------------------------

    save_class_indices(model_dir)

    # ----------------------------------------------------------
    # TAMPILKAN PREVIEW FITUR
    # ----------------------------------------------------------

    lihat_output_ekstraksi_fitur(model_dir)

    print("\n" + "=" * 70)
    print("EKSTRAKSI FITUR VGG16 SELESAI")
    print(f"Konfigurasi: {FEATURE_MODE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
