import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tensorflow.keras.applications.vgg16 import (VGG16, preprocess_input)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from config import (CLASS_ORDER, DISPLAY_NAMES, IMAGE_EXTENSIONS, IMG_SIZE, BATCH_SIZE, VGG16_FEATURES_FILENAME, CLASS_INDEX_FILENAME)

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
            if (
                p.is_file()
                and p.suffix.lower() in IMAGE_EXTENSIONS
            )
        ])
        for path in paths:
            image_paths.append(path)
            labels.append(class_index)
    return (
        image_paths,
        np.array(labels, dtype=np.int64)
    )

# ==============================================================
# MEMUAT VGG16 FEATURE EXTRACTOR
# ==============================================================
def load_feature_extractor():
    """
    Memuat VGG16 pretrained ImageNet sebagai feature extractor.
    Konfigurasi aktif:
        block4_pool + GlobalAveragePooling2D
    Ukuran input:
        224 x 224 x 3
    Ukuran output:
        512 fitur per citra
    Bobot VGG16 tidak dilatih ulang atau di-fine-tuning.
    """
    base_model = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(
            IMG_SIZE[0],
            IMG_SIZE[1],
            3
        )
    )
    # Mengambil output block4_pool
    x = base_model.get_layer(
        "block4_pool"
    ).output
    # Mengubah feature map menjadi vektor 512 fitur
    x = GlobalAveragePooling2D()(x)
    feature_model = Model(
        inputs=base_model.input,
        outputs=x,
        name="vgg16_block4_gap"
    )
    # Membekukan seluruh layer VGG16
    for layer in base_model.layers:
        layer.trainable = False
    return feature_model
# ==============================================================
# INFORMASI ARSITEKTUR VGG16
# ==============================================================
def lihat_arsitektur_vgg16(
    feature_model,
    model_dir: Path
):
    """
    Menampilkan dan menyimpan informasi arsitektur
    VGG16 yang digunakan sebagai feature extractor.
    """
    print("\n" + "=" * 70)
    print("ARSITEKTUR VGG16 SEBAGAI FEATURE EXTRACTOR")
    print("=" * 70)
    print(f"Input shape  : {feature_model.input_shape}")
    print("Output layer : block4_pool + GlobalAveragePooling2D")
    print(f"Output shape : {feature_model.output_shape}")
    print("Preprocessing: VGG16 ImageNet preprocess_input")
    print("Normalisasi  : BGR + pengurangan mean ImageNet")
    print("Fine-tuning  : Tidak dilakukan")
    print(f"Jumlah fitur : {feature_model.output_shape[-1]}")
    print("=" * 70)

    # Menyimpan summary arsitektur
    summary_path = model_dir / "vgg16_summary.txt"
    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:
        feature_model.summary(
            print_fn=lambda x: f.write(x + "\n")
        )
    print(f"Summary VGG16 disimpan di: {summary_path}")

    # Menyimpan daftar layer ke CSV
    rows = []
    for i, layer in enumerate(feature_model.layers):
        rows.append({
            "no": i + 1,
            "nama_layer": layer.name,
            "tipe_layer": layer.__class__.__name__,
            "output_shape": str(layer.output.shape),
            "jumlah_parameter": layer.count_params(),
            "trainable": layer.trainable
        })
    df_layers = pd.DataFrame(rows)
    layer_csv_path = model_dir / "vgg16_layers.csv"
    df_layers.to_csv(
        layer_csv_path,
        index=False
    )
    print(f"Daftar layer VGG16 disimpan di: {layer_csv_path}")
# ==============================================================
# LOAD DAN PREPROCESS CITRA UNTUK VGG16
# ==============================================================
def load_and_preprocess_image(path: Path):
    """
    Memuat satu citra dan menyiapkannya untuk VGG16 ImageNet.
    Tahapan:
    1. Koreksi orientasi EXIF.
    2. Konversi RGB.
    3. Crop dan resize.
    4. Konversi menjadi array float32.
    5. Preprocessing resmi VGG16 ImageNet.
    preprocess_input VGG16:
    - Mengubah urutan kanal RGB menjadi BGR.
    - Mengurangi mean ImageNet pada setiap kanal.
    - Tidak membagi piksel dengan 255.
    """
    with Image.open(path) as img:
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
    # Preprocessing resmi VGG16 ImageNet
    # Jangan menggunakan arr / 255.0
    arr = preprocess_input(arr)
    return arr
# ==============================================================
# EKSTRAKSI FITUR VGG16
# ==============================================================
def extract_features(
    feature_model,
    image_paths,
    batch_size=BATCH_SIZE
):
    """
    Mengekstraksi fitur seluruh citra secara bertahap
    menggunakan batch untuk menghemat penggunaan memori.
    """
    if batch_size < 1:
        raise ValueError(
            "Batch size harus lebih besar dari nol."
        )
    if len(image_paths) == 0:
        raise ValueError(
            "Tidak ada citra untuk diekstraksi."
        )
    all_features = []
    for start in range(
        0,
        len(image_paths),
        batch_size
    ):
        batch_paths = image_paths[
            start:start + batch_size
        ]
        batch_images = np.stack([
            load_and_preprocess_image(path)
            for path in batch_paths
        ], axis=0)
        features = feature_model.predict(
            batch_images,
            verbose=0
        )
        all_features.append(
            features.astype(np.float32)
        )
        done = min(
            start + batch_size,
            len(image_paths)
        )
        print(
            f"Ekstraksi fitur VGG16: "
            f"{done}/{len(image_paths)} citra"
        )

    features = np.vstack(
        all_features
    ).astype(np.float32)

    return features

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
                class_name
            )
        }
    output_path = (
        model_dir /
        CLASS_INDEX_FILENAME
    )
    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            class_indices,
            f,
            indent=2,
            ensure_ascii=False
        )
    print(f"Class indices disimpan di: {output_path}")

# ==============================================================
# MENYIMPAN HASIL EKSTRAKSI FITUR
# ==============================================================
def save_vgg16_features(
    model_dir: Path,
    features,
    labels,
    image_paths
):
    """
    Menyimpan fitur VGG16, label kelas, dan path citra.
    File digunakan untuk:
    - Pembagian data training dan testing.
    - Pelatihan SVM.
    - Pembangunan basis data CBIR.
    """
    output_path = (
        model_dir /
        VGG16_FEATURES_FILENAME
    )

    np.savez_compressed(
        output_path,
        features=features,
        labels=labels,
        paths=np.array(
            [str(p) for p in image_paths],
            dtype=object
        )
    )
    print(f"Fitur VGG16 disimpan di: {output_path}")

# ==============================================================
# MENAMPILKAN OUTPUT EKSTRAKSI FITUR
# ==============================================================
def lihat_output_ekstraksi_fitur(
    model_dir: Path
):
    """
    Menampilkan informasi hasil ekstraksi fitur
    dan menyimpan preview fitur ke CSV.
    """
    feature_path = (
        model_dir /
        VGG16_FEATURES_FILENAME
    )

    if not feature_path.exists():
        print(f"File fitur tidak ditemukan: {feature_path}")
        return

    data = np.load(
        feature_path,
        allow_pickle=True
    )

    features = data["features"]
    labels = data["labels"]
    paths = data["paths"]
    print("\n" + "=" * 70)
    print("OUTPUT EKSTRAKSI FITUR VGG16")
    print("=" * 70)
    print(f"Shape features   : {features.shape}")
    print(f"Shape labels     : {labels.shape}")
    print(f"Shape paths      : {paths.shape}")
    print("-" * 70)
    if len(paths) > 0:
        print(f"Contoh file pertama      : {paths[0]}")
        print(f"Label file pertama       : {labels[0]}")
        print(f"Jumlah fitur per citra   : {features.shape[1]}")
        print(f"10 fitur pertama citra 1 : {features[0][:10]}")
    print("=" * 70)

    # Membuat preview fitur dalam bentuk dataframe
    rows = []
    for i in range(len(paths)):
        image_path = Path(
            str(paths[i])
        )
        label_index = int(
            labels[i]
        )
        class_folder = CLASS_ORDER[
            label_index
        ]
        class_name = DISPLAY_NAMES.get(
            class_folder,
            class_folder
        )
        row = {
            "no": i + 1,
            "nama_file": image_path.name,
            "kelas": class_name,
            "label": label_index,
            "path": str(image_path),
            "jumlah_fitur": features.shape[1],
            "mean_fitur": float(
                np.mean(features[i])
            ),
            "std_fitur": float(
                np.std(features[i])
            ),
            "min_fitur": float(
                np.min(features[i])
            ),
            "max_fitur": float(
                np.max(features[i])
            ),
            "euclidean_norm": float(
                np.linalg.norm(features[i])
            )
        }
        # Menampilkan 20 fitur pertama sebagai preview
        for j in range(
            min(20, features.shape[1])
        ):
            row[
                f"fitur_{j + 1:04d}"
            ] = float(
                features[i][j]
            )
        rows.append(row)

    df_preview = pd.DataFrame(rows)
    preview_csv_path = (
        model_dir /
        "preview_fitur_vgg16.csv"
    )

    df_preview.to_csv(
        preview_csv_path,
        index=False
    )
    print(
        f"Preview dataframe fitur disimpan di: "
        f"{preview_csv_path}"
    )
# ==============================================================
# RINGKASAN DATASET
# ==============================================================
def print_summary(labels):
    print("=" * 70)
    print("RINGKASAN FITUR DATASET")
    print("=" * 70)
    for i, class_name in enumerate(CLASS_ORDER):
        count = int(
            np.sum(labels == i)
        )

        display_name = DISPLAY_NAMES.get(
            class_name,
            class_name
        )
    print(f"{display_name:20s}: {count:4d} citra")
    print("-" * 70)
    print(f"Total: {len(labels)} citra")
    print("=" * 70)
# ==============================================================
# MAIN PROGRAM
# ==============================================================
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Ekstraksi fitur VGG16 pretrained ImageNet "
            "menggunakan block4_pool + GAP, "
            "menghasilkan 512 fitur per citra."
        )
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
    dataset_dir = Path(
        args.dataset_dir
    )
    model_dir = Path(
        args.model_dir
    )
    model_dir.mkdir(
        parents=True,
        exist_ok=True
    )
    # ----------------------------------------------------------
    # KUMPULKAN DATASET
    # ----------------------------------------------------------
    image_paths, labels = collect_images(
        dataset_dir
    )
    if len(image_paths) == 0:
        raise FileNotFoundError(
            f"Tidak ada citra ditemukan di "
            f"{dataset_dir.resolve()}."
        )
    if len(image_paths) != len(labels):
        raise ValueError("Jumlah citra tidak sama dengan jumlah label.")
    print_summary(labels)
    # ----------------------------------------------------------
    # LOAD FEATURE EXTRACTOR
    # ----------------------------------------------------------
    print("\nMemuat VGG16 pretrained ImageNet...")
    feature_model = load_feature_extractor()
    lihat_arsitektur_vgg16(
        feature_model,
        model_dir
    )
    # ----------------------------------------------------------
    # EKSTRAKSI FITUR
    # ----------------------------------------------------------
    print("\nMulai ekstraksi fitur VGG16 Block4 + GAP...")
    print("Preprocessing: VGG16 ImageNet preprocess_input")
    print("Pembagian piksel dengan 255: Tidak dilakukan")
    features = extract_features(
        feature_model=feature_model,
        image_paths=image_paths,
        batch_size=args.batch_size
    )
    # ----------------------------------------------------------
    # VALIDASI DIMENSI FITUR
    # ----------------------------------------------------------
    expected_features = 512
    if features.shape[1] != expected_features:
        raise ValueError(
            f"Dimensi fitur tidak sesuai. "
            f"Diharapkan {expected_features}, "
            f"tetapi diperoleh {features.shape[1]}."
        )
    print("\n" + "=" * 70)
    print("VALIDASI HASIL EKSTRAKSI")
    print("=" * 70)
    print(f"Jumlah citra : {features.shape[0]}")
    print(f"Dimensi fitur: {features.shape[1]}")
    print(f"Tipe data    : {features.dtype}")
    print("=" * 70)

    # ----------------------------------------------------------
    # SIMPAN FITUR DAN LABEL
    # ----------------------------------------------------------
    save_vgg16_features(
        model_dir=model_dir,
        features=features,
        labels=labels,
        image_paths=image_paths
    )
    # ----------------------------------------------------------
    # SIMPAN INDEKS KELAS
    # ----------------------------------------------------------
    save_class_indices(
        model_dir
    )
    # ----------------------------------------------------------
    # TAMPILKAN PREVIEW FITUR
    # ----------------------------------------------------------
    lihat_output_ekstraksi_fitur(
        model_dir
    )

    print("\n" + "=" * 70)
    print("EKSTRAKSI FITUR VGG16 SELESAI")
    print("=" * 70)

if __name__ == "__main__":

    main()