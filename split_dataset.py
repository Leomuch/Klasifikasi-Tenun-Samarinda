
import argparse
from pathlib import Path

import numpy as np

from config import CLASS_ORDER, DISPLAY_NAMES


def get_split_from_path(image_path):
    """Menentukan split berdasarkan folder train/test pada path."""
    parts = Path(
        str(image_path).replace("\\", "/")
    ).parts

    split_parts = [
        part.lower()
        for part in parts
        if part.lower() in ("train", "test")
    ]

    if len(split_parts) != 1:
        raise ValueError(
            f"Path harus memuat tepat satu folder train/test: "
            f"{image_path}"
        )

    return split_parts[0]


def print_distribution(labels, indices, subset_name):
    print(f"\n{subset_name} ({len(indices)} citra):")

    for class_index, class_name in enumerate(CLASS_ORDER):
        count = int(
            np.sum(labels[indices] == class_index)
        )
        display_name = DISPLAY_NAMES.get(
            class_name, class_name
        )
        print(f"  {display_name:<25}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Menyusun split fitur VGG16 berdasarkan folder "
            "train/test yang sudah ditetapkan."
        )
    )
    parser.add_argument(
        "--model_dir",
        default="models",
        help="Folder fitur dan model."
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    features_path = model_dir / "vgg16_features.npz"
    output_path = model_dir / "split_dataset.npz"

    if not features_path.exists():
        raise FileNotFoundError(
            f"{features_path} tidak ditemukan. "
            "Jalankan ekstraksi fitur VGG16 terlebih dahulu."
        )

    data = np.load(
        features_path,
        allow_pickle=True
    )

    features = data["features"]
    labels = data["labels"]
    paths = data["paths"]

    if not (
        len(features) == len(labels) == len(paths)
    ):
        raise ValueError(
            "Jumlah fitur, label, dan path tidak sama."
        )

    # Tentukan pembagian berdasarkan lokasi file,
    # bukan dengan pembagian acak.
    split_names = [
        get_split_from_path(path)
        for path in paths
    ]

    train_idx = np.array([
        i for i, split in enumerate(split_names)
        if split == "train"
    ], dtype=int)

    test_idx = np.array([
        i for i, split in enumerate(split_names)
        if split == "test"
    ], dtype=int)

    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError(
            "Train atau test kosong. Periksa struktur folder "
            "dataset hasil preprocessing."
        )

    # Pastikan tidak ada path yang masuk ke kedua subset.
    train_paths = {str(paths[i]) for i in train_idx}
    test_paths = {str(paths[i]) for i in test_idx}

    if train_paths.intersection(test_paths):
        raise ValueError(
            "Ditemukan path yang sama di train dan test."
        )

    X_train = features[train_idx]
    X_test = features[test_idx]
    y_train = labels[train_idx]
    y_test = labels[test_idx]
    paths_train = paths[train_idx]
    paths_test = paths[test_idx]

    print("=" * 70)
    print("SPLIT DATASET BERDASARKAN FOLDER")
    print("=" * 70)
    print("Metode: menggunakan folder train/test yang sudah tersedia")
    print("Random split: tidak digunakan")

    print_distribution(labels, train_idx, "DATA TRAIN")
    print_distribution(labels, test_idx, "DATA TEST")

    np.savez_compressed(
        output_path,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        paths_train=np.array(
            [str(p) for p in paths_train],
            dtype=object
        ),
        paths_test=np.array(
            [str(p) for p in paths_test],
            dtype=object
        )
    )

    print("\n" + "=" * 70)
    print("SPLIT SELESAI")
    print("=" * 70)
    print(f"Train: {len(X_train)} citra")
    print(f"Test : {len(X_test)} citra")
    print(f"Fitur: {X_train.shape[1]} dimensi")
    print(f"Output: {output_path.resolve()}")


if __name__ == "__main__":
    main()
