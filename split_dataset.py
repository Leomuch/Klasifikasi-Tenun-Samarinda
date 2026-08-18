"""
Membagi dataset fitur VGG16 menjadi data latih dan data uji menggunakan
STRATIFIED TRAIN-TEST SPLIT dengan proporsi 60:40.

Pembagian dilakukan secara acak tetapi tetap mempertahankan proporsi
setiap kelas pada data latih dan data uji.

Total:
- 180 citra latih (60%)
- 120 citra uji (40%)

Output split_dataset.npz kompatibel dengan retrieval_db.py dan
evaluate_retrieval.py dengan key:
X_train, X_test, y_train, y_test, paths_train, paths_test.
"""

import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from config import CLASS_ORDER, DISPLAY_NAMES


def print_distribution(labels, idx, subset_name):
    """Menampilkan distribusi kelas pada subset."""
    print(f"  {subset_name} (total {len(idx)} citra):")

    for cls_i, cls_name in enumerate(CLASS_ORDER):
        n = int(np.sum(labels[idx] == cls_i))
        disp = DISPLAY_NAMES.get(cls_name, cls_name)
        print(f"    - {disp:<20} : {n}")


def main():
    parser = argparse.ArgumentParser(
        description="Stratified Train-Test Split dataset fitur VGG16 (70:30)."
    )

    parser.add_argument(
        "--model_dir",
        default="models",
        help="Folder berisi vgg16_features.npz dan output split."
    )

    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    features_path = model_dir / "vgg16_features.npz"
    output_path = model_dir / "split_dataset.npz"

    if not features_path.exists():
        raise FileNotFoundError(
            f"{features_path} tidak ditemukan. "
            f"Jalankan vgg16.py terlebih dahulu."
        )

    print("=" * 60)
    print("STRATIFIED TRAIN-TEST SPLIT 60:40")
    print("=" * 60)

    # ---------------------------------------------------------
    # Load dataset fitur VGG16
    # ---------------------------------------------------------
    data = np.load(features_path, allow_pickle=True)

    features = data["features"]
    labels = data["labels"]
    paths = data["paths"]

    print(f"Memuat {len(features)} citra dari {features_path.name}")

    # ---------------------------------------------------------
    # Membuat indeks seluruh data
    # ---------------------------------------------------------
    indices = np.arange(len(features))

    # ---------------------------------------------------------
    # Stratified split 70:30
    # ---------------------------------------------------------
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.40,
        train_size=0.60,
        stratify=labels,
        random_state=42,
        shuffle=True
    )

    # ---------------------------------------------------------
    # Ambil data berdasarkan indeks
    # ---------------------------------------------------------
    X_train = features[train_idx]
    X_test = features[test_idx]

    y_train = labels[train_idx]
    y_test = labels[test_idx]

    paths_train = paths[train_idx]
    paths_test = paths[test_idx]

    # ---------------------------------------------------------
    # Tampilkan distribusi
    # ---------------------------------------------------------
    print("\nHasil pembagian:")

    print_distribution(
        labels,
        train_idx,
        "Data latih"
    )

    print_distribution(
        labels,
        test_idx,
        "Data uji"
    )

    # ---------------------------------------------------------
    # Simpan hasil split
    # ---------------------------------------------------------
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
        ),
    )

    print(f"\nSplit disimpan ke: {output_path}")
    print(f"Total: {len(X_train)} latih / {len(X_test)} uji")
    print("SPLIT SELESAI.")


if __name__ == "__main__":
    main()