import argparse
from pathlib import Path

import numpy as np

from sklearn.model_selection import train_test_split

from config import RANDOM_STATE, TEST_SIZE, CLASS_ORDER, DISPLAY_NAMES


FEATURES_FILENAME = "vgg16_features.npz"
SPLIT_FILENAME = "split_dataset.npz"


from pathlib import Path

def is_train_image(path):
    filename = Path(str(path)).stem.lower()

    # HATTA
    if filename.startswith("hatta_"):
        num = int(filename.split("_")[1])
        return num <= 50

    # CUMI
    if filename.startswith("cumi_"):
        num = int(filename.split("_")[1])
        return num <= 50

    # REBUNG
    if filename.startswith("rebung_"):
        num = int(filename.split("_")[1])
        return num <= 59

    raise ValueError(f"File tidak dikenali: {filename}")

def main():
    parser = argparse.ArgumentParser(
        description="Membagi dataset fitur VGG16 menjadi data latih dan data uji."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models",
        help="Folder model yang berisi features_dataset.npz."
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=TEST_SIZE,
        help="Proporsi data uji."
    )

    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    feature_path = model_dir / FEATURES_FILENAME

    if not feature_path.exists():
        raise FileNotFoundError(
            f"File fitur tidak ditemukan: {feature_path}. "
            "Jalankan vgg16.py terlebih dahulu."
        )

    data = np.load(feature_path, allow_pickle=True)

    features = data["features"]
    labels = data["labels"]
    paths = data["paths"]

    unique_labels, counts = np.unique(labels, return_counts=True)

    if len(unique_labels) < 2:
        raise ValueError("Data splitting untuk training SVM membutuhkan minimal 2 kelas.")

    if counts.min() < 2:
        raise ValueError("Setiap kelas minimal membutuhkan 2 citra untuk train/test split.")

    # X_train, X_test, y_train, y_test, paths_train, paths_test = train_test_split(
    #     features,
    #     labels,
    #     paths,
    #     test_size=args.test_size,
    #     random_state=RANDOM_STATE,
    #     stratify=labels
    # )

    train_idx = []
    test_idx = []

    for i, path in enumerate(paths):

        if is_train_image(path):
            train_idx.append(i)
        else:
            test_idx.append(i)

    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)

    X_train = features[train_idx]
    X_test = features[test_idx]

    y_train = labels[train_idx]
    y_test = labels[test_idx]

    paths_train = paths[train_idx]
    paths_test = paths[test_idx]

    output_path = model_dir / SPLIT_FILENAME

    print("\nMODE SPLIT:")
    print("Hatta 001-050  -> Train")
    print("Hatta 051-100  -> Test")
    print("Cumi 001-050   -> Train")
    print("Cumi 051-100   -> Test")
    print("Rebung 001-059 -> Train")
    print("Rebung 060-100 -> Test")

    np.savez_compressed(
        output_path,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        paths_train=paths_train,
        paths_test=paths_test
    )

    print("=" * 70)
    print("DATA SPLITTING SELESAI")
    print("=" * 70)
    print(f"File split disimpan di: {output_path}")
    print(f"Train size: {len(y_train)}")
    print(f"Test size : {len(y_test)}")

    print("\nDistribusi kelas:")
    for i, class_name in enumerate(CLASS_ORDER):
        train_count = int(np.sum(y_train == i))
        test_count = int(np.sum(y_test == i))
        display_name = DISPLAY_NAMES.get(class_name, class_name)
        print(f"{display_name:20s}: train={train_count:4d}, test={test_count:4d}")


if __name__ == "__main__":
    main()