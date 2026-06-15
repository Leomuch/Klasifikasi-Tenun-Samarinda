import argparse
from pathlib import Path

from config import CLASS_ORDER, IMAGE_EXTENSIONS, DISPLAY_NAMES


def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0

    return sum(
        1 for p in folder.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )


def main():
    parser = argparse.ArgumentParser(
        description="Cek struktur dan jumlah dataset motif Tenun Samarinda."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="data/raw",
        help="Folder dataset utama."
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)

    print("=" * 60)
    print("VALIDASI DATASET")
    print("=" * 60)
    print(f"Dataset dir: {dataset_dir.resolve()}")
    print("=" * 60)

    total = 0
    class_counts = {}

    for class_name in CLASS_ORDER:
        folder = dataset_dir / class_name
        jumlah = count_images(folder)
        class_counts[class_name] = jumlah
        total += jumlah

        display_name = DISPLAY_NAMES.get(class_name, class_name)
        print(f"{display_name:20s}: {jumlah:4d} citra")

    print("=" * 60)
    print(f"Total citra: {total}")

    available_classes = [
        class_name for class_name, jumlah in class_counts.items()
        if jumlah > 0
    ]

    if len(available_classes) == 0:
        print("\nStatus:")
        print("Dataset masih kosong.")
        print("Masukkan citra ke folder data/raw/<nama_kelas>.")

    elif len(available_classes) == 1:
        print("\nStatus:")
        print("Baru tersedia 1 kelas.")
        print("Training SVM multi-kelas belum bisa dilakukan.")
        print("Namun ekstraksi fitur VGG16 dan feature database CBIR tetap bisa dibuat.")

    else:
        min_count = min(
            jumlah for jumlah in class_counts.values()
            if jumlah > 0
        )

        if min_count < 2:
            print("\nStatus:")
            print("Ada kelas yang jumlah datanya kurang dari 2 citra.")
            print("Tambahkan data agar train/test split dapat berjalan.")

        else:
            print("\nStatus:")
            print("Dataset sudah bisa digunakan untuk training awal.")

            missing_classes = [
                class_name for class_name, jumlah in class_counts.items()
                if jumlah == 0
            ]

            if missing_classes:
                print("Namun kelas berikut belum tersedia:")
                for class_name in missing_classes:
                    print(f"- {DISPLAY_NAMES.get(class_name, class_name)}")
            else:
                print("Semua kelas sudah tersedia.")

    print("\nTarget ideal sesuai proposal:")
    print("- Hatta          : 100 citra")
    print("- Pucuk Rebung  : 100 citra")
    print("- Cumi          : 100 citra")
    print("- Catur         : 100 citra")
    print("- Total         : 400 citra")


if __name__ == "__main__":
    main()