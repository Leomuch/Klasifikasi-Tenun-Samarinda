import argparse
from pathlib import Path

from PIL import Image, ImageOps

from config import CLASS_ORDER, IMAGE_EXTENSIONS, IMG_SIZE

def preprocess_image(input_path: Path, output_path: Path):
    try:
        with Image.open(input_path) as img:
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")

            # Crop proporsional agar tidak gepeng, lalu resize ke ukuran VGG16
            img = ImageOps.fit(
                img,
                IMG_SIZE,
                method=Image.Resampling.LANCZOS
            )

            output_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(output_path, format="JPEG", quality=95)

        return True, None

    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocessing dataset: resize, RGB, dan standardisasi format citra."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/raw",
        help="Folder dataset mentah."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Folder dataset hasil preprocessing."
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    print("=" * 70)
    print("PREPROCESSING DATASET")
    print("=" * 70)
    print(f"Input : {input_dir.resolve()}")
    print(f"Output: {output_dir.resolve()}")
    print(f"Size  : {IMG_SIZE[0]} x {IMG_SIZE[1]}")
    print("=" * 70)

    total_ok = 0
    total_error = 0

    for class_name in CLASS_ORDER:
        class_input_dir = input_dir / class_name
        class_output_dir = output_dir / class_name

        if not class_input_dir.exists():
            print(f"[SKIP] Folder tidak ditemukan: {class_input_dir}")
            continue

        image_paths = sorted([
            p for p in class_input_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ])

        print(f"\nKelas: {class_name}")
        print(f"Jumlah citra mentah: {len(image_paths)}")

        for idx, image_path in enumerate(image_paths, start=1):
            # output_name = f"{class_name.lower()}_{idx:03d}.jpg"
            output_name = image_path.stem + ".jpg"
            output_path = class_output_dir / output_name

            ok, error = preprocess_image(image_path, output_path)

            if ok:
                total_ok += 1
            else:
                total_error += 1
                print(f"[ERROR] {image_path} -> {error}")

        print(f"Selesai preprocessing kelas {class_name}")

    print("\n" + "=" * 70)
    print("PREPROCESSING SELESAI")
    print("=" * 70)
    print(f"Berhasil: {total_ok} citra")
    print(f"Error   : {total_error} citra")


if __name__ == "__main__":
    main()