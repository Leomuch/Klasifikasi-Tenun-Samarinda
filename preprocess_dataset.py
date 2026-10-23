
import argparse
from pathlib import Path

from PIL import Image, ImageOps

from config import CLASS_ORDER, IMAGE_EXTENSIONS, IMG_SIZE


def preprocess_image(input_path: Path, output_path: Path):
    try:
        with Image.open(input_path) as img:
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")

            img = ImageOps.fit(
                img,
                IMG_SIZE,
                method=Image.Resampling.LANCZOS
            )

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )
            img.save(output_path, format="JPEG", quality=95)

        return True, None

    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocessing dataset train dan test."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/raw"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    total_ok = 0
    total_error = 0

    print(f"Input : {input_dir.resolve()}")
    print(f"Output: {output_dir.resolve()}")
    print(f"Size  : {IMG_SIZE[0]} x {IMG_SIZE[1]}")

    for class_name in CLASS_ORDER:
        for split_name in ("train", "test"):
            class_input_dir = (
                input_dir / class_name / split_name
            )
            class_output_dir = (
                output_dir / class_name / split_name
            )

            if not class_input_dir.is_dir():
                print(f"[SKIP] Folder tidak ditemukan: {class_input_dir}")
                continue

            image_paths = sorted(
                p for p in class_input_dir.rglob("*")
                if p.is_file()
                and p.suffix.lower() in IMAGE_EXTENSIONS
            )

            split_ok = 0
            split_error = 0

            for image_path in image_paths:
                relative_path = image_path.relative_to(
                    class_input_dir
                )
                output_path = (
                    class_output_dir / relative_path
                ).with_suffix(".jpg")

                ok, error = preprocess_image(
                    image_path,
                    output_path
                )

                if ok:
                    total_ok += 1
                    split_ok += 1
                else:
                    total_error += 1
                    split_error += 1
                    print(f"[ERROR] {image_path}: {error}")

            print(
                f"{class_name}/{split_name}: "
                f"input={len(image_paths)}, "
                f"berhasil={split_ok}, "
                f"error={split_error}"
            )

    print("\nPREPROCESSING SELESAI")
    print(f"Total berhasil: {total_ok}")
    print(f"Total error   : {total_error}")


if __name__ == "__main__":
    main()