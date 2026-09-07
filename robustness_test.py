"""
PENGUJIAN ROBUSTNESS KLASIFIKASI (poin 8b).

Menguji ketahanan model terhadap gangguan pada citra uji:
rotasi, flip, blur, perubahan kecerahan, noise, dan penurunan resolusi.
Untuk tiap kondisi diukur akurasi klasifikasi pada SELURUH data uji,
lalu dibandingkan dengan kondisi asli (baseline).

Jalankan dari root repo (folder yang berisi config.py):
    python robustness_test.py

Prasyarat: sudah menjalankan split_dataset dan train.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageFilter, ImageEnhance

from tensorflow.keras.applications.vgg16 import preprocess_input

from config import IMG_SIZE
from vgg16 import load_feature_extractor
from classify_image import load_svm_model

MODEL_DIR = Path("models")
SPLIT_PATH = MODEL_DIR / "split_dataset.npz"


# ================================================================
# PREPROCESS SATU CITRA PIL -> ARRAY SIAP VGG16
# (mengikuti load_and_preprocess_image di vgg16.py)
# ================================================================

def preprocess_pil(img):
    img = img.convert("RGB")
    img = ImageOps.fit(
        img,
        IMG_SIZE,
        method=Image.Resampling.LANCZOS
    )
    arr = np.asarray(img, dtype=np.float32)
    arr = preprocess_input(arr)
    return arr


# ================================================================
# FUNGSI GANGGUAN
# ================================================================

def add_noise(img, sigma=15):
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    noise = np.random.normal(0, sigma, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def low_res(img, factor=4):
    w, h = img.size
    small = img.resize(
        (max(1, w // factor), max(1, h // factor)),
        Image.Resampling.BILINEAR
    )
    return small.resize((w, h), Image.Resampling.BILINEAR)


TRANSFORMS = {
    "Asli (baseline)":      lambda im: im,
    "Rotasi 90 derajat":    lambda im: im.rotate(90, expand=True),
    "Rotasi 180 derajat":   lambda im: im.rotate(180, expand=True),
    "Flip horizontal":      lambda im: ImageOps.mirror(im),
    "Gaussian blur":        lambda im: im.filter(ImageFilter.GaussianBlur(radius=2)),
    "Kecerahan +30%":       lambda im: ImageEnhance.Brightness(im).enhance(1.3),
    "Kecerahan -30%":       lambda im: ImageEnhance.Brightness(im).enhance(0.7),
    "Gaussian noise":       lambda im: add_noise(im, sigma=15),
    "Resolusi rendah (1/4)": lambda im: low_res(im, factor=4),
}


def main():
    # seed agar noise reproducible
    np.random.seed(50)

    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"{SPLIT_PATH} tidak ada. Jalankan split_dataset dulu."
        )

    feature_model = load_feature_extractor()
    svm_model = load_svm_model(MODEL_DIR)

    split = np.load(SPLIT_PATH, allow_pickle=True)
    paths_test = [Path(p) for p in split["paths_test"]]
    y_test = split["y_test"].astype(int)

    print(f"Jumlah citra uji : {len(paths_test)}")
    print(f"Dimensi fitur    : {feature_model.output_shape[-1]}")
    print("Memuat citra uji ke memori...\n")

    # muat semua citra asli sekali (sudah exif-transpose + RGB)
    base_images = []
    for p in paths_test:
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im)
            base_images.append(im.convert("RGB").copy())

    rows = []
    baseline_acc = None

    for name, fn in TRANSFORMS.items():

        # bangun batch fitur untuk seluruh data uji pada transformasi ini
        batch = np.stack(
            [preprocess_pil(fn(im)) for im in base_images],
            axis=0
        )

        feats = feature_model.predict(batch, verbose=0)
        preds = svm_model.predict(feats)

        benar = int(np.sum(preds == y_test))
        acc = benar / len(y_test)

        if baseline_acc is None:
            baseline_acc = acc

        rows.append({
            "Transformasi": name,
            "Akurasi (%)": round(acc * 100, 2),
            "Benar": benar,
            "Total": len(y_test),
            "Selisih vs baseline (%)": round((acc - baseline_acc) * 100, 2),
        })

        print(f"{name:24s}: {acc*100:6.2f}%  ({benar}/{len(y_test)})")

    df = pd.DataFrame(rows)

    print("\n" + "=" * 70)
    print("HASIL PENGUJIAN ROBUSTNESS (AKURASI KLASIFIKASI)")
    print("=" * 70)
    print(df.to_string(index=False))
    print("=" * 70)

    out = MODEL_DIR / "robustness_report.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"\nTabel disimpan di: {out}")

    print("\nCatatan analisis:")
    print("- VGG16 tidak invarian rotasi, jadi akurasi rotasi biasanya turun.")
    print("- Penurunan pada gangguan tertentu itu WAJAR dan justru bahan analisis.")


if __name__ == "__main__":
    main()
