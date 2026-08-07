"""
split_dataset.py
================
Membagi dataset fitur VGG16 menjadi data latih dan data uji menggunakan
GROUP-BASED SPLIT berdasarkan sumber kain (sarung).

Pembagian ini bersifat DETERMINISTIK dan mengikuti struktur sarung pada
skripsi. Setiap sarung diperlakukan sebagai satu grup utuh: seluruh citra
dari satu sarung hanya masuk ke salah satu subset (latih ATAU uji),
sehingga tidak terjadi data leakage.

  Hatta        : Sarong A 001-050 -> latih, Sarong B 051-100 -> uji
  Pucuk_Rebung : Sarong A 001-030 -> latih, Sarong B 031-059 -> latih,
                 Sarong C 060-100 -> uji
  Cumi         : Sarong A 001-050 -> latih, Sarong B 051-100 -> uji

Total: 159 citra latih / 141 citra uji.

Karena batas antar-subset jatuh TEPAT di batas antar-sarung, tidak ada
satu sarung pun yang potongannya tersebar ke latih dan uji sekaligus.
Sifat inilah yang membuat pembagian ini sah disebut Group-Based Split.

Catatan: RANDOM_STATE dan TEST_SIZE pada config.py TIDAK digunakan di sini
karena pembagian mengikuti struktur sarung yang tetap (bukan acak).

Output split_dataset.npz kompatibel dengan retrieval_db.py dan
evaluate_retrieval.py (key: X_train, X_test, y_train, y_test,
paths_train, paths_test).
"""

import argparse
import re
from pathlib import Path

import numpy as np

from config import CLASS_ORDER, DISPLAY_NAMES


# ---------------------------------------------------------------------------
# Definisi sarung per kelas (rentang nomor file inklusif) + subset tujuan.
# Ubah di sini bila struktur sarung berubah.
# ---------------------------------------------------------------------------
SARONG_GROUPS = {
    "Hatta": [
        {"sarung": "A", "range": (1, 50),   "subset": "train"},
        {"sarung": "B", "range": (51, 100), "subset": "test"},
    ],
    "Pucuk_Rebung": [
        {"sarung": "A", "range": (1, 30),   "subset": "train"},
        {"sarung": "B", "range": (31, 59),  "subset": "train"},
        {"sarung": "C", "range": (60, 100), "subset": "test"},
    ],
    "Cumi": [
        {"sarung": "A", "range": (1, 50),   "subset": "train"},
        {"sarung": "B", "range": (51, 100), "subset": "test"},
    ],
}


def parse_index(path):
    """Ambil nomor urut citra dari nama file (kelompok digit terakhir)."""
    stem = Path(path).stem
    nums = re.findall(r"\d+", stem)
    if not nums:
        raise ValueError(f"Tidak ada nomor pada nama file: {path}")
    return int(nums[-1])


def resolve_group(class_name, index, path):
    """
    Tentukan (sarung, subset) untuk sebuah citra berdasarkan kelas & nomornya.
    Raise bila nomor tidak masuk rentang sarung mana pun.
    """
    for g in SARONG_GROUPS[class_name]:
        lo, hi = g["range"]
        if lo <= index <= hi:
            return f"{class_name}_{g['sarung']}", g["subset"]
    raise ValueError(
        f"Nomor {index} pada '{path}' (kelas {class_name}) tidak masuk "
        f"rentang sarung mana pun. Periksa penamaan file / SARONG_GROUPS."
    )


def split_by_sarong(labels, paths):
    """
    Kembalikan (train_idx, test_idx, groups) berdasarkan struktur sarung.
    groups: array ID sarung tiap citra (untuk verifikasi & laporan).
    """
    train_idx, test_idx, groups = [], [], []
    for i, (lbl, p) in enumerate(zip(labels, paths)):
        class_name = CLASS_ORDER[int(lbl)]
        idx = parse_index(p)
        group_id, subset = resolve_group(class_name, idx, p)
        groups.append(group_id)
        (train_idx if subset == "train" else test_idx).append(i)
    return np.array(train_idx), np.array(test_idx), np.array(groups, dtype=object)


def assert_no_leakage(groups, train_idx, test_idx):
    """Pastikan tidak ada sarung yang muncul di latih dan uji sekaligus."""
    train_g = set(groups[train_idx])
    test_g = set(groups[test_idx])
    overlap = train_g & test_g
    assert not overlap, f"BUG: sarung bocor ke latih & uji: {overlap}"


def print_distribution(labels, idx, subset_name):
    print(f"  {subset_name} (total {len(idx)} citra):")
    for cls_i, cls_name in enumerate(CLASS_ORDER):
        n = int(np.sum(labels[idx] == cls_i))
        disp = DISPLAY_NAMES.get(cls_name, cls_name)
        print(f"    - {disp:<20} : {n}")


def main():
    parser = argparse.ArgumentParser(
        description="Group-Based Split dataset fitur VGG16 (per sarung, deterministik)."
    )
    parser.add_argument("--model_dir", default="models",
                        help="Folder berisi vgg16_features.npz dan output split.")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    features_path = model_dir / "vgg16_features.npz"
    output_path = model_dir / "split_dataset.npz"

    if not features_path.exists():
        raise FileNotFoundError(
            f"{features_path} tidak ditemukan. Jalankan vgg16.py terlebih dahulu."
        )

    print("=" * 70)
    print("GROUP-BASED SPLIT (per sarung, sesuai Tabel 3.2)")
    print("=" * 70)

    data = np.load(features_path, allow_pickle=True)
    features = data["features"]
    labels = data["labels"]
    paths = data["paths"]
    print(f"Memuat {len(features)} citra dari {features_path.name}")

    train_idx, test_idx, groups = split_by_sarong(labels, paths)
    assert_no_leakage(groups, train_idx, test_idx)
    print(f"Jumlah sarung unik: {len(set(groups))}")

    X_train, X_test = features[train_idx], features[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    paths_train, paths_test = paths[train_idx], paths[test_idx]

    print("\nHasil pembagian:")
    print_distribution(labels, train_idx, "Data latih")
    print_distribution(labels, test_idx, "Data uji")

    np.savez_compressed(
        output_path,
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        paths_train=np.array([str(p) for p in paths_train], dtype=object),
        paths_test=np.array([str(p) for p in paths_test], dtype=object),
    )
    print(f"\nSplit disimpan ke: {output_path}")
    print(f"Total: {len(X_train)} latih / {len(X_test)} uji")
    print("SPLIT SELESAI.")


if __name__ == "__main__":
    main()
