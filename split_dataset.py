"""
split_dataset.py
================
Membagi dataset fitur VGG16 menjadi data latih dan data uji menggunakan
GROUP-BASED SPLIT yang sesungguhnya, yaitu pengelompokan berdasarkan
sumber kain (sarong). Seluruh citra yang berasal dari satu sarong yang
sama dijamin hanya masuk ke salah satu subset (latih ATAU uji), sehingga
tidak terjadi data leakage.

Perbedaan dengan versi lama:
- Versi lama membagi berdasarkan ambang NOMOR file per kelas
  (mis. Hatta 1-50 latih, 51-100 uji). Itu bukan group-based split
  karena tidak menjamin potongan dari sarong yang sama tidak tersebar
  di kedua subset.
- Versi ini menggunakan sklearn.model_selection.GroupShuffleSplit
  dengan grup = ID sarong tiap citra.

Cara menentukan ID sarong (grup), berurutan prioritas:
  1. File CSV opsional (--groups_csv) berisi kolom: path,group
     -> paling andal, dipakai bila tersedia.
  2. Pola nama file: <kelas>_<sarong>_<indeks>.<ext>
     contoh: hatta_03_007.jpg  -> grup = "hatta_03"
     (aktif bila --group_from filename, default).
  3. Fallback: seluruh nomor pada nama file dijadikan grup individual
     (setara per-citra) dan akan MEMUNCULKAN PERINGATAN, karena berarti
     tidak ada informasi sarong -> bukan group-based split sejati.

Output split_dataset.npz kompatibel dengan retrieval_db.py dan
evaluate_retrieval.py (key: X_train, X_test, y_train, y_test,
paths_train, paths_test).
"""

import argparse
import csv
import os
import re
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from config import RANDOM_STATE, TEST_SIZE, CLASS_ORDER, DISPLAY_NAMES


# ---------------------------------------------------------------------------
# Penentuan ID grup (sarong)
# ---------------------------------------------------------------------------
def load_group_map_from_csv(csv_path):
    """Baca CSV berisi kolom 'path' dan 'group'. Return dict {path_str: group}."""
    mapping = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "path" not in reader.fieldnames or "group" not in reader.fieldnames:
            raise ValueError(
                "groups_csv harus punya header kolom: path,group "
                f"(ditemukan: {reader.fieldnames})"
            )
        for row in reader:
            # Normalisasi path agar cocok lintas OS (Windows/Unix).
            key = os.path.normpath(row["path"].strip())
            mapping[key] = row["group"].strip()
    return mapping


def group_id_from_filename(path):
    """
    Ambil ID sarong dari nama file berpola <kelas>_<sarong>_<indeks>.

    contoh:
      hatta_03_007.jpg   -> "hatta_03"
      pucuk_rebung_02_11 -> "pucuk_rebung_02"   (kelas boleh mengandung '_')

    Aturan: buang komponen numerik TERAKHIR (indeks citra), sisanya jadi grup.
    Return None bila pola tidak dikenali (tidak ada dua komponen bernomor).
    """
    stem = Path(path).stem.lower()
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    # Komponen terakhir harus angka (indeks citra dalam satu sarong).
    if not parts[-1].isdigit():
        return None
    # Komponen kedua-dari-belakang idealnya nomor sarong.
    group = "_".join(parts[:-1])
    return group


def build_groups(paths, group_from="filename", groups_csv=None):
    """
    Bangun array ID grup untuk setiap path.
    Return (groups: np.ndarray[str], mode_efektif: str, n_unique: int).
    """
    paths = [str(p) for p in paths]

    if groups_csv:
        mapping = load_group_map_from_csv(groups_csv)
        groups = []
        missing = []
        for p in paths:
            key = os.path.normpath(p)
            # Coba cocokkan penuh; kalau tidak, coba basename.
            g = mapping.get(key) or mapping.get(os.path.basename(p))
            if g is None:
                missing.append(p)
                g = f"__nogroup__{os.path.basename(p)}"
            groups.append(g)
        if missing:
            print(f"[PERINGATAN] {len(missing)} citra tidak ada di groups_csv; "
                  f"dianggap grup sendiri. Contoh: {missing[:3]}")
        groups = np.array(groups, dtype=object)
        return groups, "csv", len(set(groups))

    if group_from == "filename":
        groups = []
        unknown = 0
        for p in paths:
            g = group_id_from_filename(p)
            if g is None:
                unknown += 1
                # fallback: nama file utuh -> grup individual
                g = f"__solo__{Path(p).stem.lower()}"
            groups.append(g)
        groups = np.array(groups, dtype=object)
        n_unique = len(set(groups))
        if unknown > 0:
            print(f"[PERINGATAN] {unknown} nama file tidak mengikuti pola "
                  f"<kelas>_<sarong>_<indeks>. Citra tsb diperlakukan sebagai "
                  f"grup individual. Untuk group-split sejati, seragamkan "
                  f"penamaan atau pakai --groups_csv.")
        return groups, "filename", n_unique

    raise ValueError(f"group_from tidak dikenali: {group_from}")


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------
def group_based_split(features, labels, paths, groups,
                      test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """
    Bagi data dengan GroupShuffleSplit sehingga satu grup (sarong) tidak
    pernah muncul di train dan test sekaligus.
    """
    n_groups = len(set(groups))
    if n_groups < 2:
        raise ValueError(
            f"Hanya ada {n_groups} grup unik. Group-based split butuh >= 2 grup. "
            "Periksa penamaan file / groups_csv."
        )

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size,
                            random_state=random_state)
    train_idx, test_idx = next(gss.split(features, labels, groups=groups))

    # Verifikasi tidak ada grup yang bocor ke dua sisi.
    train_groups = set(np.array(groups)[train_idx])
    test_groups = set(np.array(groups)[test_idx])
    overlap = train_groups & test_groups
    assert not overlap, f"BUG: grup bocor ke train & test: {overlap}"

    return train_idx, test_idx


def print_distribution(labels, idx, subset_name):
    """Cetak jumlah citra per kelas pada subset tertentu."""
    sub = labels[idx]
    print(f"  {subset_name} (total {len(idx)} citra):")
    for cls_i, cls_name in enumerate(CLASS_ORDER):
        n = int(np.sum(sub == cls_i))
        disp = DISPLAY_NAMES.get(cls_name, cls_name)
        print(f"    - {disp:<20} : {n}")


def main():
    parser = argparse.ArgumentParser(
        description="Group-Based Split dataset fitur VGG16 (per sarong)."
    )
    parser.add_argument("--model_dir", default="models",
                        help="Folder berisi vgg16_features.npz dan output split.")
    parser.add_argument("--test_size", type=float, default=TEST_SIZE,
                        help="Proporsi GRUP untuk data uji (default dari config).")
    parser.add_argument("--random_state", type=int, default=RANDOM_STATE,
                        help="Seed untuk reprodusibilitas.")
    parser.add_argument("--group_from", default="filename",
                        choices=["filename"],
                        help="Sumber ID grup bila --groups_csv tidak diberikan.")
    parser.add_argument("--groups_csv", default=None,
                        help="CSV opsional berisi kolom path,group (paling andal).")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    features_path = model_dir / "vgg16_features.npz"
    output_path = model_dir / "split_dataset.npz"

    if not features_path.exists():
        raise FileNotFoundError(
            f"{features_path} tidak ditemukan. Jalankan vgg16.py terlebih dahulu."
        )

    print("=" * 70)
    print("GROUP-BASED SPLIT (pengelompokan per sarong)")
    print("=" * 70)

    data = np.load(features_path, allow_pickle=True)
    features = data["features"]
    labels = data["labels"]
    paths = data["paths"]
    print(f"Memuat {len(features)} citra dari {features_path.name}")

    groups, mode, n_groups = build_groups(
        paths, group_from=args.group_from, groups_csv=args.groups_csv
    )
    print(f"Mode grup: {mode} | jumlah sarong (grup) unik: {n_groups}")

    train_idx, test_idx = group_based_split(
        features, labels, paths, groups,
        test_size=args.test_size, random_state=args.random_state,
    )

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
