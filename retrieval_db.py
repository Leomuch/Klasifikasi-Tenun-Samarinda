
from pathlib import Path

import numpy as np
from sklearn.preprocessing import normalize

from config import CLASS_ORDER


MODEL_DIR = Path("models")

split_data = np.load(
    MODEL_DIR / "split_dataset.npz",
    allow_pickle=True
)

X_train = split_data["X_train"]
y_train = split_data["y_train"]
paths_train = split_data["paths_train"]


# ============================================================
# HANYA KELAS MOTIF TENUN SAMARINDA
# ============================================================

NON_TENUN_LABEL = CLASS_ORDER.index(
    "Non_Tenun_Samarinda"
)

motif_mask = y_train != NON_TENUN_LABEL

X_train_motif = X_train[motif_mask]
y_train_motif = y_train[motif_mask]
paths_train_motif = paths_train[motif_mask]


# ============================================================
# L2-NORMALIZATION UNTUK FITUR CBIR
# ============================================================

X_train_motif = normalize(
    X_train_motif,
    norm="l2",
    axis=1
)


# ============================================================
# SIMPAN DATABASE RETRIEVAL
# ============================================================

output_path = MODEL_DIR / "retrieval_database.npz"

np.savez_compressed(
    output_path,
    features=X_train_motif,
    labels=y_train_motif,
    paths=paths_train_motif
)


# ============================================================
# INFORMASI DATABASE
# ============================================================

print("=" * 70)
print("DATABASE RETRIEVAL BERHASIL DIBUAT")
print("=" * 70)

print(f"Jumlah seluruh citra train       : {len(y_train)}")
print(f"Jumlah citra motif               : {len(y_train_motif)}")
print(
    "Jumlah citra Non_Tenun_Samarinda : "
    f"{np.sum(y_train == NON_TENUN_LABEL)}"
)
print(f"Bentuk fitur retrieval           : {X_train_motif.shape}")
print(f"Norma L2 fitur pertama           : {np.linalg.norm(X_train_motif[0]):.6f}" if len(X_train_motif) else "Database motif kosong")
print(f"Database CBIR                    : {output_path}")

print("=" * 70)