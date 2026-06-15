from pathlib import Path
import numpy as np

MODEL_DIR = Path("models")

split_data = np.load(
    MODEL_DIR / "split_dataset.npz",
    allow_pickle=True
)

X_train = split_data["X_train"]
y_train = split_data["y_train"]
paths_train = split_data["paths_train"]

output_path = MODEL_DIR / "retrieval_database.npz"

np.savez_compressed(
    output_path,
    features=X_train,
    labels=y_train,
    paths=paths_train
)

print("=" * 60)
print("DATABASE RETRIEVAL BERHASIL DIBUAT")
print("=" * 60)
print(f"Jumlah citra train : {len(y_train)}")
print(f"Output             : {output_path}")