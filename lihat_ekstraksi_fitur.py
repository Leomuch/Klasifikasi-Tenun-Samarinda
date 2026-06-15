import numpy as np
import pandas as pd
from pathlib import Path

from config import CLASS_ORDER, DISPLAY_NAMES

data = np.load("models/vgg16_features.npz", allow_pickle=True)

features = data["features"]
labels = data["labels"]
paths = data["paths"]

rows = []
for i in range(len(paths)):
    path = Path(str(paths[i]))
    label_index = int(labels[i])
    class_folder = CLASS_ORDER[label_index]
    class_name = DISPLAY_NAMES.get(class_folder, class_folder)

    row = {
        "no": i + 1,
        "nama_file": path.name,
        "kelas": class_name,
        "label": label_index,
        "path": str(path),
        "jumlah_fitur": features.shape[1],
        "mean_fitur": features[i].mean(),
        "std_fitur": features[i].std(),
        "min_fitur": features[i].min(),
        "max_fitur": features[i].max(),
    }

    # tampilkan 20 fitur pertama saja
    for j in range(20):
        row[f"fitur_{j+1:04d}"] = features[i][j]

    rows.append(row)

df = pd.DataFrame(rows)

print(df.head())

df.to_csv("models/preview_fitur_vgg16.csv", index=False)
print("\nCSV disimpan ke models/preview_fitur_vgg16.csv")