
from pathlib import Path
import random
import shutil

# Lokasi dataset Batik Nusantara
SOURCE = Path("data/archive/raw_batik_v2.1/train")

# Folder kelas pembanding proyek
DEST = Path("data/raw/Non_Tenun")

JUMLAH_PER_KELAS = 5
SEED = 42

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

random.seed(SEED)

if not SOURCE.exists():
    raise FileNotFoundError(
        f"Folder train tidak ditemukan: {SOURCE.resolve()}"
    )

DEST.mkdir(parents=True, exist_ok=True)

total = 0

for class_dir in sorted(SOURCE.iterdir()):
    if not class_dir.is_dir():
        continue

    images = [
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    ]

    if not images:
        print(f"[LEWATI] Tidak ada gambar: {class_dir.name}")
        continue

    jumlah = min(JUMLAH_PER_KELAS, len(images))
    selected = random.sample(images, jumlah)

    for i, src in enumerate(selected, start=1):
        # Nama unik agar tidak menimpa gambar dari kelas lain
        filename = (
            f"BATIK_{class_dir.name}_{i:03d}{src.suffix.lower()}"
        )
        dst = DEST / filename

        if dst.exists():
            print(f"[LEWATI] File sudah ada: {dst.name}")
            continue

        shutil.copy2(src, dst)
        total += 1

    print(
        f"{class_dir.name}: "
        f"{jumlah} gambar dipilih dari {len(images)} gambar"
    )

print(f"\nSelesai. Total gambar baru disalin: {total}")
print(f"Lokasi tujuan: {DEST.resolve()}")