# Klasifikasi Motif Tenun Samarinda (CNN-SVM + CBIR)

Sistem klasifikasi motif Tenun Samarinda menggunakan **VGG16 sebagai feature
extractor (CNN)**, **SVM sebagai classifier**, dan **Content-Based Image
Retrieval (CBIR)** untuk menampilkan citra serupa. Antarmuka pengguna dibangun
dengan **Streamlit**.

Alur singkat: sebuah citra query diekstraksi fiturnya oleh VGG16 (vektor 512
dimensi), diklasifikasikan oleh SVM, lalu sistem menampilkan Top-5 citra paling
mirip dari database — dibatasi pada kelas hasil prediksi — menggunakan
Euclidean Distance.

## Kelas Motif

Terdapat **3 kelas** motif:

| Nama folder    | Nama tampil        |
| -------------- | ------------------ |
| `Hatta`        | Hatta              |
| `Pucuk_Rebung` | Pucuk Rebung       |
| `Cumi`         | Cumi / Bunga Dayak |

Daftar kelas didefinisikan di `config.py` (`CLASS_ORDER`). Nama folder pada
`data/raw/` harus persis sama dengan entri `CLASS_ORDER`.

## Kebutuhan Sistem

- Python 3.10 atau 3.11
- Paket pada `requirements.txt` (TensorFlow, scikit-learn, numpy, pandas,
  pillow, joblib, streamlit)

## Instalasi

```bash
# 1. Buat virtual environment
python -m venv .venv

# 2. Aktifkan
#    Windows (PowerShell/CMD):
.venv\Scripts\activate
#    Linux/macOS:
source .venv/bin/activate

# 3. Pasang dependency
pip install -r requirements.txt
```

## Struktur Folder Dataset

Letakkan citra mentah dengan struktur folder-per-kelas berikut:

```
data/
  raw/
    Hatta/
      hatta_01_001.jpg
      hatta_01_002.jpg
      ...
    Pucuk_Rebung/
      pucuk_rebung_01_001.jpg
      ...
    Cumi/
      cumi_01_001.jpg
      ...
```

### Penamaan file untuk Group-Based Split

Agar pembagian data benar-benar mengelompokkan per **kain/sarong** (mencegah
*data leakage*), beri nama file dengan pola:

```
<kelas>_<idsarong>_<indeks>.jpg
```

Contoh: `hatta_03_007.jpg` berarti kelas Hatta, sarong ke-03, citra ke-007.
`split_dataset.py` akan membaca `hatta_03` sebagai satu grup, sehingga seluruh
potongan dari sarong itu tidak akan tersebar ke data latih dan data uji
sekaligus.

Bila penamaan tidak seragam, sediakan file CSV pemetaan dan jalankan split
dengan `--groups_csv`:

```csv
path,group
data/processed/Hatta/hatta_a_001.jpg,hatta_sarong_a
data/processed/Hatta/hatta_a_002.jpg,hatta_sarong_a
```

Folder `data/` dan `models/` sengaja diabaikan oleh `.gitignore` sehingga tidak
ikut ter-commit.

## Cara Menjalankan

Semua tahap dapat dijalankan lewat launcher `main.py`.

### Jalankan seluruh pipeline sekaligus

```bash
python main.py pipeline
```

Mode `pipeline` menjalankan berurutan: `preprocess -> validate -> vgg16 ->
split -> train`.

### Atau jalankan per tahap

```bash
# 1. Preprocessing: data/raw -> data/processed (resize 224x224, RGB, dsb.)
python main.py preprocess

# 2. Validasi: cek jumlah citra per kelas
python main.py validate --dataset_dir data/processed

# 3. Ekstraksi fitur VGG16 (menghasilkan models/vgg16_features.npz)
python main.py vgg16

# 4. Group-Based Split (menghasilkan models/split_dataset.npz)
python main.py split
#    -> jika penamaan file tidak seragam:
python split_dataset.py --groups_csv groups.csv

# 5. Latih SVM (GridSearch, menghasilkan models/svm_cnn_model.pkl + laporan)
python main.py train
```

### Klasifikasi & retrieval satu citra

```bash
# Klasifikasi satu citra
python main.py classify --image_path path/ke/citra.jpg

# Retrieval Top-5 citra serupa (dibatasi kelas prediksi)
python main.py retrieve --image_path path/ke/citra.jpg --top_k 5
```

### Bangun database retrieval & evaluasi

```bash
# Bangun retrieval_database.npz dari data latih
python retrieval_db.py

# Evaluasi Mean Precision@5 pada data uji
python evaluate_retrieval.py
```

### Menjalankan antarmuka web

```bash
python main.py app
# atau
streamlit run app.py
```

## Output yang Dihasilkan (folder `models/`)

| File                     | Keterangan                                    |
| ------------------------ | --------------------------------------------- |
| `vgg16_features.npz`     | Fitur 512-dim seluruh citra + label + path    |
| `split_dataset.npz`      | Data latih & uji hasil group-based split      |
| `retrieval_database.npz` | Database fitur (data latih) untuk CBIR        |
| `svm_cnn_model.pkl`      | Model SVM terlatih                            |
| `class_indices.json`     | Pemetaan indeks kelas ke nama                 |
| laporan metrik/CM        | JSON metrik, classification report, CSV CM    |

## Metode Evaluasi

- **Klasifikasi:** accuracy, precision, recall, F1-score, confusion matrix.
- **Retrieval:** Mean Precision@5. Retrieval dibatasi pada kelas hasil prediksi;
  sebuah hasil dianggap relevan bila labelnya sama dengan label sebenarnya
  citra query. Karena kandidat retrieval dibatasi pada kelas prediksi, nilai
  Mean Precision@5 akan cenderung mengikuti akurasi klasifikasi — hal ini
  merupakan konsekuensi desain, bukan kebetulan.

## Catatan Reprodusibilitas

- Normalisasi citra memakai `preprocess_input` bawaan VGG16 (mean-subtraction
  ala ImageNet), bukan pembagian sederhana `/255`. Diatur lewat
  `PREPROCESS_MODE` di `config.py`.
- Seed acak diatur lewat `RANDOM_STATE` di `config.py`.

