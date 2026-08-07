# Urutan kelas harus tetap konsisten dari training sampai prediksi.
# Nama folder dataset harus sama dengan nama di CLASS_ORDER.
CLASS_ORDER = [
    "Hatta",
    "Pucuk_Rebung",
    "Cumi"
]

# Nama yang ditampilkan di aplikasi Streamlit
DISPLAY_NAMES = {
    "Hatta": "Hatta",
    "Pucuk_Rebung": "Pucuk Rebung",
    "Cumi": "Cumi / Bunga Dayak"
}

# Format gambar yang diterima
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

# Ukuran input VGG16
IMG_SIZE = (224, 224)

# Pengaturan training
BATCH_SIZE = 32
RANDOM_STATE = 50
TEST_SIZE = 0.40

PREPROCESS_MODE = "vgg16"

# ============================================================
# Nama file output
# ============================================================

# Model hasil training SVM
MODEL_FILENAME = "svm_cnn_model.pkl"

VGG16_FEATURES_FILENAME = "vgg16_features.npz"

# Mapping label angka ke nama kelas
CLASS_INDEX_FILENAME = "class_indices.json"