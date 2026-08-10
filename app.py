import uuid
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps

from classify_image import (
    load_feature_extractor,
    load_svm_model,
    load_class_indices,
    extract_single_feature,
    predict_feature
)

from retrieval import retrieve_similar

from config import (
    CLASS_ORDER,
    DISPLAY_NAMES,
    MODEL_FILENAME,
    VGG16_FEATURES_FILENAME,
    CLASS_INDEX_FILENAME
)


# ==============================================================
# KONFIGURASI STREAMLIT
# ==============================================================

st.set_page_config(
    page_title="Klasifikasi dan Retrieval Motif Tenun",
    layout="wide"
)


# ==============================================================
# PATH
# ==============================================================

MODEL_DIR = Path("models")

MODEL_PATH = (
    MODEL_DIR /
    MODEL_FILENAME
)

FEATURE_DB_PATH = (
    MODEL_DIR /
    VGG16_FEATURES_FILENAME
)

CLASS_INDEX_PATH = (
    MODEL_DIR /
    CLASS_INDEX_FILENAME
)

TEMP_DIR = Path(
    "temp_uploads"
)

TEMP_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==============================================================
# CACHE MODEL
# ==============================================================

@st.cache_resource
def cached_feature_extractor():

    return load_feature_extractor()


@st.cache_resource
def cached_svm_model():

    return load_svm_model(
        MODEL_DIR
    )


@st.cache_data
def cached_class_indices():

    return load_class_indices(
        MODEL_DIR
    )


# ==============================================================
# SIMPAN FILE UPLOAD
# ==============================================================

def save_uploaded_file(
    uploaded_file
):

    suffix = Path(
        uploaded_file.name
    ).suffix.lower()

    if suffix == "":
        suffix = ".jpg"

    temp_path = (
        TEMP_DIR /
        f"query_{uuid.uuid4().hex}{suffix}"
    )

    with open(
        temp_path,
        "wb"
    ) as f:

        f.write(
            uploaded_file.getbuffer()
        )

    return temp_path


# ==============================================================
# BUKA CITRA
# ==============================================================

def open_display_image(
    uploaded_file
):

    image = Image.open(
        uploaded_file
    )

    image = ImageOps.exif_transpose(
        image
    )

    image = image.convert(
        "RGB"
    )

    return image


# ==============================================================
# SESSION STATE
# ==============================================================

if "classified" not in st.session_state:

    st.session_state.classified = False


if "predicted_label" not in st.session_state:

    st.session_state.predicted_label = None


if "confidence" not in st.session_state:

    st.session_state.confidence = None


if "probability_dict" not in st.session_state:

    st.session_state.probability_dict = {}


if "retrieval_results" not in st.session_state:

    st.session_state.retrieval_results = []


# ==============================================================
# HEADER
# ==============================================================

st.markdown(
    """
    <h2 style='text-align:center;'>
        Sistem Klasifikasi dan Retrieval Motif Tenun Samarinda
    </h2>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <p style='text-align:center; font-size:14px;'>
        Alur sistem: preprocessing dataset, ekstraksi fitur VGG16,
        data splitting, training SVM, klasifikasi citra,
        lalu retrieval citra paling mirip.
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()


# ==============================================================
# VALIDASI FILE
# ==============================================================

missing_files = []


if not MODEL_PATH.exists():

    missing_files.append(
        str(MODEL_PATH)
    )


if not FEATURE_DB_PATH.exists():

    missing_files.append(
        str(FEATURE_DB_PATH)
    )


if not CLASS_INDEX_PATH.exists():

    missing_files.append(
        str(CLASS_INDEX_PATH)
    )


if missing_files:

    st.error(
        "Model atau file pendukung belum ditemukan."
    )

    st.write(
        "Jalankan pipeline terlebih dahulu:"
    )

    st.code(
        "python main.py preprocess\n"
        "python main.py extract_features\n"
        "python main.py split\n"
        "python main.py train",
        language="powershell"
    )

    st.write(
        "File yang belum ditemukan:"
    )

    for file in missing_files:

        st.write(
            f"- `{file}`"
        )

    st.stop()


# ==============================================================
# LOAD MODEL
# ==============================================================

try:

    feature_model = (
        cached_feature_extractor()
    )

    svm_model = (
        cached_svm_model()
    )

    class_indices = (
        cached_class_indices()
    )

except Exception as e:

    st.error(
        "Gagal memuat model atau feature extractor."
    )

    st.exception(e)

    st.stop()


# ==============================================================
# INFORMASI MODEL
# ==============================================================

with st.expander(
    "Informasi Model"
):

    st.write(
        f"Model SVM: `{MODEL_PATH}`"
    )

    st.write(
        f"Feature database: `{FEATURE_DB_PATH}`"
    )

    st.write(
        f"Class indices: `{CLASS_INDEX_PATH}`"
    )

    # ----------------------------------------------------------
    # INFORMASI KELAS SVM
    # ----------------------------------------------------------

    if hasattr(
        svm_model,
        "classes_"
    ):

        st.write(
            "Kelas yang sudah dipelajari model:"
        )

        for class_index in svm_model.classes_:

            class_index = int(
                class_index
            )

            class_name = (
                CLASS_ORDER[class_index]
            )

            display_name = (
                DISPLAY_NAMES.get(
                    class_name,
                    class_name
                )
            )

            st.write(
                f"- {display_name}"
            )

        if len(
            svm_model.classes_
        ) < len(CLASS_ORDER):

            st.warning(
                "Model saat ini belum mempelajari semua kelas."
            )


# ==============================================================
# LAYOUT
# ==============================================================

col1, col2 = st.columns(
    [1, 1.2]
)


# ==============================================================
# KOLOM KIRI
# ==============================================================

with col1:

    st.subheader(
        "Upload Citra"
    )

    uploaded_file = st.file_uploader(
        "Format: JPG, JPEG, PNG, BMP, WEBP",
        type=[
            "jpg",
            "jpeg",
            "png",
            "bmp",
            "webp"
        ]
    )

    # ----------------------------------------------------------
    # TOP-K
    # ----------------------------------------------------------

    top_k = st.slider(
        "Jumlah hasil retrieval",
        min_value=1,
        max_value=10,
        value=5
    )

    # ----------------------------------------------------------
    # UPLOAD CITRA
    # ----------------------------------------------------------

    if uploaded_file is not None:

        try:

            display_image = (
                open_display_image(
                    uploaded_file
                )
            )

            st.image(
                display_image,
                width=260
            )

            st.caption(
                f"Nama File: {uploaded_file.name}"
            )

            st.caption(
                f"Dimensi Asli: "
                f"{display_image.size[0]} x "
                f"{display_image.size[1]} px"
            )

            # --------------------------------------------------
            # BUTTON
            # --------------------------------------------------

            if st.button(
                "Klasifikasi dan Retrieval",
                type="primary"
            ):

                with st.spinner(
                    "Memproses citra..."
                ):

                    # ==========================================
                    # 1. SIMPAN CITRA QUERY
                    # ==========================================

                    temp_image_path = (
                        save_uploaded_file(
                            uploaded_file
                        )
                    )

                    # ==========================================
                    # 2. EKSTRAKSI FITUR VGG16
                    # ==========================================

                    query_feature = (
                        extract_single_feature(
                            feature_model=feature_model,
                            image_path=temp_image_path
                        )
                    )

                    # ==========================================
                    # INFORMASI FITUR
                    # ==========================================

                    st.write(
                        f"Query feature shape: "
                        f"{query_feature.shape}"
                    )

                    if (
                        query_feature.shape[-1]
                        == 512
                    ):

                        st.write(
                            "✓ Query menggunakan "
                            "512 fitur "
                            "(Block4 + GAP)."
                        )

                    # ==========================================
                    # 3. KLASIFIKASI SVM
                    # ==========================================

                    prediction = (
                        predict_feature(
                            svm_model=svm_model,
                            query_feature=query_feature,
                            class_indices=class_indices
                        )
                    )

                    # ==========================================
                    # 4. RETRIEVAL
                    # ==========================================
                    #
                    # Retrieval dilakukan dari SELURUH kelas.
                    #
                    # Tidak menggunakan predicted_label
                    # sebagai filter.
                    #
                    # Tidak menggunakan StandardScaler.
                    #
                    # Query:
                    #     VGG16 Block4 + GAP = 512D
                    #
                    # Database:
                    #     VGG16 Block4 + GAP = 512D
                    #
                    # Distance:
                    #     Euclidean Distance
                    # ==========================================

                    retrieval_results = (
                        retrieve_similar(
                            query_feature=query_feature,
                            model_dir=MODEL_DIR,
                            top_k=top_k
                        )
                    )

                # ==============================================
                # SAVE SESSION STATE
                # ==============================================

                st.session_state.classified = True

                st.session_state.predicted_label = (
                    prediction["label_name"]
                )

                st.session_state.confidence = (
                    prediction["confidence"]
                )

                st.session_state.probability_dict = (
                    prediction["probability_dict"]
                )

                st.session_state.retrieval_results = (
                    retrieval_results
                )

        except Exception as e:

            st.error(
                "Terjadi error saat membaca "
                "atau memproses gambar."
            )

            st.exception(e)


# ==============================================================
# KOLOM KANAN
# ==============================================================

with col2:

    st.subheader(
        "Hasil Klasifikasi"
    )

    if st.session_state.classified:

        st.success(
            "Klasifikasi berhasil"
        )

        st.write(
            f"**Label Motif:** "
            f"{st.session_state.predicted_label}"
        )

        st.write(
            f"**Confidence Score:** "
            f"{st.session_state.confidence:.2f}%"
        )

        # ------------------------------------------------------
        # PROBABILITAS
        # ------------------------------------------------------

        if (
            st.session_state.probability_dict
        ):

            st.markdown(
                "**Distribusi Probabilitas Kelas**"
            )

            st.bar_chart(
                st.session_state.probability_dict
            )

        else:

            st.info(
                "Probabilitas kelas tidak tersedia."
            )

    else:

        st.info(
            "Hasil klasifikasi akan tampil "
            "setelah citra diproses."
        )


# ==============================================================
# RETRIEVAL
# ==============================================================

st.divider()

st.subheader(
    "Top Citra Paling Mirip"
)


if st.session_state.classified:

    results = (
        st.session_state.retrieval_results
    )

    if results:

        retrieval_cols = st.columns(
            len(results)
        )

        for col, item in zip(
            retrieval_cols,
            results
        ):

            with col:

                img_path = Path(
                    item["path"]
                )

                if img_path.exists():

                    img = (
                        Image.open(
                            img_path
                        ).convert("RGB")
                    )

                    st.image(
                        img,
                        caption=(
                            f"Top-{item['rank']}"
                        ),
                        use_container_width=True
                    )

                    st.caption(
                        f"Kelas: "
                        f"{item['label_name']}"
                    )

                    st.caption(
                        f"Distance: "
                        f"{item['distance']:.4f}"
                    )

                    st.caption(
                        f"File: "
                        f"{img_path.name}"
                    )

                else:

                    st.warning(
                        f"File tidak ditemukan: "
                        f"{img_path}"
                    )

    else:

        st.warning(
            "Tidak ada hasil retrieval yang ditemukan."
        )

else:

    st.info(
        "Hasil retrieval akan tampil "
        "setelah klasifikasi selesai."
    )