import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.svm import SVC

from config import (
    CLASS_ORDER,
    DISPLAY_NAMES,
    RANDOM_STATE,
    MODEL_FILENAME
)

SPLIT_FILENAME = "split_dataset.npz"


def train_svm(X_train, X_test, y_train, y_test):

    # ==============================================================
    # INFORMASI DATA
    # ==============================================================

    print("\n" + "=" * 70)
    print("INFORMASI DATA TRAINING")
    print("=" * 70)

    print(f"X_train shape : {X_train.shape}")
    print(f"X_test shape  : {X_test.shape}")
    print(f"y_train shape : {y_train.shape}")
    print(f"y_test shape  : {y_test.shape}")

    unique_train, train_counts = np.unique(
        y_train,
        return_counts=True
    )

    print("\nDistribusi kelas data training:")

    for label, count in zip(unique_train, train_counts):

        label = int(label)

        class_name = CLASS_ORDER[label]

        display_name = DISPLAY_NAMES.get(
            class_name,
            class_name
        )

        print(
            f"- {display_name:25s}: "
            f"{int(count):4d} citra"
        )

    # ==============================================================
    # VALIDASI JUMLAH DATA
    # ==============================================================

    min_class_count = int(train_counts.min())

    # Maksimal 5-fold Cross Validation
    cv_splits = min(
        5,
        min_class_count
    )

    if cv_splits < 2:

        raise ValueError(
            "Jumlah data latih per kelas terlalu sedikit "
            "untuk cross-validation."
        )

    print("\n" + "=" * 70)
    print("KONFIGURASI CROSS-VALIDATION")
    print("=" * 70)

    print(f"Jumlah fold : {cv_splits}")
    print("Strategi    : StratifiedKFold")
    print("Shuffle     : True")

    # ==============================================================
    # SVM TANPA STANDARD SCALER
    # ==============================================================

    svm = SVC(
        decision_function_shape="ovr",
        probability=True,
        random_state=RANDOM_STATE
    )

    # ==============================================================
    # PARAMETER GRID
    # ==============================================================

    param_grid = [

        # ----------------------------------------------------------
        # LINEAR SVM
        # ----------------------------------------------------------

        {
            "kernel": ["linear"],
            "C": [
                0.001,
                0.01,
                0.1,
                1,
                10,
                100,
                1000
            ]
        },

        # ----------------------------------------------------------
        # RBF SVM
        # ----------------------------------------------------------

        {
            "kernel": ["rbf"],
            "C": [
                0.01,
                0.1,
                1,
                10,
                100,
                1000
            ],
            "gamma": [
                "scale",
                "auto",
                0.0001,
                0.001,
                0.01
            ]
        }
    ]

    total_combinations = (
        7
        +
        (6 * 5)
    )

    print("\n" + "=" * 70)
    print("GRID SEARCH SVM TANPA STANDARD SCALER")
    print("=" * 70)

    print(
        f"Total kombinasi parameter : "
        f"{total_combinations}"
    )

    print(
        f"Cross-validation          : "
        f"{cv_splits}-fold"
    )

    print(
        f"Total training eksperimen : "
        f"{total_combinations * cv_splits}"
    )

    # ==============================================================
    # STRATIFIED K-FOLD
    # ==============================================================

    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    # ==============================================================
    # GRID SEARCH
    # ==============================================================

    grid = GridSearchCV(
        estimator=svm,
        param_grid=param_grid,
        scoring="f1_weighted",
        cv=cv,
        n_jobs=-1,
        verbose=3,
        return_train_score=True
    )

    print(
        "\nMulai GridSearchCV tanpa StandardScaler...\n"
    )

    grid.fit(
        X_train,
        y_train
    )

    # ==============================================================
    # MODEL TERBAIK
    # ==============================================================

    best_model = grid.best_estimator_

    print("\n" + "=" * 70)
    print("MODEL TERBAIK")
    print("=" * 70)

    print(
        f"Best parameters : "
        f"{grid.best_params_}"
    )

    print(
        f"Best CV F1      : "
        f"{grid.best_score_:.4f}"
    )

    # ==============================================================
    # PREDIKSI DATA TEST
    # ==============================================================

    y_pred = best_model.predict(
        X_test
    )

    y_proba = best_model.predict_proba(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    # ==============================================================
    # LABEL AKTIF
    # ==============================================================

    active_labels = sorted(
        np.unique(
            np.concatenate(
                [y_train, y_test]
            )
        ).tolist()
    )

    active_names = [
        DISPLAY_NAMES.get(
            CLASS_ORDER[int(i)],
            CLASS_ORDER[int(i)]
        )
        for i in active_labels
    ]

    # ==============================================================
    # CLASSIFICATION REPORT
    # ==============================================================

    report_text = classification_report(
        y_test,
        y_pred,
        labels=active_labels,
        target_names=active_names,
        zero_division=0
    )

    # ==============================================================
    # CONFUSION MATRIX
    # ==============================================================

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=active_labels
    )

    # ==============================================================
    # METRICS
    # ==============================================================

    metrics = {

        "feature_dimension": int(
            X_train.shape[1]
        ),

        "scaler": None,

        "model": "SVM",

        "best_params": grid.best_params_,

        "best_cv_score_f1_weighted": float(
            grid.best_score_
        ),

        "test_accuracy": float(
            accuracy
        ),

        "train_size": int(
            len(y_train)
        ),

        "test_size": int(
            len(y_test)
        ),

        "cv_folds": int(
            cv_splits
        ),

        "active_classes": active_names
    }

    return (
        best_model,
        metrics,
        report_text,
        cm,
        active_names,
        y_pred,
        y_proba
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Training model SVM menggunakan "
            "fitur VGG16 Block4 + GAP "
            "tanpa StandardScaler."
        )
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        default="models",
        help="Folder model."
    )

    args = parser.parse_args()

    model_dir = Path(
        args.model_dir
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    split_path = (
        model_dir /
        SPLIT_FILENAME
    )

    if not split_path.exists():

        raise FileNotFoundError(
            f"File split tidak ditemukan: "
            f"{split_path}. "
            "Jalankan split_dataset.py terlebih dahulu."
        )

    # ==============================================================
    # LOAD DATA SPLIT
    # ==============================================================

    data = np.load(
        split_path,
        allow_pickle=True
    )

    X_train = data["X_train"]
    X_test = data["X_test"]

    y_train = data["y_train"]
    y_test = data["y_test"]

    paths_test = data["paths_test"]

    print("=" * 70)
    print("TRAINING SVM")
    print("=" * 70)

    print(
        f"Train size : {len(y_train)}"
    )

    print(
        f"Test size  : {len(y_test)}"
    )

    print(
        f"Paths Test : {len(paths_test)}"
    )

    print(
        f"Feature dimension : "
        f"{X_train.shape[1]}"
    )

    # ==============================================================
    # VALIDASI FEATURE
    # ==============================================================

    if X_train.shape[1] != 512:

        print(
            "\nWARNING:"
            "\nFeature dimension bukan 512."
            "\nPastikan menggunakan VGG16 Block4 + GAP."
        )

    # ==============================================================
    # TRAIN
    # ==============================================================

    print(
        "\nMulai training SVM "
        "dengan GridSearchCV "
        "tanpa StandardScaler..."
    )

    (
        best_model,
        metrics,
        report_text,
        cm,
        active_names,
        y_pred,
        y_proba
    ) = train_svm(
        X_train,
        X_test,
        y_train,
        y_test
    )

    # ==============================================================
    # SAVE MODEL
    # ==============================================================

    model_path = (
        model_dir /
        MODEL_FILENAME
    )

    joblib.dump(
        best_model,
        model_path
    )

    # ==============================================================
    # SAVE TRAINING REPORT JSON
    # ==============================================================

    report_json_path = (
        model_dir /
        "training_report.json"
    )

    with open(
        report_json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2,
            ensure_ascii=False
        )

    # ==============================================================
    # SAVE CLASSIFICATION REPORT
    # ==============================================================

    report_txt_path = (
        model_dir /
        "classification_report.txt"
    )

    with open(
        report_txt_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            report_text
        )

    # ==============================================================
    # SAVE CONFUSION MATRIX
    # ==============================================================

    cm_df = pd.DataFrame(
        cm,
        index=active_names,
        columns=active_names
    )

    cm_path = (
        model_dir /
        "confusion_matrix.csv"
    )

    cm_df.to_csv(
        cm_path,
        encoding="utf-8"
    )

    # ==============================================================
    # HASIL TRAINING
    # ==============================================================

    print("\n" + "=" * 70)
    print("TRAINING SVM SELESAI")
    print("=" * 70)

    print(
        f"Model tersimpan : "
        f"{model_path}"
    )

    print(
        f"StandardScaler  : "
        f"Tidak digunakan"
    )

    print(
        f"Best params     : "
        f"{metrics['best_params']}"
    )

    print(
        f"Best CV F1      : "
        f"{metrics['best_cv_score_f1_weighted']:.4f}"
    )

    print(
        f"Test accuracy   : "
        f"{metrics['test_accuracy']:.4f}"
    )

    print(
        f"CV folds        : "
        f"{metrics['cv_folds']}"
    )

    # ==============================================================
    # CLASSIFICATION REPORT
    # ==============================================================

    print(
        "\nClassification Report:"
    )

    print(
        report_text
    )

    # ==============================================================
    # CONFUSION MATRIX
    # ==============================================================

    print(
        "\nConfusion Matrix:"
    )

    print(
        "=" * 70
    )

    print(
        cm_df.to_string()
    )

    # ==============================================================
    # GAMBAR SALAH KLASIFIKASI
    # ==============================================================

    wrong_idx = np.where(
        y_pred != y_test
    )[0]

    print("\n")
    print("=" * 70)
    print("GAMBAR SALAH KLASIFIKASI")
    print("=" * 70)

    if len(wrong_idx) == 0:

        print(
            "Tidak ada gambar yang "
            "salah diklasifikasikan."
        )

    else:

        for idx in wrong_idx:

            true_label = int(
                y_test[idx]
            )

            pred_label = int(
                y_pred[idx]
            )

            true_class = CLASS_ORDER[
                true_label
            ]

            pred_class = CLASS_ORDER[
                pred_label
            ]

            true_name = DISPLAY_NAMES.get(
                true_class,
                true_class
            )

            pred_name = DISPLAY_NAMES.get(
                pred_class,
                pred_class
            )

            confidence = (
                np.max(
                    y_proba[idx]
                ) * 100
            )

            print(
                f"File       : "
                f"{paths_test[idx]}"
            )

            print(
                f"True Class : "
                f"{true_name}"
            )

            print(
                f"Pred Class : "
                f"{pred_name}"
            )

            print(
                f"Confidence : "
                f"{confidence:.2f}%"
            )

            print(
                "-" * 70
            )


if __name__ == "__main__":
    main()