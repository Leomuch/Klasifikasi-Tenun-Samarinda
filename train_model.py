import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import (
    CLASS_ORDER,
    DISPLAY_NAMES,
    RANDOM_STATE,
    MODEL_FILENAME
)


SPLIT_FILENAME = "split_dataset.npz"

def train_svm(X_train, X_test, y_train, y_test):
    _, train_counts = np.unique(y_train, return_counts=True)

    cv_splits = min(3, int(train_counts.min()))

    if cv_splits < 2:
        raise ValueError(
            "Jumlah data latih per kelas terlalu sedikit untuk cross-validation."
        )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            decision_function_shape="ovr",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ])

    param_grid = [
        {
            "svm__kernel": ["linear"],
            "svm__C": [0.1, 1, 10, 100]
        },
        {
            "svm__kernel": ["rbf"],
            "svm__C": [0.01, 0.1, 1, 10, 100],
            "svm__gamma": ["scale", "auto"]
        }
    ]

    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_weighted",
        cv=cv,
        n_jobs=1,
        verbose=1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    active_labels = sorted(np.unique(np.concatenate([y_train, y_test])).tolist())
    active_names = [
        DISPLAY_NAMES.get(CLASS_ORDER[i], CLASS_ORDER[i])
        for i in active_labels
    ]

    report_text = classification_report(
        y_test,
        y_pred,
        labels=active_labels,
        target_names=active_names,
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=active_labels
    )

    metrics = {
        "best_params": grid.best_params_,
        "best_cv_score_f1_weighted": float(grid.best_score_),
        "test_accuracy": float(accuracy),
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
        "active_classes": active_names
    }

    return best_model, metrics, report_text, cm, active_names, y_pred, y_proba


def main():
    parser = argparse.ArgumentParser(
        description="Training model SVM menggunakan fitur VGG16."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models",
        help="Folder model."
    )

    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    split_path = model_dir / SPLIT_FILENAME

    if not split_path.exists():
        raise FileNotFoundError(
            f"File split tidak ditemukan: {split_path}. "
            "Jalankan split_dataset.py terlebih dahulu."
        )

    data = np.load(split_path, allow_pickle=True)

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    paths_test = data["paths_test"]

    print("=" * 70)
    print("TRAINING SVM")
    print("=" * 70)
    print(f"Train size: {len(y_train)}")
    print(f"Test size : {len(y_test)}")
    print(f"Paths Tes : {len(paths_test)}")

    print("\nMulai training SVM dengan GridSearchCV...")
    best_model, metrics, report_text, cm, active_names, y_pred, y_proba = train_svm(
        X_train,
        X_test,
        y_train,
        y_test
    )

    model_path = model_dir / MODEL_FILENAME
    joblib.dump(best_model, model_path)

    report_json_path = model_dir / "training_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    report_txt_path = model_dir / "classification_report.txt"
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    cm_df = pd.DataFrame(
        cm,
        index=active_names,
        columns=active_names
    )

    cm_path = model_dir / "confusion_matrix.csv"
    cm_df.to_csv(cm_path, encoding="utf-8")

    print("\n" + "=" * 70)
    print("TRAINING SVM SELESAI")
    print("=" * 70)
    print(f"Model tersimpan di: {model_path}")
    print(f"Best params: {metrics['best_params']}")
    print(f"Best CV F1 weighted: {metrics['best_cv_score_f1_weighted']:.4f}")
    print(f"Test accuracy: {metrics['test_accuracy']:.4f}")
    print("\nClassification Report:")
    print(report_text)
    print("\nConfusion Matrix:")
    print("=" * 70)
    print(cm_df.to_string())
    wrong_idx = np.where(y_pred != y_test)[0]

    print("\n")
    print("=" * 70)
    print("GAMBAR SALAH KLASIFIKASI")
    print("=" * 70)

    for idx in wrong_idx:

        true_name = active_names[y_test[idx]]
        pred_name = active_names[y_pred[idx]]

        confidence = np.max(y_proba[idx]) * 100

        print(f"File       : {paths_test[idx]}")
        print(f"True Class : {true_name}")
        print(f"Pred Class : {pred_name}")
        print(f"Confidence : {confidence:.2f}%")
        print("-" * 70)


if __name__ == "__main__":
    main()