
from pathlib import Path

import numpy as np

from vgg16 import load_feature_extractor
from classify_image import extract_single_feature
from retrieval import retrieve_similar
from config import CLASS_ORDER, DISPLAY_NAMES


MODEL_DIR = Path("models")
TOP_K = 5


def main():
    print("Memuat feature extractor VGG16...")
    feature_model = load_feature_extractor()
    print("Feature extractor berhasil dimuat.")

    data = np.load(
        MODEL_DIR / "split_dataset.npz",
        allow_pickle=True
    )

    paths_test = data["paths_test"]
    y_test = data["y_test"]

    # Evaluasi hanya pada tiga kelas motif Tenun Samarinda.
    non_tenun_label = CLASS_ORDER.index(
        "Non_Tenun_Samarinda"
    )

    motif_mask = y_test != non_tenun_label

    paths_test = paths_test[motif_mask]
    y_test = y_test[motif_mask]

    if len(y_test) == 0:
        raise ValueError(
            "Tidak ada citra uji motif Tenun Samarinda."
        )

    precision_scores = []
    scores_by_class = {
        i: []
        for i, name in enumerate(CLASS_ORDER)
        if name != "Non_Tenun_Samarinda"
    }

    print(f"\nJumlah query motif tenun: {len(paths_test)}")
    print(f"Top-K: {TOP_K}")

    for i, (image_path, true_label) in enumerate(
        zip(paths_test, y_test),
        start=1
    ):
        image_path = Path(str(image_path))
        true_label = int(true_label)

        query_feature = extract_single_feature(
            feature_model,
            image_path
        )

        results = retrieve_similar(
            query_feature=query_feature,
            model_dir=MODEL_DIR,
            top_k=TOP_K
        )

        # Precision@5 berdasarkan label asli query.
        relevant = sum(
            1
            for item in results
            if item["label_index"] == true_label
        )

        precision_at_5 = relevant / TOP_K

        precision_scores.append(precision_at_5)
        scores_by_class[true_label].append(
            precision_at_5
        )

        print(
            f"[{i}/{len(paths_test)}] "
            f"{image_path.name}: "
            f"Precision@{TOP_K} = {precision_at_5:.2f}"
        )

    print("\n" + "=" * 60)
    print("EVALUASI RETRIEVAL TENUN SAMARINDA")
    print("=" * 60)
    print(f"Jumlah query: {len(precision_scores)}")
    print(f"Top-K: {TOP_K}")
    print(
        f"Mean Precision@{TOP_K}: "
        f"{np.mean(precision_scores):.4f}"
    )

    print("\nPrecision@5 per kelas:")

    for class_index, class_scores in scores_by_class.items():
        class_name = CLASS_ORDER[class_index]
        display_name = DISPLAY_NAMES.get(
            class_name,
            class_name
        )

        if class_scores:
            print(
                f"- {display_name}: "
                f"{np.mean(class_scores):.4f} "
                f"({len(class_scores)} query)"
            )
        else:
            print(
                f"- {display_name}: tidak ada query"
            )


if __name__ == "__main__":
    main()
