import argparse
import subprocess
import sys

def run_command(command):
    print("\nMenjalankan:")
    print(" ".join(command))
    print("-" * 70)
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Main launcher project Klasifikasi Motif Tenun Samarinda CNN-SVM-CBIR"
    )

    parser.add_argument(
        "mode",
        choices=[
            "validate",
            "preprocess",
            "vgg16",
            "split",
            "train",
            "classify",
            "retrieve",
            "app",
            "pipeline"
        ],
        help=(
            "validate, preprocess, vgg16, split, train, "
            "classify, retrieve, app, atau pipeline."
        )
    )

    parser.add_argument(
        "--dataset_dir",
        default="data/raw",
        help="Folder dataset mentah. Default: data/raw"
    )

    parser.add_argument(
        "--processed_dir",
        default="data/processed",
        help="Folder dataset hasil preprocessing. Default: data/processed"
    )

    parser.add_argument(
        "--model_dir",
        default="models",
        help="Folder penyimpanan model. Default: models"
    )

    parser.add_argument(
        "--image_path",
        default=None,
        help="Path citra untuk mode classify atau retrieve."
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Jumlah hasil retrieval."
    )

    args = parser.parse_args()
    python_exe = sys.executable

    if args.mode == "validate":
        run_command([
            python_exe,
            "validate_dataset.py",
            "--dataset_dir",
            args.processed_dir
        ])

    elif args.mode == "preprocess":
        run_command([
            python_exe,
            "preprocess_dataset.py",
            "--input_dir",
            args.dataset_dir,
            "--output_dir",
            args.processed_dir
        ])

    elif args.mode == "vgg16":
        run_command([
            python_exe,
            "vgg16.py",
            "--dataset_dir",
            args.processed_dir,
            "--model_dir",
            args.model_dir
        ])

    elif args.mode == "split":
        run_command([
            python_exe,
            "split_dataset.py",
            "--model_dir",
            args.model_dir
        ])

    elif args.mode == "train":
        run_command([
            python_exe,
            "train_model.py",
            "--model_dir",
            args.model_dir
        ])

    elif args.mode == "classify":
        if args.image_path is None:
            raise ValueError("Mode classify membutuhkan --image_path")

        run_command([
            python_exe,
            "classify_image.py",
            "--image_path",
            args.image_path,
            "--model_dir",
            args.model_dir
        ])

    elif args.mode == "retrieve":
        if args.image_path is None:
            raise ValueError("Mode retrieve membutuhkan --image_path")

        run_command([
            python_exe,
            "retrieval.py",
            "--image_path",
            args.image_path,
            "--model_dir",
            args.model_dir,
            "--top_k",
            str(args.top_k)
        ])

    elif args.mode == "app":
        run_command([
            python_exe,
            "-m",
            "streamlit",
            "run",
            "app.py"
        ])

    elif args.mode == "pipeline":
        run_command([
            python_exe,
            "preprocess_dataset.py",
            "--input_dir",
            args.dataset_dir,
            "--output_dir",
            args.processed_dir
        ])

        run_command([
            python_exe,
            "validate_dataset.py",
            "--dataset_dir",
            args.processed_dir
        ])

        run_command([
            python_exe,
            "vgg16.py",
            "--dataset_dir",
            args.processed_dir,
            "--model_dir",
            args.model_dir
        ])

        run_command([
            python_exe,
            "split_dataset.py",
            "--model_dir",
            args.model_dir
        ])

        run_command([
            python_exe,
            "train_model.py",
            "--model_dir",
            args.model_dir
        ])


if __name__ == "__main__":
    main()