import subprocess
from pathlib import Path

DATASETS = [
    "nisargpatel344/student-course-completion-prediction-dataset",
    "prince7489/online-learning-and-course-consumption-dataset",
    "mitul1999/online-courses-usage-and-history-dataset",
]

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def download_dataset(dataset: str):
    print(f"\nDownloading: {dataset}")
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d", dataset,
            "-p", str(DATA_DIR),
            "--unzip",
        ],
        check=True,
    )


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for ds in DATASETS:
        download_dataset(ds)

    print("\nAll datasets downloaded successfully.")
    print(f"Files saved in: {DATA_DIR}")


if __name__ == "__main__":
    main()