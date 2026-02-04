import subprocess
from pathlib import Path

DATASET = "nisargpatel344/student-course-completion-prediction-dataset"
DATA_DIR = Path("data")


def main():
    DATA_DIR.mkdir(exist_ok=True)

    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d", DATASET,
            "-p", str(DATA_DIR),
            "--unzip",
        ],
        check=True,
    )

    print("Done. Files downloaded to ./data/")


if __name__ == "__main__":
    main()
