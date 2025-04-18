import kagglehub
import shutil
import os


def download_dataset():
    destination = "dataset"
    os.makedirs(destination, exist_ok=True)
    path = kagglehub.dataset_download("natezhang123/social-anxiety-dataset", force_download=True)

    for file_name in os.listdir(path):
        shutil.move(os.path.join(path, file_name), os.path.join(destination, file_name))
