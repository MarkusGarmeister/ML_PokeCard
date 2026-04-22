from datasets import load_dataset
import pandas as pd
import requests
import os
from PIL import Image
from io import BytesIO
from data.rarity_classes import RARITY_CLASSES, RARITY_MAPPING
import numpy as np

IMAGE_SIZE = (120, 168)  # width x height
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Dataset:
    def __init__(self):
        self.DATASET_PATH = os.path.join(BASE_DIR, "data", "dataset") + "/"
        self.IMAGE_PATH = os.path.join(BASE_DIR, "data", "dataset", "img") + "/"

    def dowload_dataset(self) -> pd.DataFrame:
        ds = load_dataset("tooni/pokemoncards", data_files="cards.csv", split="train")
        df = pd.DataFrame(ds)
        return df

    def get_dataset(self) -> pd.DataFrame:
        if not os.path.exists(self.DATASET_PATH + "cards.csv"):
            df = self.dowload_dataset()
            return df
        else:
            df = pd.read_csv(self.DATASET_PATH + "cards.csv")
            return df

    def download_image(self, url):
        try:
            img_url = url.strip('"')
            response = requests.get(img_url, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            img = img.convert("RGB")
            return img
        except Exception:
            print(f"Failed to download image from {img_url}")
            return None

    def download_images(self, df: pd.DataFrame):
        x_images = []
        y_labels = []
        failed = 0

        for index, row in df.iterrows():
            image_url = row["small_image_source"]
            image_url = image_url.strip('"')
            print(image_url)
            image_name = f"{row['id']}.png"
            image_path = self.IMAGE_PATH + image_name
            try:
                response = requests.get(image_url)
                response.raise_for_status()
                img = Image.open(BytesIO(response.content))
                img = img.convert("RGB")
                img.save(image_path)

                img_resized = img.resize(IMAGE_SIZE)
                x_images.append(np.array(img_resized))
                y_labels.append(row["mapped_rarity"])

            except requests.exceptions.RequestException as e:
                print(f"Failed to download {image_name}: {e}")
                failed += 1

        return x_images, y_labels, failed

    def map_rarity(self, rarity):
        return RARITY_MAPPING.get(rarity, None)

    def get_mapped_dataset(self) -> pd.DataFrame:
        df = self.get_dataset()
        df["mapped_rarity"] = df["rarity"].apply(self.map_rarity)
        return df


if __name__ == "__main__":
    dataset = Dataset()
    df = dataset.dowload_dataset()
    dataset.download_images(df)
