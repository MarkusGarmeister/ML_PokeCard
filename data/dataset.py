from datasets import load_dataset
import pandas as pd
import requests
import os


class Dataset:
    def __init__(self):
        self.DATASET_PATH = "data/dataset/"
        self.IMAGE_PATH = "data/dataset/img/"

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

    def download_images(self, df: pd.DataFrame):
        for index, row in df.iterrows():
            image_url = row["small_image_source"]
            image_url = image_url.strip('"')
            print(image_url)
            image_name = f"{row['id']}.png"
            image_path = self.IMAGE_PATH + image_name
            try:
                response = requests.get(image_url)
                response.raise_for_status()
                with open(image_path, "wb") as f:
                    f.write(response.content)
                print(f"Downloaded {image_name}")

            except requests.exceptions.RequestException as e:
                print(f"Failed to download {image_name}: {e}")


if __name__ == "__main__":
    dataset = Dataset()
    df = dataset.dowload_dataset()
    dataset.download_images(df)
