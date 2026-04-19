from datasets import load_dataset
import pandas as pd


class Dataset:
    def __init__(self):
        self.DATASET_PATH = "data/dataset/"

    def dowload_dataset(self) -> pd.DataFrame:
        ds = load_dataset("tooni/pokemoncards", data_files="cards.csv", split="train")
        ds.to_csv(self.DATASET_PATH + "cards.csv", index=False)


if __name__ == "__main__":
    dataset = Dataset()
    df = dataset.dowload_dataset()
