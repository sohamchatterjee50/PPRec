import os
from typing import Literal

from torch.utils.data import Dataset
import pandas as pd

DATA_FOLDER = os.environ.get("PPREC_DATA_FOLDER")


class EBNeRDSplit:
    """

    This class loads the EBNeRD dataset for a specific split and size.
    The resulting object contains three dataframes: articles, behaviors, and history.

    """

    def __init__(
        self,
        split: Literal["train", "val"] = "train",
        size: Literal["demo", "small", "large"] = "demo",
        data_folder: str | None = None,
    ):
        """

        Creates an instance and loads all dataframes into memory.

        split: str
            The split of the dataset to load. Either "train" or "val".

        size: str
            The size of the dataset to load. Either "demo", "small", or "large".

        data_folder: str, optional
            The path to the folder containing the dataset. If not provided, the 
            path is read from the environment variable 'PPREC_DATA_FOLDER'.
        
        """

        if data_folder is not None:
            pass
        elif DATA_FOLDER is not None:
            data_folder = DATA_FOLDER
        else:
            raise ValueError(
                "data_folder must be provided as an argument or set as an environment variable 'PPREC_DATA_FOLDER'"
            )

        match size:
            case "demo":
                dataset_folder = os.path.join(data_folder, "ebnerd_demo")
            case "small":
                dataset_folder = os.path.join(data_folder, "ebnerd_small")
            case "large":
                dataset_folder = os.path.join(data_folder, "ebnerd_large")

        articles_parquet_path = os.path.join(dataset_folder, "articles.parquet")

        match split:
            case "train":
                split_folder = os.path.join(dataset_folder, "train")
            case "val":
                split_folder = os.path.join(dataset_folder, "val")

        behaviors_parquet_path = os.path.join(split_folder, "behaviors.parquet")
        history_parquet_path = os.path.join(split_folder, "history.parquet")

        self.articles = pd.read_parquet(articles_parquet_path)
        self.behaviors = pd.read_parquet(behaviors_parquet_path)
        self.history = pd.read_parquet(history_parquet_path)

    def summarize(self, show_head: bool = False, show_columns: bool = False):
        """
        
        Prints a summary of the articles, behaviors, and history dataframes.

        show_head: bool, optional
            If True, prints the first few rows of each dataframe.
        
        show_columns: bool, optional
            If True, prints the columns of each dataframe.
        
        """

        print(f"Articles: {self.articles.shape}")
        if show_columns:
            print("Columns Articles: ", self.articles.columns)
        if show_head:
            print(self.articles.head())

        print("\n")

        print(f"Behaviors: {self.behaviors.shape}")
        if show_columns:
            print("Columns Behaviors: ", self.behaviors.columns)
        if show_head:
            print(self.behaviors.head())

        print("\n")

        print(f"History: {self.history.shape}")
        if show_columns:
            print("Columns History: ", self.history.columns)
        if show_head:
            print(self.history.head())
