from datetime import datetime
import os
from typing import Literal
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pydantic

from ..utils import get_data_folder


DatasetSize = Literal["demo", "small", "large"]
DatasetSplit = Literal["train", "val"]


class Article(pydantic.BaseModel):
    article_id: int
    title: str
    subtitle: str
    last_modified_time: datetime
    premium: bool
    body: str
    published_time: datetime
    image_ids: list[int] | None
    article_type: str
    url: str
    ner_clusters: list[str]
    entity_groups: list[str]
    topics: list[str]
    category: int
    subcategory: list[int]
    category_str: str
    total_inviews: int | None
    total_pageviews: int | None
    total_read_time: int | None
    sentiment_score: float
    sentiment_label: str


class Behavior(pydantic.BaseModel):
    impression_id: int
    article_id: int | None
    impression_time: datetime
    read_time: int
    scroll_percentage: float
    device_type: int
    article_ids_inview: list[int]
    article_ids_clicked: list[int]
    user_id: int
    is_sso_user: bool
    gender: int | None
    postcode: int | None
    age: int | None
    is_subscriber: bool
    session_id: int
    next_read_time: int | None
    next_scroll_percentage: float | None


class History(pydantic.BaseModel):
    user_id: int
    impression_time_fixed: list[datetime]
    scroll_percentage_fixed: list[float | None]
    article_id_fixed: list[int]
    read_time_fixed: list[float]


class EBNeRDSplit:
    """

    This class loads the EBNeRD dataset for a specific split and size.
    The resulting object contains three dataframes: articles, behaviors, and history.

    """

    def __init__(
        self,
        split: DatasetSplit = "train",
        size: DatasetSize = "demo",
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

        data_folder = get_data_folder(data_folder)

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
                split_folder = os.path.join(dataset_folder, "validation")

        behaviors_parquet_path = os.path.join(split_folder, "behaviors.parquet")
        history_parquet_path = os.path.join(split_folder, "history.parquet")

        self.articles = pd.read_parquet(articles_parquet_path)
        self.behaviors = pd.read_parquet(behaviors_parquet_path)
        self.history = pd.read_parquet(history_parquet_path)

        # In the articlestotal_inviews, total_pageviews and total_read_time are supposed
        # to be integers looking at the data, but they are floats in the parquet file.
        # Note that I had to use the Int64 type, because it contains NaN values.
        self.articles["total_inviews"] = self.articles["total_inviews"].astype("Int64")
        self.articles["total_pageviews"] = self.articles["total_pageviews"].astype(
            "Int64"
        )
        self.articles["total_read_time"] = self.articles["total_read_time"].astype(
            "Int64"
        )

        # Same for the article id, postcode, gender age and next_read_time in the behaviors dataframe
        self.behaviors["article_id"] = self.behaviors["article_id"].astype("Int64")
        self.behaviors["postcode"] = self.behaviors["postcode"].astype("Int64")
        self.behaviors["gender"] = self.behaviors["gender"].astype("Int64")
        self.behaviors["age"] = self.behaviors["age"].astype("Int64")
        self.behaviors["next_read_time"] = self.behaviors["next_read_time"].astype(
            "Int64"
        )

        self.articles.set_index("article_id", inplace=True, drop=False)
        self.behaviors.set_index("impression_id", inplace=True, drop=False)
        self.history.set_index("user_id", inplace=True, drop=False)

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

    def get_random_article_id(self) -> int:
        """

        Returns a random article id.

        """

        return self.articles.sample().index[0]

    def get_random_impression_id(self) -> int:
        """

        Returns a random impression id.

        """

        return self.behaviors.sample().index[0]

    def get_random_user_id(self) -> int:
        """

        Returns a random user id.

        """

        return self.history.sample().index[0]

    def get_random_article(self) -> Article:
        """

        Returns a random article.

        """

        article_id = self.get_random_article_id()
        return self.get_article(article_id)

    def get_random_behavior(self) -> Behavior:
        """

        Returns a random behavior.

        """

        impression_id = self.get_random_impression_id()
        return self.get_behavior(impression_id)

    def get_random_history(self) -> History:
        """

        Returns a random history.

        """

        user_id = self.get_random_user_id()
        return self.get_history(user_id)

    def get_article(self, article_id: int) -> Article:
        """

        Returns the article of the given article_id.

        """

        article = self.articles.loc[article_id].to_dict()
        return Article.model_validate(article)

    def get_behavior(self, impression_id: int) -> Behavior:
        """

        Returns the behavior of the given impression_id.

        """

        behavior = self.behaviors.loc[impression_id].to_dict()
        return Behavior.model_validate(behavior)

    def get_history(self, user_id: int) -> History:
        """

        Returns the history of the given user_id.

        """

        history = self.history.loc[user_id].to_dict()

        # impression_time_fixed are np.datetime64 objects, which don't work with pydantic
        # so we convert them to datetime objects
        history["impression_time_fixed"] = [
            pd.Timestamp(date).to_pydatetime() for date in history["impression_time_fixed"]
        ]

        return History.model_validate(history)
