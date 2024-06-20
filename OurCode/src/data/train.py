from dataclasses import dataclass
import random
from datetime import datetime

from torch.utils.data import Dataset
import torch

from .split import EBNeRDSplit, DatasetSize, Article, Behavior


@dataclass
class CandidateArticle:
    # needed for the LookupNewsEncoder
    article_id: int

    # the time since publication to prediction, in hours
    recency: float

    # click-through rate
    ctr: float


@dataclass
class ClickedArticle:
    article_id: int
    ctr: float

    # used for determining the most recent clicks
    clicktime: datetime


@dataclass
class TrainDataPoint:
    user_clicks: list[ClickedArticle]
    good_article: CandidateArticle
    bad_article: CandidateArticle


@dataclass
class Test:
    yoo: int


class EBNeRDTrainDataset(Dataset):

    def __init__(
        self,
        size: DatasetSize = "demo",
        data_folder: str | None = None,
        ctr_delta: float = 0.001,
    ):
        self.split = EBNeRDSplit(split="train", size=size, data_folder=data_folder)
        self.ctr_delta = ctr_delta

    def __len__(self) -> int:
        return len(self.split._behaviors)

    def __getitem__(self, idx: int) -> TrainDataPoint:

        impression = self.split.get_behaviour_by_idx(idx)

        user_history = self.split.get_history(impression.user_id)
        user_history_articles = [
            self.split.get_article(article_id)
            for article_id in user_history.article_id_fixed
        ]

        assert len(impression.article_ids_clicked) > 0
        good_article_id = random.choice(impression.article_ids_clicked)
        good_article = self.split.get_article(good_article_id)

        article_ids_not_clicked = [
            article_id
            for article_id in impression.article_ids_inview
            if article_id != good_article_id
        ]
        assert len(article_ids_not_clicked) > 0
        bad_article_id = random.choice(article_ids_not_clicked)
        bad_article = self.split.get_article(bad_article_id)

        datapoint = TrainDataPoint(
            user_clicks=[
                ClickedArticle(
                    article_id=article.article_id,
                    ctr=self.click_through_rate(article),
                    clicktime=clicktime,
                )
                for article, clicktime in zip(
                    user_history_articles, user_history.impression_time_fixed
                )
            ],
            good_article=CandidateArticle(
                article_id=good_article.article_id,
                recency=self.recency(good_article, impression),
                ctr=self.click_through_rate(good_article),
            ),
            bad_article=CandidateArticle(
                article_id=bad_article.article_id,
                recency=self.recency(bad_article, impression),
                ctr=self.click_through_rate(bad_article),
            ),
        )

        return datapoint

    def click_through_rate(self, article: Article) -> float:
        """

        Calculate the click through rate of an article.

        """

        if article.total_pageviews is None:
            total_pageviews = 0.0
        else:
            total_pageviews = article.total_pageviews

        if article.total_inviews is None:
            total_inviews = 0.0
        else:
            total_inviews = article.total_inviews

        return total_pageviews / (total_inviews + self.ctr_delta)

    def recency(self, article: Article, impression: Behavior) -> float:
        """

        Calculate the recency of an article. The time since publication to prediction, in hours.

        """

        return (impression.impression_time - article.published_time).seconds / 3600

    def collate_fn(self, batch: list[TrainDataPoint]) -> list[TrainDataPoint]:
        """

        The collate fn to be used with this dataset. It just return the batch as is.
        So when defining a dataloader take this as an argument.

        Like

        dataloader = DataLoader(
            dataset,
            batch_size=64,
            collate_fn=dataset.collate_fn
        )

        """

        return batch
