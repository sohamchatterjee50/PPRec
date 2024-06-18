from dataclasses import dataclass
import random

from torch.utils.data import Dataset
import torch

from .split import EBNeRDSplit, DatasetSize


@dataclass
class CandidateArticle:
    # needed for the LookupNewsEncoder
    article_id: int

    # the time since publication to prediction
    recency: int

    # click-through rate
    ctr: float

@dataclass
class ClickedArticle:
    article_id: int
    ctr: float


@dataclass
class TrainDataPoint:
    user_clicks: list[ClickedArticle]
    good_article: CandidateArticle
    bad_article: CandidateArticle


@dataclass
class Test:
    yoo: int


class EBNeRDTrainDataset(Dataset):

    def __init__(self, size: DatasetSize = "demo", data_folder: str | None = None):
        self.split = EBNeRDSplit(split="train", size=size, data_folder=data_folder)

    def __len__(self) -> int:
        return len(self.split._behaviors)

    def __getitem__(self, idx: int) -> TrainDataPoint:

        impression = self.split.get_behaviour_by_idx(idx)

        user_history = self.split.get_history(impression.user_id)
        user_history_articles = [self.split.get_article(article_id) for article_id in user_history.article_id_fixed]

        assert len(impression.article_ids_clicked) > 0
        good_article_id = random.choice(impression.article_ids_clicked)
        good_article = self.split.get_article(good_article_id)

        article_ids_not_clicked = [article_id for article_id in impression.article_ids_inview if article_id != good_article_id]
        assert len(article_ids_not_clicked) > 0
        bad_article_id = random.choice(article_ids_not_clicked)
        bad_article = self.split.get_article(bad_article_id)

        # Just have to find the recencies, and the click through rates
        raise NotImplementedError()
    

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

