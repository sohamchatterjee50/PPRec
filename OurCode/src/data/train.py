from dataclasses import dataclass

from torch.utils.data import Dataset
import torch

from .split import EBNeRDSplit, DatasetSize


@dataclass
class TrainArticle:
    # something like this, dont know what we need exactly

    title: str
    body: str
    text: str
    article_id: int
    crt: float
    publish_time: int
    impression_time: int

    # we can find these in the artifacts
    # for a given embedding model and article id
    text_embeddings: list[float]


@dataclass
class TrainUserClick:
    impression_id: int
    article_id: int
    article: TrainArticle


@dataclass
class TrainUserHistory:
    # something like this

    user_id: int
    clicks: list[TrainUserClick]
    user_encoder_input: torch.Tensor


@dataclass
class TrainDataPoint:
    user_behavior: TrainUserHistory
    good_article: TrainArticle
    bad_article: TrainArticle


class EBNeRDTrainDataset(Dataset):

    def __init__(self, size: DatasetSize = "demo", data_folder: str | None = None):
        self.split = EBNeRDSplit(split="train", size=size, data_folder=data_folder)

    def __len__(self) -> int:
        return len(self.split.behaviors)

    def __get__(self, user_id: int) -> TrainDataPoint:

        # Looking at the loss function, every i in D whould be
        # a pair of a positive and a negative article, for a given
        # user. So i'd say this is the starting point:
        user_history = self.split.history.loc[user_id]

        # So we output some user data/history, and one good
        # and one bad article. And for these articles
        raise NotImplementedError()
