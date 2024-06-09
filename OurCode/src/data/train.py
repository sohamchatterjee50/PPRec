from dataclasses import dataclass

from torch.utils.data import Dataset
import torch

from .split import EBNeRDSplit, DatasetSize


@dataclass
class TrainArticle:
    # needed for the LookupNewsEncoder
    article_id: int

    # going to be needed when we have an actual news encoder
    # news_encoder_input: torch.Tensor

    # the time since publication to prediction
    recency: int

    # click-through rate
    ctr: float


@dataclass
class TrainDataPoint:
    user_encoder_input: torch.Tensor
    good_article: TrainArticle
    bad_article: TrainArticle


class EBNeRDTrainDataset(Dataset):

    def __init__(self, size: DatasetSize = "demo", data_folder: str | None = None):
        self.split = EBNeRDSplit(split="train", size=size, data_folder=data_folder)

    def __len__(self) -> int:
        return len(self.split._behaviors)

    def __get__(self, user_id: int) -> TrainDataPoint:

        # Looking at the loss function, every i in D whould be
        # a pair of a positive and a negative article, for a given
        # user. So i'd say this is the starting point:
        user_history = self.split.get_history(user_id)

        # So we output some user data/history, and one good
        # and one bad article. And for these articles
        raise NotImplementedError()
