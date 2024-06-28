from typing import Literal
from abc import ABC, abstractmethod

from torch import nn
import torch

from ..data.dataset import TrainDataPoint
from ..model.pprec import PPRec
from .utils import create_click_batch, create_positive_negative_candidate_batch

LossType = Literal["bpr", "cross_entropy"]


def loss_from_type(loss_type: LossType) -> "PPRecLoss":
    if loss_type == "bpr":
        return BPRPairwiseLoss()
    elif loss_type == "cross_entropy":
        return CrossEntropyLoss()
    else:
        raise ValueError(f"Loss function {loss_type} not supported")


class PPRecLoss(ABC, nn.Module):
    @abstractmethod
    def forward(self, outputs: PPRec.BatchPredictions) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def preprocess_train_batch(
        self, batch: list[TrainDataPoint], max_clicked: int
    ) -> "PPRec.Inputs":
        raise NotImplementedError

    @abstractmethod
    def correct_labels(self, batch_size: int) -> torch.Tensor:
        """

        Returns a tensor of shape (batch_size) with the correct labels for the batch.

        """

        raise NotImplementedError


class BPRPairwiseLoss(PPRecLoss):
    """

    The loss function as implemented in the original paper.

    """

    def __init__(self):
        super().__init__()

    def forward(self, outputs: PPRec.BatchPredictions) -> torch.Tensor:
        """

        Calculate the loss for the given batch.

        Uses the scores which are a tensor of shape (batch_size, candidate_size)
        where candidate size should be 2 for this loss function. The first candidate
        is a postive sample, and the second candidate is a negative sample.

        The loss function is defined as:

        L = - 1/D sum(log(sigmoid(score_positive - score_negative)))

        """

        assert outputs.score.shape[1] == 2

        differences = outputs.score[:, 0] - outputs.score[:, 1]
        activated_differences = torch.sigmoid(differences)
        log_activations = torch.log(activated_differences)
        loss = -torch.mean(log_activations)

        return loss

    def correct_labels(self, batch_size: int) -> torch.Tensor:
        """

        The correct labels belonging to the batch. Tensor of shape (batch_size).
        The index of the positive class.

        """

        return torch.tensor([0] * batch_size, dtype=torch.long)

    def preprocess_train_batch(
        self, batch: list[TrainDataPoint], max_clicked: int
    ) -> "PPRec.Inputs":
        """

        Preprocesses a batch of TrainDataPoints into the format needed for the forward
        function. It's concat the good and bad articles into one batch.

        """

        candidate_batch = create_positive_negative_candidate_batch(batch)
        clicks_batch = create_click_batch(batch, max_clicked=max_clicked)

        return PPRec.Inputs(candidates=candidate_batch, clicks=clicks_batch)


class CrossEntropyLoss(PPRecLoss):
    """

    The loss they use in the implementation, after taking
    a softmax over the candidate scores. Can be used with
    more than 2 candidates, but assumes the first candidate
    is the only positive one.

    I assume this loss is equal to the BPR loss, but I'm not sure.

    """

    def __init__(self):
        super().__init__()
        cel = nn.CrossEntropyLoss()

    def forward(self, outputs: PPRec.BatchPredictions) -> torch.Tensor:
        """

        Calculated the cross entropy loss for the given batch.

        """

        logits = torch.softmax(outputs.score, dim=1)
        loss = self.cel(logits, self.correct_labels(outputs.score.shape[0]))

        return loss

    def correct_labels(self, batch_size: int) -> torch.Tensor:
        """

        The correct labels belonging to the batch. Tensor of shape (batch_size).
        The index of the positive class.

        """

        return torch.tensor([0] * batch_size, dtype=torch.long)

    def preprocess_train_batch(
        self, batch: list[TrainDataPoint], max_clicked: int
    ) -> "PPRec.Inputs":
        """

        Preprocesses a batch of TrainDataPoints into the format needed for the forward
        function. It's concat the good and bad articles into one batch.

        """

        candidate_batch = create_positive_negative_candidate_batch(batch)
        clicks_batch = create_click_batch(batch, max_clicked=max_clicked)

        return PPRec.Inputs(candidates=candidate_batch, clicks=clicks_batch)
