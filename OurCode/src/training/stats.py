from typing import TypedDict

import torch

from ..model.pprec import PPRec


class TrainStats(TypedDict):
    avg_loss: float
    n_batches: int


def init_train_stats() -> TrainStats:
    return {"avg_loss": 0.0, "n_batches": 0}


def update_train_stats(train_stats: TrainStats, loss: float) -> TrainStats:
    """

    Updates the train stats after one batch. Keeps a running average of the loss.

    """

    n_batches = train_stats["n_batches"]
    avg_loss = train_stats["avg_loss"]

    new_n_batches = n_batches + 1

    avg_loss = _update_loss(
        loss=loss,
        avg_loss=avg_loss,
        old_n_batches=n_batches,
        new_n_batches=new_n_batches,
    )

    return {"avg_loss": avg_loss, "n_batches": new_n_batches}


class ValidationStats(TypedDict):
    avg_loss: float
    accuracy: float
    # precision: float
    # recall: float
    # auc: float
    # mrr: float
    # ndgc5: float
    # ndgc10: float
    n_samples: int
    n_batches: int


def init_validation_stats() -> ValidationStats:
    return {
        "avg_loss": 0.0,
        "accuracy": 0.0,
        # "precision": 0.0,
        # "recall": 0.0,
        # "auc": 0.0,
        # "mrr": 0.0,
        # "ndgc5": 0.0,
        # "ndgc10": 0.0,
        "n_samples": 0,
        "n_batches": 0,
    }


def update_validation_stats(
    validation_stats: ValidationStats,
    loss: float,
    predictions: PPRec.BatchPredictions,
    labels: torch.Tensor,
) -> ValidationStats:
    """

    Updates the validation stats after one batch. Keeps track of running values, that
    can be updates after each batch.

    """

    predictions.to_device(torch.device("cpu"))

    n_samples = validation_stats["n_samples"]
    n_batches = validation_stats["n_batches"]

    new_n_samples = n_samples + labels.shape[0]
    new_n_batches = n_batches + 1

    avg_loss = _update_loss(
        loss=loss,
        avg_loss=validation_stats["avg_loss"],
        old_n_batches=n_batches,
        new_n_batches=new_n_batches,
    )

    accuracy = _update_accuracy(
        predictions=predictions,
        labels=labels,
        old_accuracy=validation_stats["accuracy"],
        old_n_samples=n_samples,
        new_n_samples=new_n_samples,
    )

    return {
        "avg_loss": avg_loss,
        "accuracy": accuracy,
        "n_samples": new_n_samples,
        "n_batches": new_n_batches,
    }


def _update_loss(
    loss: float, avg_loss: float, old_n_batches: int, new_n_batches: int
) -> float:

    new_loss = (avg_loss * old_n_batches + loss) / new_n_batches

    return new_loss


def _update_accuracy(
    predictions: PPRec.BatchPredictions,
    labels: torch.Tensor,
    old_accuracy: float,
    old_n_samples: int,
    new_n_samples: int,
) -> float:
    classification = torch.argmax(predictions.score, dim=1)
    correct = torch.sum(classification == labels).item()
    return (old_accuracy * old_n_samples + correct) / new_n_samples
