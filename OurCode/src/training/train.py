from typing import TypedDict
from dataclasses import dataclass
import os
import json
from datetime import datetime
from pprint import pprint

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from tqdm import tqdm

from ..data.split import DatasetSize, EBNeRDSplit
from ..data.dataset import EBNeRDTrainDataset
from ..model.pprec import PPRec

from .loss import loss_from_type, LossType, PPRecLoss
from .stats import (
    TrainStats,
    ValidationStats,
    init_train_stats,
    update_train_stats,
    init_validation_stats,
    update_validation_stats,
)


@dataclass
class TrainConfig:
    # they use 32
    batch_size: int

    # They use a 0.0001 learning rate
    lr: float

    # they use the BPR loss
    criterion: LossType


class EpochStats(TypedDict):
    epoch: int
    train: "TrainStats"
    validation: "ValidationStats"


def train(
    model: PPRec,
    device: torch.device,
    # the folder the best model is saved to and where
    # the statistics are saved to in jsonl format.
    folder: str,
    description: str,
    dataset_size: DatasetSize,
    max_epochs: int,
    data_folder: str | None,
    config: TrainConfig,
    console_log: bool,
    stats_filename: str = "train_stats.jsonl",
    best_model_filename: str = "best_model.pt",
    description_filename: str = "description.txt",
):
    # assuming the model uses the lookup encoder, check if devices match
    assert model.user_news_encoder.device == device
    assert model.popularity_news_encoder.device == device

    optimizer = Adam(model.parameters(), lr=config.lr)
    criterion = loss_from_type(config.criterion)

    train_split = EBNeRDSplit(split="train", size=dataset_size, data_folder=data_folder)
    train_dataset = EBNeRDTrainDataset(train_split)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )

    validation_split = EBNeRDSplit(
        split="val", size=dataset_size, data_folder=data_folder
    )
    validation_dataset = EBNeRDTrainDataset(validation_split)
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=validation_dataset.collate_fn,
    )

    model.to(device)

    # ensure always a new folder
    folder = folder + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        raise ValueError(f"Folder {folder} already exists")

    best_model_filepath = os.path.join(folder, best_model_filename)
    stats_filepath = os.path.join(folder, stats_filename)
    description_filepath = os.path.join(folder, description_filename)

    with open(description_filepath, "w") as description_file:
        description_file.write(description)

    best_validation_stats: ValidationStats | None = None

    if console_log:
        n_params = sum(p.numel() for p in model.parameters())

        print("Starting training")
        print("\n")

        print("Train config:", end=" ")
        pprint(config)
        print("\n")

        print(f"Description: {description}")
        print(f"Data size: {dataset_size}")
        print(f"Max epochs: {max_epochs}")
        print(f"Device: {device}")
        print(f"Data folder: {data_folder}")
        print(f"Number of parameters: {n_params}")
        print("\n")

        print(f"Saving best model to {best_model_filepath}")
        print(f"Saving stats to {stats_filepath}")
        print(f"Saved description to {description_filepath}")
        print("\n")

    for epoch in range(max_epochs):

        if console_log:
            print(f"Epoch {epoch + 1}/{max_epochs}")
            print("\n")

        train_stats = _train_epoch(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_dataloader=train_dataloader,
            device=device,
            console_log=console_log,
        )

        if console_log:
            print("Train stats:")
            pprint(train_stats)
            print("\n")

        validation_stats = _validate_epoch(
            model=model,
            criterion=criterion,
            validation_dataloader=validation_dataloader,
            device=device,
            console_log=console_log,
        )

        if console_log:
            print("Validation stats:")
            pprint(validation_stats)
            print("\n")

        epoch_stats = EpochStats(
            train=train_stats, validation=validation_stats, epoch=epoch
        )

        if console_log:
            print(f"Saving epoch stats to {stats_filepath}")

        with open(stats_filepath, "a") as stats_file:
            stats_file.write(json.dumps(epoch_stats))
            stats_file.write("\n")

        best_validation_stats = _save_model_if_better(
            model=model,
            best_model_filepath=best_model_filepath,
            validation_stats=validation_stats,
            best_validation_stats=best_validation_stats,
            console_log=console_log,
        )


def _save_model_if_better(
    model: PPRec,
    best_model_filepath: str,
    validation_stats: ValidationStats,
    best_validation_stats: ValidationStats | None,
    console_log: bool,
) -> ValidationStats:

    if (
        best_validation_stats is None
        or validation_stats["avg_loss"] < best_validation_stats["avg_loss"]
    ):

        if console_log:
            print(f"New best model found! Saving to {best_model_filepath}")
            print("\n")

        torch.save(model.state_dict(), best_model_filepath)
        return validation_stats

    return best_validation_stats


def _train_epoch(
    model: PPRec,
    optimizer: Adam,
    criterion: PPRecLoss,
    train_dataloader: DataLoader,
    device: torch.device,
    console_log: bool,
):
    model.train()

    train_stats = init_train_stats()

    if console_log:
        train_dataloader_ = tqdm(train_dataloader, desc="Training")
    else:
        train_dataloader_ = train_dataloader

    for batch in train_dataloader_:
        inputs = criterion.preprocess_train_batch(
            batch=batch, max_clicked=model.max_clicked
        )
        inputs.to_device(device)

        optimizer.zero_grad()
        predictions: PPRec.BatchPredictions = model(inputs)

        loss = criterion(predictions)
        loss.backward()
        optimizer.step()

        train_stats = update_train_stats(
            train_stats=train_stats,
            loss=loss.item(),
        )

    return train_stats


def _validate_epoch(
    model: PPRec,
    criterion: PPRecLoss,
    validation_dataloader: DataLoader,
    device: torch.device,
    console_log: bool,
) -> ValidationStats:
    model.eval()

    validation_stats = init_validation_stats()

    if console_log:
        validation_dataloader_ = tqdm(validation_dataloader, desc="Validating")
    else:
        validation_dataloader_ = validation_dataloader

    with torch.no_grad():
        for batch in validation_dataloader_:
            inputs = criterion.preprocess_train_batch(
                batch=batch, max_clicked=model.max_clicked
            )
            inputs.to_device(device)

            predictions: PPRec.BatchPredictions = model(inputs)

            loss = criterion(predictions)

            validation_stats = update_validation_stats(
                validation_stats=validation_stats,
                loss=loss.item(),
                predictions=predictions,
                labels=criterion.correct_labels(batch_size=len(batch)),
            )

    return validation_stats
