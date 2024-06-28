import argparse
from typing import get_args

import torch

from src.training.train import train
from src.training.configs import TrainConfigType, train_config_from_type

from src.model.pprec import PPRec
from src.model.configs import ModelConfigType, config_from_type

from src.data.split import DatasetSize


def main():
    parser = argparse.ArgumentParser(description="Train a new PPRec model")

    parser.add_argument(
        "--data-folder",
        type=str,
        required=True,
        help="The folder containing the data",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="The folder to save the model and statistics to",
    )

    parser.add_argument(
        "--description",
        type=str,
        required=True,
        help="A short description of the experiment, the model, and the training session",
    )

    parser.add_argument(
        "--model-config",
        type=str,
        required=True,
        default="minimal",
        choices=get_args(ModelConfigType),
        help="The configuration of the model",
    )

    parser.add_argument(
        "--data-size",
        type=str,
        required=True,
        default="mini",
        choices=get_args(DatasetSize),
        help="The size of the dataset",
    )

    parser.add_argument(
        "--max-epochs",
        type=int,
        required=True,
        default=10,
        help="The maximum number of epochs to train for",
    )

    parser.add_argument(
        "--train-config",
        type=str,
        required=True,
        default="paper",
        choices=get_args(TrainConfigType),
        help="The training configuration to use",
    )

    parser.add_argument(
        "--device",
        type=str,
        required=True,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="The device to train on",
    )

    parser.add_argument(
        "--max-clicked",
        type=int,
        required=True,
        default=50,
    )

    parser.add_argument(
        "--log",
        action="store_true",
        help="Log to the console",
    )

    args = parser.parse_args()

    model_config = config_from_type(args.model_config)
    train_config = train_config_from_type(args.train_config)

    device = torch.device(args.device)

    model = PPRec(
        config=model_config,
        device=device,
        max_clicked=args.max_clicked,
        data_folder=args.data_folder,
    )

    train(
        model=model,
        device=device,
        folder=args.output,
        description=args.description,
        dataset_size=args.data_size,
        max_epochs=args.max_epochs,
        data_folder=args.data_folder,
        config=train_config,
        console_log=args.log,
    )


if __name__ == "__main__":
    main()
