import json
import os
import shutil
import argparse
import datetime

import matplotlib.pyplot as plt

from src.training.train import EpochStats


def main():
    parser = argparse.ArgumentParser(description="Analyse training statistics")

    parser.add_argument(
        "--stats-path",
        type=str,
        required=True,
        help="The path to the training statistics file, outputted by the training script. Should be a jsonl file.",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="The folder to save the figures and analysis to.",
    )

    args = parser.parse_args()

    analyse(
        stats_path=args.stats_path,
        output_folder=args.output,
    )


def analyse(
    stats_path: str,
    output_folder: str,
    stats_filename: str = "stats.jsonl",
    losses_figure_filename: str = "losses.png",
    accuracies_figure_filename: str = "accuracies.png",
):
    """

    Analyse the training statistics and save figures to the output folder.
    Also copies the stats file to the output folder.

    """

    print(f"Analysing training statistics from {stats_path}")

    output_folder = output_folder + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    if os.path.exists(output_folder):
        raise ValueError(f"Output folder {output_folder} already exists")
    else:
        os.makedirs(output_folder)

    print("Saving analysis to", os.path.abspath(output_folder))

    stats = _load_stats_from_file(stats_path)

    stats_output_path = os.path.join(output_folder, stats_filename)

    print(f"Copying stats file to {output_folder}")
    shutil.copyfile(stats_path, stats_output_path)

    train_losses = [epoch["train"]["avg_loss"] for epoch in stats]
    validation_losses = [epoch["validation"]["avg_loss"] for epoch in stats]
    validation_accuracies = [epoch["validation"]["accuracy"] for epoch in stats]

    losses_figure_output_path = os.path.join(output_folder, losses_figure_filename)
    accuracies_figure_output_path = os.path.join(
        output_folder, accuracies_figure_filename
    )

    _plot_losses(train_losses, validation_losses, losses_figure_output_path)
    _plot_accuracies(validation_accuracies, accuracies_figure_output_path)


def _plot_losses(
    train_losses: list[float], validation_losses: list[float], output_path: str
):
    """

    Plots the training and validation losses and saves the figure to the output path.

    """

    print("Plotting losses and saving to", output_path)

    plt.figure(figsize=(10, 6))

    plt.plot(train_losses, label="Train loss")
    plt.plot(validation_losses, label="Validation loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation losses")

    plt.legend()

    plt.savefig(output_path)


def _plot_accuracies(validation_accuracies: list[float], output_path: str):
    """

    Plots the validation accuracies and saves the figure to the output path.

    """

    print("Plotting accuracies and saving to", output_path)

    plt.figure(figsize=(10, 6))

    plt.plot(validation_accuracies, label="Validation accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation accuracy")

    plt.legend()

    plt.savefig(output_path)


def _load_stats_from_file(filepath: str) -> list[EpochStats]:
    """

    Loads the saved epoch stats from the jsonl file.

    """

    stats = []

    with open(filepath, "r") as stats_file:
        for line in stats_file:
            epoch_stats = json.loads(line)
            stats.append(epoch_stats)

    return stats


if __name__ == "__main__":
    main()
