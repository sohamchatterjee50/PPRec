from typing import get_args
from argparse import ArgumentParser

import numpy as np
import torch

from src.model.news_encoder import LookupNewsEncoder, TextEncodeModel, NEConfig
from src.data.split import EBNeRDSplit, DatasetSize, DatasetSplit


def main():

    parser = ArgumentParser(
        description="Testing script for checking whether everything works as expected."
    )

    parser.add_argument(
        "--hard", help="Better tests, but takes longer.", action="store_true"
    )

    args = parser.parse_args()

    print("Loading encoders")

    news_encoder_config = NEConfig()

    lookup_news_encoders = [
        LookupNewsEncoder(model=model, config=news_encoder_config)
        for model in get_args(TextEncodeModel)
    ]

    print("Loading datasets")

    ebnerd_splits = [
        EBNeRDSplit(split=split, size=size)
        for split in get_args(DatasetSplit)
        for size in get_args(DatasetSize)
        if _should_include_split(size=size, hard_testing=args.hard)
    ]

    assert len(ebnerd_splits) > 0
    assert len(lookup_news_encoders) > 0

    print("Testing dataset tools")

    for split in ebnerd_splits:
        _test_ebnerd_split(split)

    print("Testing news encoders")

    for news_encoder in lookup_news_encoders:
        _test_news_encoder(news_encoder, ebnerd_splits)


def _test_news_encoder(
    news_encoder: LookupNewsEncoder,
    ebnerd_splits: list[EBNeRDSplit],
    n_articles_per_split: int = 50,
    batch_size: int = 64,
) -> None:

    print(f"Testing LookupNewsEncoder with model {news_encoder.model}")

    random_articles = []

    for split in ebnerd_splits:
        random_articles_in_split = [
            split.get_random_article() for _ in range(n_articles_per_split)
        ]
        random_articles.extend(random_articles_in_split)

    print("Testing LookupNewsEncoder.get_embeddings")

    for article in random_articles:
        embeddings = news_encoder.get_embeddings(article.article_id)

        assert isinstance(embeddings, np.ndarray)
        assert len(embeddings.shape) == 1
        assert embeddings.shape[0] == 768 or embeddings.shape[0] == 300

    print("Testing LookupNewsEncoder.get_embeddings_batch")

    article_ids = [article.article_id for article in random_articles]

    article_ids_batched = [
        article_ids[i : i + batch_size] for i in range(0, len(article_ids), batch_size)
    ]

    for article_ids_batch in article_ids_batched:
        embeddings_batch = news_encoder.get_embeddings_batch(article_ids_batch)

        assert isinstance(embeddings_batch, np.ndarray)
        assert len(embeddings_batch.shape) == 2
        assert embeddings_batch.shape[1] == 768 or embeddings_batch.shape[1] == 300
        assert embeddings_batch.shape[0] == len(article_ids_batch)

    print("Testing LookupNewsEncoder.forward")

    for article_ids_batch in article_ids_batched:
        embeddings_batch = news_encoder.get_embeddings_batch(article_ids_batch)
        n = news_encoder.forward(article_ids_batch)

        assert isinstance(n, torch.Tensor)
        assert len(n.shape) == 2
        assert n.shape[1] == news_encoder.config.get_size_n()
        assert n.shape[0] == len(article_ids_batch)


def _test_ebnerd_split(split: EBNeRDSplit, n: int = 50) -> None:

    print(f"Testing EBNeRDSplit with split {split.split} and size {split.size}")

    assert len(split._articles) > 0
    assert len(split._behaviors) > 0
    assert len(split._history) > 0

    # these test the functionality of get_article, get_behavior, and get_history
    # using pydantic to validate the data
    for _ in range(n):
        split.get_random_article()
        split.get_random_behavior()
        split.get_random_history()


def _should_include_split(size: DatasetSize, hard_testing: bool) -> bool:

    if hard_testing:
        return True

    if size == "large" or size == "small":
        return False

    return True


if __name__ == "__main__":
    main()
