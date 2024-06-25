import random

import numpy as np
import torch

from ..data.dataset import TrainDataPoint
from ..model.pprec import PPRec


def create_positive_negative_candidate_batch(
    batch: list[TrainDataPoint],
) -> "PPRec.CandidateBatch":

    batch_size = len(batch)
    candidate_size = 2

    good_articles = [random.choice(point.good_articles) for point in batch]
    bad_articles = [random.choice(point.bad_articles) for point in batch]

    candidate_ids = np.array(
        [
            [article.article_id for article in good_articles],
            [article.article_id for article in bad_articles],
        ]
    )
    candidate_ids = candidate_ids.transpose(1, 0)

    candidate_ctr = torch.tensor(
        [
            [article.ctr for article in good_articles],
            [article.ctr for article in bad_articles],
        ],
        dtype=torch.float32,
    )
    candidate_ctr = candidate_ctr.transpose(1, 0)

    candidate_recencies = torch.tensor(
        [
            [article.recency for article in good_articles],
            [article.recency for article in bad_articles],
        ],
        dtype=torch.float32,
    )
    candidate_recencies = candidate_recencies.transpose(1, 0)

    assert candidate_ids.shape == (batch_size, candidate_size)
    assert candidate_ctr.shape == (batch_size, candidate_size)
    assert candidate_recencies.shape == (batch_size, candidate_size)

    candidate_batch = PPRec.CandidateBatch(
        ids=candidate_ids, ctr=candidate_ctr, recencies=candidate_recencies
    )

    return candidate_batch


def create_click_batch(
    batch: list[TrainDataPoint], max_clicked: int
) -> "PPRec.ClicksBatch":
    """

    Creates the input data for the click history of the user.
    It takes the most recent max_clicked articles the user has clicked on.
    If a user has clicked on less articles, it pads with 0.

    The ctr is a tensor of shape (batch_size, max_clicked)
    The ids is a numpy array of shape (batch_size, max_clicked)

    """

    batch_size = len(batch)

    clicked_ids = []
    clicked_ctr = []

    for point in batch:
        clicks_sorted_on_recency = sorted(
            point.user_clicks, key=lambda article: article.clicktime, reverse=True
        )

        if len(clicks_sorted_on_recency) > max_clicked:
            most_recent_clicks = clicks_sorted_on_recency[:max_clicked]
            ids = np.array([article.article_id for article in most_recent_clicks])
            ctr = torch.tensor(
                [article.ctr for article in most_recent_clicks], dtype=torch.float32
            )

        # If a user has clicked on less articles, we pad with 0
        # The article id 0 does not exist, and for this special id the lookup news encoder
        # will also return zeros for this article
        elif len(clicks_sorted_on_recency) < max_clicked:
            ids = np.array(
                [article.article_id for article in clicks_sorted_on_recency]
                + [0] * (max_clicked - len(clicks_sorted_on_recency))
            )
            ctr = torch.tensor(
                [article.ctr for article in clicks_sorted_on_recency]
                + [0.0] * (max_clicked - len(clicks_sorted_on_recency)),
                dtype=torch.float32,
            )

        elif len(clicks_sorted_on_recency) == max_clicked:
            ids = np.array([article.article_id for article in clicks_sorted_on_recency])
            ctr = torch.tensor(
                [article.ctr for article in clicks_sorted_on_recency],
                dtype=torch.float32,
            )

        else:
            raise ValueError("This should not happen")

        clicked_ids.append(ids)
        clicked_ctr.append(ctr)

    clicked_ids = np.array(clicked_ids)
    clicked_ctr = torch.stack(clicked_ctr)

    assert clicked_ids.shape == (batch_size, max_clicked)
    assert clicked_ctr.shape == (batch_size, max_clicked)

    clicks_batch = PPRec.ClicksBatch(ids=clicked_ids, ctr=clicked_ctr)

    return clicks_batch
