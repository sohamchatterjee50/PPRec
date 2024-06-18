"""

This file contains the implementation of the full PPRec model.
Using the popularity predictor, the user encoder, and the news 
encoder.

"""

from dataclasses import dataclass

from torch import nn
import torch
import numpy as np

from .news_encoder import (
    KnowledgeAwareNewsEncoder,
    ContentAwareNewsEncoder,
    LookupNewsEncoder,
    NEConfig,
    CANEConfig,
    KANEConfig,
    LNEConfig,
)
from .popularity_predictor import TimeAwareNewsPopularityPredictor, TANPPConfig
from .user_encoder import PopularityAwareUserEncoder, PAUEConfig


@dataclass
class PPRConfig:
    user_news_encoder_config: NEConfig
    popularity_news_encoder_config: NEConfig
    user_encoder_config: PAUEConfig
    popularity_predictor_config: TANPPConfig
    aggregator_gate_config: "PAGConfig"


class PPRec(nn.Module):
    """

    Implementation of PPRec. Figure 2 in the paper shows the architecture.
    Outputs a ranking score for some candidate news articles.

    """

    def __init__(
        self,
        # The maximum articles a user has clicked on in the past.
        # Depends on the dataloader used.
        max_clicked: int,
        # Needed by the lookup news encoder. So it can put its
        # looked up embeddings on the right device.
        device: torch.device,
        config: PPRConfig,
    ):

        super().__init__()

        self.config = config
        self.popularity_size_n = config.popularity_news_encoder_config.get_size_n()
        self.user_size_n = config.user_news_encoder_config.get_size_n()
        self.max_clicked = max_clicked

        if isinstance(config.user_news_encoder_config, CANEConfig):
            self.user_news_encoder = ContentAwareNewsEncoder(
                config.user_news_encoder_config
            )
        elif isinstance(config.user_news_encoder_config, KANEConfig):
            self.user_news_encoder = KnowledgeAwareNewsEncoder(
                config.user_news_encoder_config
            )
        elif isinstance(config.user_news_encoder_config, LNEConfig):
            self.user_news_encoder = LookupNewsEncoder(
                config=config.user_news_encoder_config, device=device
            )

        if isinstance(config.popularity_news_encoder_config, CANEConfig):
            self.popularity_news_encoder = ContentAwareNewsEncoder(
                config.popularity_news_encoder_config
            )
        elif isinstance(config.popularity_news_encoder_config, KANEConfig):
            self.popularity_news_encoder = KnowledgeAwareNewsEncoder(
                config.popularity_news_encoder_config
            )
        elif isinstance(config.popularity_news_encoder_config, LNEConfig):
            self.popularity_news_encoder = LookupNewsEncoder(
                config=config.popularity_news_encoder_config, device=device
            )

        self.popularity_predictor = TimeAwareNewsPopularityPredictor(
            config=config.popularity_predictor_config, size_n=self.popularity_size_n
        )
        self.user_encoder = PopularityAwareUserEncoder(
            config=config.user_encoder_config,
            size_n=self.user_size_n,
            max_clicked=max_clicked,
        )
        self.aggregator_gate = PersonalizedAggregatorGate(
            config=config.aggregator_gate_config,
            size_u=config.user_encoder_config.get_size_u(),
        )

    # Note: in these inputs there is no dimension for the 'npfactor'
    # but we can just reshape the batch_size to facilitate to these
    # dimensions. Makes no difference, and this makes a lot more sense.

    @dataclass
    class CandidateBatch:
        # np array of size (batch_size)
        # article ids for every article
        ids: np.ndarray

        # tensor of shape (batch_size)
        # click-through rate for every article (0-1)
        ctr: torch.Tensor

        # tensor of shape (batch_size)
        # recency for every article
        # the number of hours since the article was published
        recencies: torch.Tensor

    @dataclass
    class ClicksBatch:

        # np array of shape (batch_size, max_clicked)
        # article id for every article
        ids: np.ndarray

        # tensor of shape (batch_size, max_clicked)
        # click-through rate for every article (0-1)
        ctr: torch.Tensor

    def forward(
        self,
        clicks: ClicksBatch,
        candidates: CandidateBatch,
    ):
        """

        Returns the ranking scores for a batch of candidate news articles, given the user's
        past click history.

        """

        assert len(clicks.ids.shape) == 2
        batch_size, max_clicked = clicks.ids.shape
        assert max_clicked == self.max_clicked

        assert len(clicks.ctr.shape) == 2
        assert clicks.ctr.shape == (batch_size, max_clicked)

        assert len(candidates.ids.shape) == 1
        assert candidates.ids.shape[0] == batch_size

        assert len(candidates.ctr.shape) == 1
        assert candidates.ctr.shape[0] == batch_size

        assert len(candidates.recencies.shape) == 1
        assert candidates.recencies.shape[0] == batch_size

        # First we get all news embeddings n

        candidate_n_for_user_comparison = self.user_news_encoder(
            candidates.ids
        )  # (batch_size, size_n)
        assert len(candidate_n_for_user_comparison.shape) == 2
        assert candidate_n_for_user_comparison.shape[0] == batch_size
        assert candidate_n_for_user_comparison.shape[1] == self.user_size_n

        candidate_n_for_popularity = self.popularity_news_encoder(
            candidates.ids
        )  # (batch_size, size_n)
        assert len(candidate_n_for_popularity.shape) == 2
        assert candidate_n_for_popularity.shape[0] == batch_size
        assert candidate_n_for_popularity.shape[1] == self.popularity_size_n

        clicks_input_news = clicks.ids.reshape(-1)  # (batch_size * max_clicked)
        assert len(clicks_input_news.shape) == 1
        assert clicks_input_news.shape[0] == batch_size * max_clicked

        clicks_n_for_user_comparison = self.user_news_encoder(
            clicks_input_news
        )  # (batch_size * max_clicked, size_n)
        assert len(clicks_n_for_user_comparison.shape) == 2
        assert clicks_n_for_user_comparison.shape[0] == batch_size * max_clicked
        assert clicks_n_for_user_comparison.shape[1] == self.user_size_n

        clicks_n_for_user_comparison = clicks_n_for_user_comparison.reshape(
            batch_size, max_clicked, -1
        )  # (batch_size, max_clicked, size_n)
        assert len(clicks_n_for_user_comparison.shape) == 3
        assert clicks_n_for_user_comparison.shape[0] == batch_size
        assert clicks_n_for_user_comparison.shape[1] == max_clicked
        assert clicks_n_for_user_comparison.shape[2] == self.user_size_n

        # Then we create the personalized matching score sm

        user_embedding = self.user_encoder(
            n=clicks_n_for_user_comparison, ctr=clicks.ctr
        )  # (batch_size, size_u)
        assert len(user_embedding.shape) == 2
        assert user_embedding.shape[0] == batch_size
        assert user_embedding.shape[1] == self.user_encoder.size_u

        # size u should be the same as size n, is already enforced I think but to be sure
        assert user_embedding.shape[1] == candidate_n_for_user_comparison.shape[1]
        assert user_embedding.shape[1] == self.user_size_n

        # dot product over last dimension
        personalized_matching_score = torch.sum(
            user_embedding * candidate_n_for_user_comparison, dim=-1
        )  # (batch_size)
        assert len(personalized_matching_score.shape) == 1
        assert personalized_matching_score.shape[0] == batch_size

        # Then we create the popularity score

        popularity_score = self.popularity_predictor(
            n=candidate_n_for_popularity,
            ctr=candidates.ctr,
            recencies=candidates.recencies,
        )  # (batch_size)
        assert len(popularity_score.shape) == 1
        assert popularity_score.shape[0] == batch_size

        # Then we create the personalized aggregator gate value eta

        eta = self.aggregator_gate(user_embedding)  # (batch_size)
        assert len(eta.size()) == 1
        assert eta.size(0) == batch_size

        # Then the final score

        final_score = eta * personalized_matching_score + (1 - eta) * popularity_score
        assert len(final_score.size()) == 1
        assert final_score.size(0) == batch_size

        return final_score, personalized_matching_score, popularity_score


@dataclass
class PAGConfig:
    hidden_layers: list[int]


class PersonalizedAggregatorGate(nn.Module):
    r"""

    Implements the calucaltion of $\eta$ from formula (3) in section 3.5
    of the paper. So it basically implements the personalized aggregator gate.

    This is just a simple dense network. Again, in the paper they specify something
    else than I can find in their code. Lets check what we are going to be using layer.
    I have done a single linear layer for the time being.

    """

    def __init__(self, size_u: int, config: PAGConfig):
        super().__init__()

        self.config = config
        self.size_u = size_u

        self.linear = nn.Linear(size_u, 1)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        r"""

        Calculates the personalized aggregator gate value $\eta$ based on the
        user embeddings $u$.

        u is a tensor of shape (batch_size, size_u)

        eta is a tensor of shape (batch_size)

        """

        assert len(u.shape) == 2
        batch_size, size_u = u.shape
        assert size_u == self.size_u

        eta = torch.sigmoid(self.linear(u))  # shape (batch_size, 1)
        assert len(eta.shape) == 2
        assert eta.shape[1] == 1
        assert eta.shape[0] == batch_size

        eta = eta.squeeze(-1)  # shape (batch_size)
        assert len(eta.shape) == 1
        assert eta.shape[0] == batch_size

        return eta
