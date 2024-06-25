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

from .utils import DenseConfig, dense_from_hiddens_layers

from ..data.dataset import TrainDataPoint


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
        # Used by the lookup news encoder to load the embeddings
        data_folder: str | None,
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
                config=config.user_news_encoder_config,
                device=device,
                data_folder=data_folder,
            )
        else:
            raise ValueError("Unknown news encoder config")

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
                config=config.popularity_news_encoder_config,
                device=device,
                data_folder=data_folder,
            )
        else:
            raise ValueError("Unknown news encoder config")

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
            size_n=self.user_size_n,
        )

    @dataclass
    class CandidateBatch:
        # np array of size (batch_size, candidate_size)
        # article ids for every article
        # candidate_size is like npfactor in their code
        ids: np.ndarray

        # tensor of shape (batch_size, candidate_size)
        # click-through rate for every article (0-1)
        ctr: torch.Tensor

        # tensor of shape (batch_size, candidate_size)
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

    @dataclass
    class Inputs:
        candidates: "PPRec.CandidateBatch"
        clicks: "PPRec.ClicksBatch"

        def to_device(self, device: torch.device):
            self.candidates.recencies = self.candidates.recencies.to(device)
            self.candidates.ctr = self.candidates.ctr.to(device)
            self.clicks.ctr = self.clicks.ctr.to(device)

    @dataclass
    class BatchPredictions:
        # tensor of shape (batch_size, candidate_size)
        # the final ranking scores for every article
        # these scores aren't normalized or softmaxed.
        # softmax over the second dimension to get a
        # 'probability' distribution.
        score: torch.Tensor

        # tensor of shape (batch_size, candidate_size)
        # the personalized matching score for every article
        personalized_matching_score: torch.Tensor

        # tensor of shape (batch_size, candidate_size)
        # the popularity score for every article
        popularity_score: torch.Tensor

        # tensor of shape (batch_size, candidate_size, size_n)
        # the news embeddings for every article, outputted by the
        # user news encoder
        user_news_embeddings: torch.Tensor

        # tensor of shape (batch_size, candidate_size, size_n)
        # the news embeddings for every article, outputted by the
        # popularity news encoder
        popularity_news_embeddings: torch.Tensor

        def to_device(self, device: torch.device):
            self.score = self.score.to(device)
            self.personalized_matching_score = self.personalized_matching_score.to(
                device
            )
            self.popularity_score = self.popularity_score.to(device)
            self.user_news_embeddings = self.user_news_embeddings.to(device)
            self.popularity_news_embeddings = self.popularity_news_embeddings.to(device)

    def forward(
        self,
        inputs: "PPRec.Inputs",
    ):
        """

        Returns the ranking scores for a batch of candidate news articles, given the user's
        past click history.

        """

        candidates = inputs.candidates
        clicks = inputs.clicks

        assert len(clicks.ids.shape) == 2
        batch_size, max_clicked = clicks.ids.shape
        assert max_clicked == self.max_clicked

        assert len(clicks.ctr.shape) == 2
        assert clicks.ctr.shape == (batch_size, max_clicked)

        assert len(candidates.ids.shape) == 2
        assert candidates.ids.shape[0] == batch_size
        candidate_size = candidates.ids.shape[1]

        assert len(candidates.ctr.shape) == 2
        assert candidates.ctr.shape[0] == batch_size
        assert candidates.ctr.shape[1] == candidate_size

        assert len(candidates.recencies.shape) == 2
        assert candidates.recencies.shape[0] == batch_size
        assert candidates.recencies.shape[1] == candidate_size

        # First we get all news embeddings n

        candidates_ids_input = candidates.ids.reshape(
            -1
        )  # (batch_size * candidate_size)

        candidate_n_for_user_comparison = self.user_news_encoder(
            candidates_ids_input
        )  # (batch_size * candidate_size, size_n)
        assert len(candidate_n_for_user_comparison.shape) == 2
        assert candidate_n_for_user_comparison.shape[0] == batch_size * candidate_size
        assert candidate_n_for_user_comparison.shape[1] == self.user_size_n

        candidate_n_for_user_comparison = candidate_n_for_user_comparison.reshape(
            batch_size, candidate_size, -1
        )  # (batch_size, candidate_size, size_n)
        assert len(candidate_n_for_user_comparison.shape) == 3
        assert candidate_n_for_user_comparison.shape[0] == batch_size
        assert candidate_n_for_user_comparison.shape[1] == candidate_size
        assert candidate_n_for_user_comparison.shape[2] == self.user_size_n

        candidate_n_for_popularity = self.popularity_news_encoder(
            candidates_ids_input
        )  # (batch_size * candidate_size, size_n)
        assert len(candidate_n_for_popularity.shape) == 2
        assert candidate_n_for_popularity.shape[0] == batch_size * candidate_size
        assert candidate_n_for_popularity.shape[1] == self.popularity_size_n

        clicks_ids_input = clicks.ids.reshape(-1)  # (batch_size * max_clicked)
        assert len(clicks_ids_input.shape) == 1
        assert clicks_ids_input.shape[0] == batch_size * max_clicked

        clicks_n_for_user_comparison = self.user_news_encoder(
            clicks_ids_input
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
        assert user_embedding.shape[1] == self.user_encoder.size_n

        # size u should be the same as size n, is already enforced I think but to be sure
        assert user_embedding.shape[1] == candidate_n_for_user_comparison.shape[2]
        assert user_embedding.shape[1] == self.user_size_n

        # dot product over last dimension

        user_embedding_pms = user_embedding.unsqueeze(1)  # (batch_size, 1, size_u)
        assert len(user_embedding_pms.shape) == 3
        assert user_embedding_pms.shape[0] == batch_size
        assert user_embedding_pms.shape[1] == 1
        assert user_embedding_pms.shape[2] == self.user_size_n

        personalized_matching_score = torch.sum(
            user_embedding_pms * candidate_n_for_user_comparison, dim=-1
        )  # (batch_size, candidate_size)
        assert len(personalized_matching_score.shape) == 2
        assert personalized_matching_score.shape[0] == batch_size
        assert personalized_matching_score.shape[1] == candidate_size

        # Then we create the popularity score

        popularity_ctr_input = candidates.ctr.reshape(
            -1
        )  # (batch_size * candidate_size)
        assert len(popularity_ctr_input.shape) == 1
        assert popularity_ctr_input.shape[0] == batch_size * candidate_size

        popularity_recencies_input = candidates.recencies.reshape(
            -1
        )  # (batch_size * candidate_size)
        assert len(popularity_recencies_input.shape) == 1
        assert popularity_recencies_input.shape[0] == batch_size * candidate_size

        popularity_score = self.popularity_predictor(
            n=candidate_n_for_popularity,
            ctr=popularity_ctr_input,
            recencies=popularity_recencies_input,
        )  # (batch_size * candidate_size)
        assert len(popularity_score.shape) == 1
        assert popularity_score.shape[0] == batch_size * candidate_size

        popularity_score = popularity_score.reshape(
            batch_size, candidate_size
        )  # (batch_size, candidate_size)
        assert len(popularity_score.shape) == 2
        assert popularity_score.shape[0] == batch_size
        assert popularity_score.shape[1] == candidate_size

        # Then we create the personalized aggregator gate value eta

        eta = self.aggregator_gate(user_embedding)  # (batch_size)
        assert len(eta.size()) == 1
        assert eta.size(0) == batch_size

        # Then the final score

        eta = eta.unsqueeze(1)  # (batch_size, 1)
        assert len(eta.shape) == 2
        assert eta.shape[0] == batch_size
        assert eta.shape[1] == 1

        final_score = (
            eta * personalized_matching_score + (1 - eta) * popularity_score
        )  # (batch_size, candidate_size)
        assert len(final_score.shape) == 2
        assert final_score.shape[0] == batch_size
        assert final_score.shape[1] == candidate_size

        return self.BatchPredictions(
            score=final_score,
            personalized_matching_score=personalized_matching_score,
            popularity_score=popularity_score,
            user_news_embeddings=candidate_n_for_user_comparison,
            popularity_news_embeddings=candidate_n_for_popularity,
        )


@dataclass
class PAGConfig:
    dense_config: DenseConfig


class PersonalizedAggregatorGate(nn.Module):
    r"""

    Implements the calucaltion of $\eta$ from formula (3) in section 3.5
    of the paper. So it basically implements the personalized aggregator gate.

    This is just a simple dense network. Again, in the paper they specify something
    else than I can find in their code. Lets check what we are going to be using layer.
    I have done a single linear layer for the time being.

    """

    def __init__(self, size_n: int, config: PAGConfig):
        super().__init__()

        self.config = config
        self.size_n = size_n

        self.dense = dense_from_hiddens_layers(
            input_size=size_n,
            output_size=1,
            config=config.dense_config,
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        r"""

        Calculates the personalized aggregator gate value $\eta$ based on the
        user embeddings $u$.

        u is a tensor of shape (batch_size, size_u)

        eta is a tensor of shape (batch_size)

        """

        assert len(u.shape) == 2
        batch_size, size_u = u.shape
        assert size_u == self.size_n

        eta = torch.sigmoid(self.dense(u))  # shape (batch_size, 1)
        assert len(eta.shape) == 2
        assert eta.shape[1] == 1
        assert eta.shape[0] == batch_size

        eta = eta.squeeze(-1)  # shape (batch_size)
        assert len(eta.shape) == 1
        assert eta.shape[0] == batch_size

        return eta
