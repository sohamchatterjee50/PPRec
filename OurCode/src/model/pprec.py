from dataclasses import dataclass

from torch import nn
import torch

from .news_encoder import ContentAwareNewsEncoder, NEConfig
from .popularity_predictor import TimeAwareNewsPopularityPredictor, TANPPConfig
from .user_encoder import PopularityAwareUserEncoder, PAUEConfig


class PPRConfig:
    user_news_encoder_config: NEConfig
    popularity_news_encoder_config: NEConfig
    user_encoder_config: PAUEConfig
    popularity_predictor_config: TANPPConfig
    aggregator_gate_config: "PAGConfig"


class PPRec(nn.Module):
    def __init__(self, config: PPRConfig):

        super().__init__()

        # output of the user encoder (user embeddings u) should equal the size the
        # aggregator gate expects.
        assert (
            config.user_encoder_config.get_size_u()
            == config.aggregator_gate_config.size_u
        )

        # output of the user news encoder (news_embeddings n) should equal the size
        # that the user encoder expects.
        assert (
            config.user_news_encoder_config.get_size_n()
            == config.user_encoder_config.size_n
        )

        # output of the popularity news encoder (news_embeddings n) should equal the size
        # that the popularity predictor expects.
        assert (
            config.popularity_news_encoder_config.get_size_n()
            == config.popularity_predictor_config.size_n
        )

        self.config = config

        user_news_encoder = ContentAwareNewsEncoder(config.user_news_encoder_config)
        popularity_news_encoder = ContentAwareNewsEncoder(
            config.popularity_news_encoder_config
        )
        popularity_predictor = TimeAwareNewsPopularityPredictor(
            config.popularity_predictor_config
        )
        user_encoder = PopularityAwareUserEncoder(config.user_encoder_config)
        aggregator_gate = PersonalizedAggregatorGate(config.aggregator_gate_config)


@dataclass
class PAGConfig:
    size_u: int


class PersonalizedAggregatorGate(nn.Module):
    r"""

    Implements the calucaltion of $\eta$ from formula (3) in section 3.5
    of the paper. So it basically implements the personalized aggregator gate.

    This is just a simple dense network. Again, in the paper they specify something
    else than I can find in their code. Lets check what we are going to be using layer.
    I have done a single linear layer for the time being.

    """

    def __init__(self, config: PAGConfig):
        super(PersonalizedAggregatorGate, self).__init__()

        self.config = config

        self.linear = nn.Linear(config.size_u, 1)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        r"""

        Calculates the personalized aggregator gate value $\eta$ based on the
        user embeddings $u$.

        u is a tensor of shape (batch_size, size_u)

        eta is a tensor of shape (batch_size)

        """

        assert len(u.shape) == 2
        batch_size, size_u = u.shape
        assert size_u == self.config.size_u

        eta = torch.sigmoid(self.linear(u))  # shape (batch_size, 1)
        assert len(eta.shape) == 2
        assert eta.shape[1] == 1
        assert eta.shape[0] == batch_size

        eta = eta.squeeze(-1)  # shape (batch_size)
        assert len(eta.shape) == 1
        assert eta.shape[0] == batch_size

        return eta
