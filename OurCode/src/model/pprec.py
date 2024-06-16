"""

This file contains the implementation of the full PPRec model.
Using the popularity predictor, the user encoder, and the news 
encoder.

"""

from dataclasses import dataclass

from torch import nn
import torch

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

    def forward(
        self,
    ):
        """

        Returns the ranking scores for a batch of candidate news articles, given the user's
        past click history.

        """

        raise NotImplementedError()


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
