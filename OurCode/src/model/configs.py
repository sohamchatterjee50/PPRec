from typing import Literal

from src.model.news_encoder import LNEConfig
from src.model.user_encoder import CPJAConfig, PEConfig, NSAConfig, PAUEConfig
from src.model.popularity_predictor import (
    TANPPConfig,
    CRGConfig,
    CBPDConfig,
    RBPDConfig,
    REConfig,
)
from src.model.pprec import PPRConfig, PAGConfig
from src.model.utils import DenseConfig


# This is the actual configuration they use in their code
CONFIG_IN_IMPLEMENTATION = PPRConfig(
    user_news_encoder_config=LNEConfig(size_n=400, model="bert"),
    popularity_news_encoder_config=LNEConfig(size_n=400, model="bert"),
    user_encoder_config=PAUEConfig(
        popularity_embedding_config=PEConfig(
            size_p=100,
            max_ctr=200,
        ),
        news_self_attention_config=NSAConfig(
            n_attention_heads=20,
            head_output_size=20,
        ),
        content_popularity_joint_attention_config=CPJAConfig(weight_size=100),
    ),
    popularity_predictor_config=TANPPConfig(
        recency_based_popularity_dense_config=RBPDConfig(
            dense_config=DenseConfig(
                hidden_layers=[64, 64],
                activation="tanh",
                dropout=None,
                batch_norm=False,
            )
        ),
        content_based_popularity_dense_config=CBPDConfig(
            dense_config=DenseConfig(
                hidden_layers=[256, 265],
                activation="tanh",
                dropout=None,
                batch_norm=False,
            )
        ),
        recency_embedding_config=REConfig(
            r_size=100,
            max_recency=1500,
            recency_factor=0.5,
        ),
        content_recency_gate_config=CRGConfig(
            dense_config=DenseConfig(
                hidden_layers=[128, 64],
                activation="tanh",
                dropout=None,
                batch_norm=False,
            )
        ),
    ),
    aggregator_gate_config=PAGConfig(
        dense_config=DenseConfig(
            hidden_layers=[128, 64],
            activation="tanh",
            dropout=None,
            batch_norm=False,
        )
    ),
)

# In the paper they specified another configuration, for instance 2 layer dense
# networks with 100 dimensional hidden unit. And for the aggregator gate just
# a single layer (perceptron).
CONFIG_IN_PAPER = PPRConfig(
    user_news_encoder_config=LNEConfig(size_n=400, model="bert"),
    popularity_news_encoder_config=LNEConfig(size_n=400, model="bert"),
    user_encoder_config=PAUEConfig(
        popularity_embedding_config=PEConfig(
            size_p=100,
            max_ctr=200,
        ),
        news_self_attention_config=NSAConfig(
            n_attention_heads=20,
            head_output_size=20,
        ),
        content_popularity_joint_attention_config=CPJAConfig(weight_size=100),
    ),
    popularity_predictor_config=TANPPConfig(
        recency_based_popularity_dense_config=RBPDConfig(
            dense_config=DenseConfig(
                hidden_layers=[100],
                activation="tanh",
                dropout=None,
                batch_norm=False,
            )
        ),
        content_based_popularity_dense_config=CBPDConfig(
            dense_config=DenseConfig(
                hidden_layers=[100],
                activation="tanh",
                dropout=None,
                batch_norm=False,
            )
        ),
        recency_embedding_config=REConfig(
            r_size=100,
            max_recency=1500,
            recency_factor=0.5,
        ),
        content_recency_gate_config=CRGConfig(
            dense_config=DenseConfig(
                hidden_layers=[100],
                activation="tanh",
                dropout=None,
                batch_norm=False,
            )
        ),
    ),
    aggregator_gate_config=PAGConfig(
        dense_config=DenseConfig(
            hidden_layers=[],
            activation="tanh",
            dropout=None,
            batch_norm=False,
        )
    ),
)

# A minimal config, using as little parameters as possible but still
# having a reasonable configuration.
MINIMAL_CONFIG = PPRConfig(
    user_news_encoder_config=LNEConfig(size_n=100, model="bert"),
    popularity_news_encoder_config=LNEConfig(size_n=100, model="bert"),
    user_encoder_config=PAUEConfig(
        popularity_embedding_config=PEConfig(
            size_p=30,
            max_ctr=20,
        ),
        news_self_attention_config=NSAConfig(
            n_attention_heads=2,
            head_output_size=10,
        ),
        content_popularity_joint_attention_config=CPJAConfig(weight_size=50),
    ),
    popularity_predictor_config=TANPPConfig(
        recency_based_popularity_dense_config=RBPDConfig(
            dense_config=DenseConfig(
                hidden_layers=[],
                activation="tanh",
                dropout=None,
                batch_norm=False,
            )
        ),
        content_based_popularity_dense_config=CBPDConfig(
            dense_config=DenseConfig(
                hidden_layers=[],
                activation="tanh",
                dropout=None,
                batch_norm=False,
            )
        ),
        recency_embedding_config=REConfig(
            r_size=30,
            max_recency=300,
            recency_factor=0.1,
        ),
        content_recency_gate_config=CRGConfig(
            dense_config=DenseConfig(
                hidden_layers=[],
                activation="tanh",
                dropout=None,
                batch_norm=False,
            )
        ),
    ),
    aggregator_gate_config=PAGConfig(
        dense_config=DenseConfig(
            hidden_layers=[],
            activation="tanh",
            dropout=None,
            batch_norm=False,
        )
    ),
)


ModelConfigType = Literal["implementation", "paper", "minimal"]


def config_from_type(config_type: ModelConfigType) -> PPRConfig:
    if config_type == "implementation":
        return CONFIG_IN_IMPLEMENTATION
    elif config_type == "paper":
        return CONFIG_IN_PAPER
    elif config_type == "minimal":
        return MINIMAL_CONFIG
    else:
        raise ValueError(f"Config type {config_type} not supported")
