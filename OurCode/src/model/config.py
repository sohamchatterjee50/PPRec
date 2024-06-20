from src.model.news_encoder import LNEConfig
from src.model.user_encoder import CPJAConfig, PEConfig, NSAConfig, PAUEConfig
from src.model.popularity_predictor import (
    TANPPConfig,
    CRGConfig,
    CBPDConfig,
    RBPDConfig,
    REConfig,
)
from src.model.pprec import PPRConfig, PAGConfig, PPRec
from src.model.utils import DenseConfig


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
