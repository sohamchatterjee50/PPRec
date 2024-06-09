from dataclasses import dataclass

from torch import nn
import torch

from .news_encoder import NewsEncoderConfig


@dataclass
class PAUEConfig:
    popularity_embedding_size: int = 100
    n_attention_heads: int = 20
    head_output_size: int = 20

    news_encoder_config: NewsEncoderConfig = NewsEncoderConfig()

    def m_vector_size(self) -> int:
        return self.n_attention_heads * self.head_output_size


class PopularityAwareUserEncoder(nn.Module):
    pass


@dataclass
class PEConfig:
    p_size: int = 100


class PopularityEmbedding:
    """

    Implementation of the popularity embedding module for the popularity-aware user encoder.

    Here, as stated in the paper, news recency and content are removed, to avoid non-
    differentiable quantization operations. So only the click-through rate is used
    in the popularity predictor to calculate the popularity score $s_p$.

    This module does not implement the popularity predictor, this is implemented in
    popularity_predictor.py. But this module does implement the conversion from the
    popularity scores to the popularity embeddings, via a simple linear transformation.

    We discussed with Songga that this is probably the way they implemented it, since
    the paper doesn't specifically discuss how this conversion is done. All it states is:

    "Second, we uniformly quantify the popularity of the i-th clicked news predicted by
    the time-aware news popularity predictor and convert it into an embedding vector $p_i$
    via popularity embedding."

    They also leave a footnote, right after 'popularity predictor'.

    "We remove news recency and content here to avoid non-differentiable quantization operation."

    """

    def __init__(self, config: PEConfig):
        super().__init__()

        self.w = nn.Parameter(torch.rand(config.p_size))

    def forward(self, sp: torch.Tensor):
        """

        Calculates the popularity embeddings p based on the popularity scores sp.
        These popularity scores are the output of the popularity predictor, but this
        time should only be calculated based on the click-through rate, to avoid
        non-differentiable quantization operations.

        sp is a tensor of shape (batch_size, N), where N is the number
        of clicked articles by the user.

        p is a vector of shape (batch_size, N, popularity_embedding_size)

        """

        p = torch.mul(sp.unsqueeze(2), self.w)  # (batch_size, N, p_size)

        return p


@dataclass
class NSAConfig:
    news_embedding_size: int

    n_attention_heads: int = 20
    head_output_size: int = 20


class NewsSelfAttention(nn.Module):
    """

    Implementation of the news self-attention module for
    the popularity-aware user encoder.

    """

    def __init__(self, config: NSAConfig):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=config.head_output_size * config.n_attention_heads,
            num_heads=config.n_attention_heads,
            batch_first=True,  # so the input is (batch_size, N, news_embedding_size)
        )

        self.config = config

    def forward(self, n: torch.Tensor) -> torch.Tensor:
        """

        Calculates the contextual news representations m, based on the
        news embeddings n outputted by the news encoder.

        n is a tensor of shape (batch_size, N, news_embedding_size)
        m is a tensor of shape (batch_size, N, head_output_size)
        where N is the number of clicked articles by the user.

        """

        raise NotImplementedError()


@dataclass
class CPJAConfig:
    m_size: int
    p_size: int

    # The size of q, and the height of matrix W
    weight_size: int = 100

    def get_width_Wu(self) -> int:
        return self.m_size + self.p_size


class ContentPopularityJointAttention(nn.Module):
    """

    Implementation of the content-popularity joint attention module
    for the popularity-aware user encoder.

    This is based on formula (2) in 3.4 of the paper.

    """

    def __init__(self, config: CPJAConfig):
        super().__init__()

        # Q: should the weights be initialized randomly?
        # Its not stated in the paper, as far as I can see.
        # I guess, there is a better way to initialize them.
        # Lets check in their code, or ask Songga, or do some
        # research on the topic.

        self.Wu = nn.Parameter(torch.rand(config.weight_size, config.get_width_Wu()))
        self.b = nn.Parameter(torch.rand(config.weight_size))

        self.config = config

    def forward(self, m: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """

        Calculates the user interest embeddings u, based on the
        the popularity embeddings p, and the contextual news
        representations m.

        m is a tensor of shape (batch_size, N, m_size)
        p is a tensor of shape (batch_size, N, p_size)
        u is a tensor of shape (batch_size, m_size)
        where N is the number of clicked articles by the user.

        """

        mp = torch.cat((m, p), dim=2)  # (batch_size, N, m_size + p_size)
        Wu_mp = torch.matmul(mp, self.Wu.T)  # (batch_size, N, weight_size)
        tanh_Wu_mp = torch.tanh(Wu_mp)  # (batch_size, N, weight_size)
        b_tanh_Wu_mp = torch.matmul(tanh_Wu_mp, self.b)  # (batch_size, N)
        sum_b_tanh_Wu_mp = torch.sum(b_tanh_Wu_mp, dim=1)  # (batch_size)
        a = torch.div(b_tanh_Wu_mp, sum_b_tanh_Wu_mp.unsqueeze(1))  # (batch_size, N)
        am = torch.mul(a.unsqueeze(2), m)  # (batch_size, N, m_size)
        u = torch.sum(am, dim=1)  # (batch_size, m_size)

        return u
