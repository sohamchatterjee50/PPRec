"""

This file contains implementations for the Popularity Aware User Encoder,
described in Qi et al. (2021) in section 3.4, and visualized in figure 5.

The only module thats shown in figure 5, and is not available in this code
is the news encoder, since its implemented in news_encoder.py. 

"""

from dataclasses import dataclass
import math

from torch import nn
import torch

from .news_encoder import NEConfig


@dataclass
class PAUEConfig:
    popularity_embedding_config: "PEConfig"
    news_self_attention_config: "NSAConfig"
    content_popularity_joint_attention_config: "CPJAConfig"


class PopularityAwareUserEncoder(nn.Module):
    """

    Implementation of the popularity-aware user encoder module.
    This is figure 5 in the paper. Or the Popularity-aware User Encoder
    module in figure 2, when looking at the overall PPRec achitecture.
    Uses the popularity embedding, news self-attention, and content-popularity
    joint attention modules.

    """

    def __init__(
        self,
        size_n: int,
        # How many articles the user clicked
        # on as a maximum. This is configurable
        # in their code as well.
        # First line of `create_pe_model`
        # But its based on dataloader settings,
        # so lets put it here.
        max_clicked: int,
        config: PAUEConfig,
    ):
        super().__init__()

        self.config = config
        self.size_p = config.popularity_embedding_config.size_p
        self.size_n = size_n
        self.max_clicked = max_clicked

        self.popularity_embedding = PopularityEmbedding(
            max_clicked=max_clicked, config=config.popularity_embedding_config
        )
        self.news_self_attention = NewsSelfAttention(
            n_size=size_n,
            max_clicked=max_clicked,
            config=config.news_self_attention_config,
        )
        self.content_popularity_joint_attention = ContentPopularityJointAttention(
            max_clicked=max_clicked,
            n_size=self.size_n,
            p_size=self.size_p,
            config=config.content_popularity_joint_attention_config,
        )

    def forward(self, n: torch.Tensor, ctr: torch.Tensor) -> torch.Tensor:
        """

        Calculates the user interest embeddings u, based on the news embeddings n,
        and the click through rates ctr of the articles.

        n is a tensor of shape (batch_size, max_clicked, size_n) representing
        the news embeddings for all the articles the user clicked on.

        ctr is a tensor of shape (batch_size, max_clicked) and represents the click
        through rates of the articles the user clicked on. This is a value between 0 and 1,
        representing the number of clicks divided by the number of impressions. So like
        a popularity score.

        u is a tensor of shape (batch_size, size_u), representing the user interest embeddings.

        """

        assert len(n.size()) == 3
        batch_size, max_clicked, size_n = n.size()
        assert size_n == self.size_n
        assert max_clicked == self.max_clicked

        assert len(ctr.size()) == 2
        assert ctr.size(0) == batch_size
        assert ctr.size(1) == max_clicked

        p = self.popularity_embedding(ctr)  # (batch_size, max_clicked, size_p)
        assert len(p.size()) == 3
        assert p.size(0) == batch_size
        assert p.size(1) == max_clicked
        assert p.size(2) == self.size_p

        m = self.news_self_attention(n)  # (batch_size, max_clicked, size_n)
        assert len(m.size()) == 3
        assert m.size(0) == batch_size
        assert m.size(1) == max_clicked
        assert m.size(2) == self.size_n

        u = self.content_popularity_joint_attention(m, p)  # (batch_size, size_n)
        assert len(u.size()) == 2
        assert u.size(0) == batch_size
        assert u.size(1) == self.size_n

        return u


@dataclass
class PEConfig:

    # They use 100 for the popularity embedding size
    size_p: int

    # In their code this 200 I think
    # check `popularity_embedding_layer` in `Encoders.create_pe_model`
    # Context: for the user encoder, their ctr needs to be an integer,
    # so it can be used in the Embedding layer (they call it popularity
    # embdding in the paper). They scale the ctr from 0 to 200. So to be
    # clear, this ctr is not the absolute number of clicks. Its still the
    # click through rate, the number of clicks divided by the number of
    # impressions. They just scale and rount it to be an integer between
    # 0 and 200.
    max_ctr: int


class PopularityEmbedding(nn.Module):
    """

    Implementation of the popularity embedding module for the popularity-aware user encoder.

    "Second, we uniformly quantify the popularity of the i-th clicked news predicted by
    the time-aware news popularity predictor and convert it into an embedding vector $p_i$
    via popularity embedding."

    They also leave a footnote, right after 'popularity predictor'.

    "We remove news recency and content here to avoid non-differentiable quantization operation."

    Looking at their code, this popularity embedding, or the user encoder in general, has
    nothing to do with the popularity predictor. It just also has click-through rates as input,
    but this time not for the canditates, but for the clicked articles.

    """

    def __init__(self, max_clicked: int, config: PEConfig):
        super().__init__()

        self.embedding = nn.Embedding(config.max_ctr, config.size_p)
        self.config = config
        self.max_clicked = max_clicked

    def forward(self, ctr: torch.Tensor) -> torch.Tensor:
        """

        Calculates the popularity embeddings p based on the click through rate
        of every article.

        ctr is a tensor of shape (batch_size, max_clicked), where max_clicked is the max
        number of clicked articles by the user. The values in ctr are the click through
        rates of the articles. A value between 0 and 1. The number of impressions divided
        by the number of clicks. For this module, they need to be integers between 0 and max_ctr.
        The authors already do this in the dataloader, but to me it seems its better practice
        to do it here, so it doesnt get confusing why the ctr is an int between 0 and max_ctr
        all of a sudden.

        p is a vector of shape (batch_size, max_clicked, popularity_embedding_size)

        """

        assert len(ctr.size()) == 2
        batch_size, max_clicked = ctr.size()
        assert max_clicked == self.max_clicked
        assert ctr.dtype == torch.float32 or ctr.dtype == torch.float64
        assert ctr.max() <= 1.0 and ctr.min() >= 0.0

        # Here we scale the ctr to be between 0 and max_ctr
        ctr_clipped = torch.mul(ctr, self.config.max_ctr).long()

        p = self.embedding(ctr_clipped)  # (batch_size, max_clicked, p_size)
        assert p.size(0) == batch_size
        assert p.size(1) == max_clicked
        assert p.size(2) == self.config.size_p

        return p


@dataclass
class NSAConfig:

    # And also in section 4.1: 20 heads with 20 dimensions output per head
    n_attention_heads: int
    head_output_size: int


class NewsSelfAttention(nn.Module):
    """

    Implementation of the news self-attention module for
    the popularity-aware user encoder.

    """

    def __init__(self, n_size: int, max_clicked: int, config: NSAConfig):
        super().__init__()

        size_u = config.n_attention_heads * config.head_output_size

        # In their code, they use some kind of glorot initialization
        # lets fix that later, for now we just use random initialization
        self.Wq = nn.Parameter(torch.rand(n_size, size_u))
        self.Wk = nn.Parameter(torch.rand(n_size, size_u))
        self.Wv = nn.Parameter(torch.rand(n_size, size_u))

        # Dense layer to bring the output of the self-attention
        # back to the original size of the news embeddings. Not used
        # in the paper, but this allows us to use different number of
        # heads and head output sizes, without having to change the
        # size of the news embeddings.
        self.output_dense = nn.Linear(size_u, n_size)

        self.config = config
        self.n_size = n_size
        self.max_clicked = max_clicked
        self.size_u = size_u

    def forward(self, n: torch.Tensor) -> torch.Tensor:
        """

        Calculates the contextual news representations m, based on the
        news embeddings n outputted by the news encoder.

        n is a tensor of shape (batch_size, max_clicked, news_embedding_size)
        m is a tensor of shape (batch_size, max_clicked, head_output_size * n_attention_heads)
        where max_clicked is the number of clicked articles by the user.

        """

        batch_size, max_clicked, n_size = n.size()
        assert n_size == self.n_size
        assert max_clicked == self.max_clicked

        q = torch.matmul(n, self.Wq)  # (batch_size, max_clicked, size_u)
        assert len(q.size()) == 3
        assert q.size(0) == batch_size
        assert q.size(1) == max_clicked
        assert q.size(2) == self.size_u

        k = torch.matmul(n, self.Wk)  # (batch_size, max_clicked, size_u)
        assert len(k.size()) == 3
        assert k.size(0) == batch_size
        assert k.size(1) == max_clicked
        assert k.size(2) == self.size_u

        v = torch.matmul(n, self.Wv)  # (batch_size, max_clicked, size_u)
        assert len(v.size()) == 3
        assert v.size(0) == batch_size
        assert v.size(1) == max_clicked
        assert v.size(2) == self.size_u

        q = q.view(
            batch_size,
            max_clicked,
            self.config.n_attention_heads,
            self.config.head_output_size,
        )  # (batch_size, max_clicked, n_attention_heads, head_output_size)
        assert len(q.size()) == 4
        assert q.size(0) == batch_size
        assert q.size(1) == max_clicked
        assert q.size(2) == self.config.n_attention_heads
        assert q.size(3) == self.config.head_output_size

        q = q.permute(
            0, 2, 1, 3
        )  # (batch_size, n_attention_heads, max_clicked, head_output_size)
        assert len(q.size()) == 4
        assert q.size(0) == batch_size
        assert q.size(1) == self.config.n_attention_heads
        assert q.size(2) == max_clicked
        assert q.size(3) == self.config.head_output_size

        k = k.view(
            batch_size,
            max_clicked,
            self.config.n_attention_heads,
            self.config.head_output_size,
        )
        assert len(k.size()) == 4
        assert k.size(0) == batch_size
        assert k.size(1) == max_clicked
        assert k.size(2) == self.config.n_attention_heads
        assert k.size(3) == self.config.head_output_size

        k = k.permute(
            0, 2, 1, 3
        )  # (batch_size, n_attention_heads, max_clicked, head_output_size)
        assert len(k.size()) == 4
        assert k.size(0) == batch_size
        assert k.size(1) == self.config.n_attention_heads
        assert k.size(2) == max_clicked
        assert k.size(3) == self.config.head_output_size

        v = v.view(
            batch_size,
            max_clicked,
            self.config.n_attention_heads,
            self.config.head_output_size,
        )  # (batch_size, max_clicked, n_attention_heads, head_output_size)
        assert len(v.size()) == 4
        assert v.size(0) == batch_size
        assert v.size(1) == max_clicked
        assert v.size(2) == self.config.n_attention_heads
        assert v.size(3) == self.config.head_output_size

        v = v.permute(
            0, 2, 1, 3
        )  # (batch_size, n_attention_heads, max_clicked, head_output_size)
        assert len(v.size()) == 4
        assert v.size(0) == batch_size
        assert v.size(1) == self.config.n_attention_heads
        assert v.size(2) == max_clicked
        assert v.size(3) == self.config.head_output_size

        qk = torch.matmul(
            q, k.permute(0, 1, 3, 2)
        )  # (batch_size, n_attention_heads, max_clicked, max_clicked)
        assert len(qk.size()) == 4
        assert qk.size(0) == batch_size
        assert qk.size(1) == self.config.n_attention_heads
        assert qk.size(2) == max_clicked
        assert qk.size(3) == max_clicked

        qk_scaled = torch.div(
            qk, math.sqrt(self.config.head_output_size)
        )  # (batch_size, n_attention_heads, max_clicked, max_clicked)
        assert len(qk_scaled.size()) == 4
        assert qk_scaled.size(0) == batch_size
        assert qk_scaled.size(1) == self.config.n_attention_heads
        assert qk_scaled.size(2) == max_clicked
        assert qk_scaled.size(3) == max_clicked

        attention = torch.nn.functional.softmax(
            qk_scaled, dim=3
        )  # (batch_size, n_attention_heads, max_clicked, max_clicked)
        assert len(attention.size()) == 4
        assert attention.size(0) == batch_size
        assert attention.size(1) == self.config.n_attention_heads
        assert attention.size(2) == max_clicked
        assert attention.size(3) == max_clicked

        m = torch.matmul(
            attention, v
        )  # (batch_size, n_attention_heads, max_clicked, head_output_size)
        assert len(m.size()) == 4
        assert m.size(0) == batch_size
        assert m.size(1) == self.config.n_attention_heads
        assert m.size(2) == max_clicked
        assert m.size(3) == self.config.head_output_size

        m = m.permute(
            0, 2, 1, 3
        )  # (batch_size, max_clicked, n_attention_heads, head_output_size)
        assert len(m.size()) == 4
        assert m.size(0) == batch_size
        assert m.size(1) == max_clicked
        assert m.size(2) == self.config.n_attention_heads
        assert m.size(3) == self.config.head_output_size

        # I have to use reshape here, the data is not contiguous in memory...
        m = m.reshape(
            batch_size,
            max_clicked,
            self.config.head_output_size * self.config.n_attention_heads,
        )  # (batch_size, max_clicked, head_output_size * n_attention_heads)
        assert len(m.size()) == 3
        assert m.size(0) == batch_size
        assert m.size(1) == max_clicked
        assert m.size(2) == self.config.head_output_size * self.config.n_attention_heads

        m = self.output_dense(m)  # (batch_size, max_clicked, n_size)
        assert len(m.size()) == 3
        assert m.size(0) == batch_size
        assert m.size(1) == max_clicked
        assert m.size(2) == self.n_size

        return m


@dataclass
class CPJAConfig:
    # The size of q, and the height of matrix W
    # Paper doesnt specify a default I think. Lets look
    # at their code.
    weight_size: int


class ContentPopularityJointAttention(nn.Module):
    """

    Implementation of the content-popularity joint attention module
    for the popularity-aware user encoder.

    This is based on formula (2) in 3.4 of the paper.

    """

    def __init__(self, max_clicked: int, n_size: int, p_size: int, config: CPJAConfig):
        super().__init__()

        # Q: should the weights be initialized randomly?
        # Its not stated in the paper, as far as I can see.
        # I guess, there is a better way to initialize them.
        # Lets check in their code, or ask Songga, or do some
        # research on the topic.

        self.Wu = nn.Parameter(torch.rand(config.weight_size, n_size + p_size))
        self.b = nn.Parameter(torch.rand(config.weight_size))

        self.config = config
        self.n_size = n_size
        self.p_size = p_size
        self.max_clicked = max_clicked

    def forward(self, m: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """

        Calculates the user interest embeddings u, based on the
        the popularity embeddings p, and the contextual news
        representations m.

        m is a tensor of shape (batch_size, max_clicked, m_size)
        p is a tensor of shape (batch_size, max_clicked, p_size)
        u is a tensor of shape (batch_size, m_size)
        where max_clicked is the number of clicked articles by the user.

        """

        assert len(m.size()) == 3
        batch_size, max_clicked, m_size = m.size()
        assert m_size == self.n_size
        assert max_clicked == self.max_clicked

        assert len(p.size()) == 3
        assert p.size(0) == batch_size
        assert p.size(1) == max_clicked
        assert p.size(2) == self.p_size

        mp = torch.cat((m, p), dim=2)  # (batch_size, max_clicked, n_size + p_size)
        assert len(mp.size()) == 3
        assert mp.size(0) == batch_size
        assert mp.size(1) == max_clicked
        assert mp.size(2) == self.n_size + self.p_size

        Wu_mp = torch.matmul(mp, self.Wu.T)  # (batch_size, max_clicked, weight_size)
        assert len(Wu_mp.size()) == 3
        assert Wu_mp.size(0) == batch_size
        assert Wu_mp.size(1) == max_clicked
        assert Wu_mp.size(2) == self.config.weight_size

        tanh_Wu_mp = torch.tanh(Wu_mp)  # (batch_size, max_clicked, weight_size)
        assert len(tanh_Wu_mp.size()) == 3
        assert tanh_Wu_mp.size(0) == batch_size
        assert tanh_Wu_mp.size(1) == max_clicked
        assert tanh_Wu_mp.size(2) == self.config.weight_size

        b_tanh_Wu_mp = torch.matmul(tanh_Wu_mp, self.b)  # (batch_size, max_clicked)
        assert len(b_tanh_Wu_mp.size()) == 2
        assert b_tanh_Wu_mp.size(0) == batch_size
        assert b_tanh_Wu_mp.size(1) == max_clicked

        sum_b_tanh_Wu_mp = torch.sum(b_tanh_Wu_mp, dim=1)  # (batch_size)
        assert len(sum_b_tanh_Wu_mp.size()) == 1
        assert sum_b_tanh_Wu_mp.size(0) == batch_size

        a = torch.div(
            b_tanh_Wu_mp, sum_b_tanh_Wu_mp.unsqueeze(1)
        )  # (batch_size, max_clicked)
        assert len(a.size()) == 2
        assert a.size(0) == batch_size
        assert a.size(1) == max_clicked

        am = torch.mul(a.unsqueeze(2), m)  # (batch_size, max_clicked, n_size)
        assert len(am.size()) == 3
        assert am.size(0) == batch_size
        assert am.size(1) == max_clicked
        assert am.size(2) == self.n_size

        u = torch.sum(am, dim=1)  # (batch_size, n_size)
        assert len(u.size()) == 2
        assert u.size(0) == batch_size
        assert u.size(1) == self.n_size

        return u
