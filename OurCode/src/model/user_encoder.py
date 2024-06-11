"""

This file contains implementations for the Popularity Aware User Encoder,
described in Qi et al. (2021) in section 3.4, and visualized in figure 5.

The only module thats shown in figure 5, and is not available in this code
is the news encoder, since its implemented in news_encoder.py. At least I guess
the News Encoder in figure 5 is the same as the one depicted in figure 3.

Q: Is the News Encoder module in figure 5 the same as the one shown in figure 3?

Q: Do we share the weights between the news encoder used in the user encoder,
and the one used in the popularity predictor?

"""

from dataclasses import dataclass
import math

from torch import nn
import torch

from .news_encoder import NEConfig


@dataclass
class PAUEConfig:
    # Its said in section 4.1 the popularity embedding size they used is 100
    p_size: int

    # And also in section 4.1: 20 heads with 20 dimensions output per head
    n_attention_heads: int
    head_output_size: int

    # This is configurable in their code
    # First line of `create_pe_model`
    # Q: How are the cased handled when a user has less clicked than 
    # the max_clicked? 
    max_clicked: int

    def m_size(self) -> int:
        return self.n_attention_heads * self.head_output_size


class PopularityAwareUserEncoder(nn.Module):
    pass


@dataclass
class PEConfig:
    p_size: int

    # In their code this 200 I think
    # check `popularity_embedding_layer` in `Encoders.create_pe_model`
    max_ctr: int

    # This is configurable in their code
    # First line of `create_pe_model`
    max_clicked: int


class PopularityEmbedding(nn.Module):
    """

    Implementation of the popularity embedding module for the popularity-aware user encoder.

    "Second, we uniformly quantify the popularity of the i-th clicked news predicted by
    the time-aware news popularity predictor and convert it into an embedding vector $p_i$
    via popularity embedding."

    They also leave a footnote, right after 'popularity predictor'.

    "We remove news recency and content here to avoid non-differentiable quantization operation."

    I don't really get this, how is the does anything here. To me it just seems like we have
    to use the click though rates with this Embedding layer. I looked at their code for a
    bit and noticed they do some kind of Attentive mechanism and I feel like this does something.

    Q: What has the popularity predictor to do with the popularity embedding?

    """

    def __init__(self, config: PEConfig):
        super().__init__()

        self.embedding = nn.Embedding(config.max_ctr, config.p_size)
        self.config = config

    def forward(self, ctr: torch.Tensor):
        """

        Calculates the popularity embeddings p based on the click through rate
        of every article.

        ctr is a tensor of shape (batch_size, max_clicked), where max_clicked is the max
        number of clicked articles by the user. The values in ctr are the click through
        rates of the articles, so they are integers. I thought at first the ctr
        would be some kind of division between clicks and impressions, but it seems
        like its just the number of clicks in some period? Got this from looking at
        their code. They must be integers to be used in this kind of embedding layer.

        p is a vector of shape (batch_size, max_clicked, popularity_embedding_size)

        """

        assert len(ctr.size()) == 2
        batch_size, max_clicked = ctr.size()
        assert max_clicked == self.config.max_clicked
        assert ctr.dtype == torch.int64 or ctr.dtype == torch.int32

        # The embedding layer has a specific max value, so we have to clip the ctr I think?
        # I feel like this is handled differently though.
        ctr_clipped = torch.clamp(ctr, 0, self.config.max_ctr - 1)

        p = self.embedding(ctr_clipped)  # (batch_size, max_clicked, p_size)
        assert p.size(0) == batch_size
        assert p.size(1) == max_clicked
        assert p.size(2) == self.config.p_size

        return p


@dataclass
class NSAConfig:
    n_size: int

    # And also in section 4.1: 20 heads with 20 dimensions output per head
    n_attention_heads: int
    head_output_size: int

    # This is configurable in their code
    # First line of `create_pe_model`
    max_clicked: int

    def get_size_m(self) -> int:
        return self.n_attention_heads * self.head_output_size


class NewsSelfAttention(nn.Module):
    """

    Implementation of the news self-attention module for
    the popularity-aware user encoder.

    """

    def __init__(self, config: NSAConfig):
        super().__init__()

        # The output size of the matrix should be the same as the
        # size of the news embeddings, since this is just the value
        # embeddings size in the self-attention mechanism.
        W_width = config.n_size

        # In their code, they use some kind of glorot initialization
        # lets fix that later, for now we just use random initialization
        self.Wq = nn.Parameter(torch.rand(config.n_size, W_width))
        self.Wk = nn.Parameter(torch.rand(config.n_size, W_width))
        self.Wv = nn.Parameter(torch.rand(config.n_size, W_width))

        self.config = config

    def forward(self, n: torch.Tensor) -> torch.Tensor:
        """

        Calculates the contextual news representations m, based on the
        news embeddings n outputted by the news encoder.

        n is a tensor of shape (batch_size, max_clicked, news_embedding_size)
        m is a tensor of shape (batch_size, max_clicked, head_output_size * n_attention_heads)
        where max_clicked is the number of clicked articles by the user.

        """

        batch_size, max_clicked, n_size = n.size()
        assert n_size == self.config.n_size
        assert max_clicked == self.config.max_clicked

        q = torch.matmul(n, self.Wq)  # (batch_size, max_clicked, n_size)
        assert len(q.size()) == 3
        assert q.size(0) == batch_size
        assert q.size(1) == max_clicked
        assert q.size(2) == self.config.n_size

        k = torch.matmul(n, self.Wk)  # (batch_size, max_clicked, n_size)
        assert len(k.size()) == 3
        assert k.size(0) == batch_size
        assert k.size(1) == max_clicked
        assert k.size(2) == self.config.n_size

        v = torch.matmul(n, self.Wv)  # (batch_size, max_clicked, n_size)
        assert len(v.size()) == 3
        assert v.size(0) == batch_size
        assert v.size(1) == max_clicked
        assert v.size(2) == self.config.n_size

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
            batch_size, max_clicked, self.config.get_size_m()
        )  # (batch_size, max_clicked, head_output_size * n_attention_heads)
        assert len(m.size()) == 3
        assert m.size(0) == batch_size
        assert m.size(1) == max_clicked
        assert m.size(2) == self.config.get_size_m()

        return m


@dataclass
class CPJAConfig:
    m_size: int
    p_size: int

    # The size of q, and the height of matrix W
    # Paper doesnt specify a default I think. Lets look
    # at their code.
    weight_size: int

    # This is configurable in their code
    # First line of `create_pe_model`
    max_clicked: int

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

        m is a tensor of shape (batch_size, max_clicked, m_size)
        p is a tensor of shape (batch_size, max_clicked, p_size)
        u is a tensor of shape (batch_size, m_size)
        where max_clicked is the number of clicked articles by the user.

        """

        assert len(m.size()) == 3
        batch_size, max_clicked, m_size = m.size()
        assert m_size == self.config.m_size
        assert max_clicked == self.config.max_clicked

        assert len(p.size()) == 3
        assert p.size(0) == batch_size
        assert p.size(1) == max_clicked
        assert p.size(2) == self.config.p_size

        mp = torch.cat((m, p), dim=2)  # (batch_size, max_clicked, m_size + p_size)
        assert len(mp.size()) == 3
        assert mp.size(0) == batch_size
        assert mp.size(1) == max_clicked
        assert mp.size(2) == self.config.m_size + self.config.p_size

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

        am = torch.mul(a.unsqueeze(2), m)  # (batch_size, max_clicked, m_size)
        assert len(am.size()) == 3
        assert am.size(0) == batch_size
        assert am.size(1) == max_clicked
        assert am.size(2) == self.config.m_size

        u = torch.sum(am, dim=1)  # (batch_size, m_size)
        assert len(u.size()) == 2
        assert u.size(0) == batch_size
        assert u.size(1) == self.config.m_size

        return u
