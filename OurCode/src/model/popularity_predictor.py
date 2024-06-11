"""

This module contains the implementations for the Time-Aware News Popularity Predictor,
described in Qi et al. (2021) in section 3.3, and visualized in figure 4.

The only module that is not available in this code, but is shown in the figure, is
the news encoder. This is implemented in news_encoder.py. My question here would be,
just like in for the Popularity Aware User Encoder file, do we share weights
between the News Encoder used in the user encoder? At least if the news encoder in 
figure 4 is the one shown in figure 3.

Q: Is the News Encoder module in figure 5 the same as the one shown in figure 3?

Q: Do we share the weights between the news encoder used in the user encoder,
and the one used in the popularity predictor?

"""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class TANPPConfig:
    n_size: int

    # Its said in section 4.1 the recency embedding size they used is 100
    r_size: int

    # In their code this is 1500 I think
    # check `time_embedding_layer` in `Encoders.create_pe_model`
    max_recency: int


class TimeAwareNewsPopularityPredictor(nn.Module):

    def __init__(self, config: TANPPConfig):
        super().__init__()

        self.config = config

        self.wc = nn.Parameter(torch.rand(1))
        self.wp = nn.Parameter(torch.rand(1))

        self.recency_embedding = RecencyEmbedding(
            REConfig(r_size=config.r_size, max_recency=config.max_recency)
        )
        self.recency_based_popularity_dense = RecencyBasedPopularityDense(
            RBPDConfig(r_size=config.r_size)
        )
        self.content_based_popularity_dense = ContentBasedPopularityDense(
            CBPDConfig(n_size=config.n_size)
        )
        self.content_recency_gate = ContentRecencyGate(
            CRGConfig(r_size=config.r_size, n_size=config.n_size)
        )

    def forward(
        self, n: torch.Tensor, recencies: torch.Tensor, crt: torch.Tensor
    ) -> torch.Tensor:
        r"""

        Calculate the popularity score $s_p$ based on the news embeddings n and the
        article recencies (the hours since the news article was published, to the time
        of prediction).

        n is a tensor of shape (batch_size, n_size) that contains the news embeddings
        for each news article in the batch.

        recencies is an integer tensor of shape (batch_size) that contains the recencies (the
        hours since the news article was published, to the time of prediction) for each
        news article in the batch.

        crt is a tensor of shape (batch_size) that contains the click through rates for
        each news article in the batch. Its a float this time, not an integer like in the
        user encoder. So a value between 0 and 1.

        sp is a tensor of shape (batch_size) that contains the popularity scores for
        each news article in the batch.

        """

        batch_size, n_size = n.size()
        assert recencies.size(0) == batch_size
        assert n_size == self.config.n_size
        assert crt.max() <= 1.0 and crt.min() >= 0.0
        assert crt.dtype == torch.float32 or crt.dtype == torch.float64

        r = self.recency_embedding(recencies)  # (batch_size, r_size)
        assert len(r.size()) == 2
        assert r.size(1) == self.config.r_size
        assert r.size(0) == batch_size

        pr = self.recency_based_popularity_dense(r)  # (batch_size)
        assert len(pr.size()) == 1
        assert pr.size(0) == batch_size

        pc = self.content_based_popularity_dense(n)  # (batch_size)
        assert len(pc.size()) == 1
        assert pc.size(0) == batch_size

        theta = self.content_recency_gate(r, n)  # (batch_size)
        assert len(theta.size()) == 1
        assert theta.size(0) == batch_size

        p = theta * pc + (1 - theta) * pr  # (batch_size)
        assert len(p.size()) == 1
        assert p.size(0) == batch_size

        sp = self.wc * crt + self.wp * p  # (batch_size)
        assert len(sp.size()) == 1
        assert sp.size(0) == batch_size

        return sp


@dataclass
class REConfig:
    r_size: int

    # In their code this is 1500 I think
    # check `time_embedding_layer` in `Encoders.create_pe_model`
    max_recency: int


class RecencyEmbedding(nn.Module):
    """

    The Recency Embedding module in figure 4, converting the recencies
    (the rounded hours since the news article was published, to the time
    of prediction) to a recency embedding r.

    """

    def __init__(self, config: REConfig):
        super().__init__()

        self.config = config
        self.embedding = nn.Embedding(config.max_recency, config.r_size)

    def forward(self, recency: torch.Tensor) -> torch.Tensor:
        r"""

        Calculate the recency embedding r based on the recency, which is the hours
        since the news article was published, to the time of prediction.

        recency is a tensor of shape (batch_size) that contains the recencies (the
        hours since the news article was published, to the time of prediction) for
        each news article in the batch. These are rounded to the nearest hour, so
        an integer.

        """

        assert recency.dim() == 1
        assert recency.dtype == torch.int64 or recency.dtype == torch.int32

        # The embedding has a maximum value for which it can be used, so we clip
        # the recency to that value.
        recency_clipped = torch.clamp(recency, max=self.config.max_recency - 1)

        r = self.embedding(recency_clipped)  # (batch_size, r_size)
        assert len(r.size()) == 2
        assert r.size(1) == self.config.r_size
        assert r.size(0) == recency.size(0)

        return r


@dataclass
class RBPDConfig:
    r_size: int


class RecencyBasedPopularityDense(nn.Module):
    r"""

    The Dense module in figure 4, which is a simple feedforward neural network,
    but the one used to caluculate the recency based popularity $\hat{p}_r$
    based on the recency embedding r.

    This one just has one layer, but looking at their code they use a 2-layer
    network: 400 -> 256 -> 1. But not sure, the Keras code is unreadable.

    """

    def __init__(self, config: RBPDConfig):
        super().__init__()

        self.config = config
        self.dense = nn.Linear(config.r_size, 1)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        r"""

        Calculate the recency based popularity $\hat{p}_r$ based on the recency
        embedding r.

        r is a tensor of shape (batch_size, r_size) that contains the recency
        embeddings for each news article in the batch.

        """

        assert r.size(1) == self.config.r_size

        pr = self.dense(r)
        return pr


@dataclass
class CBPDConfig:
    n_size: int


class ContentBasedPopularityDense(nn.Module):
    r"""

    The Dense module in figure 4, which is a simple feedforward neural network,
    but the one used to caluculate the content based popularity $\hat{p}_c$
    based on the news embedding n.

    This one just has one layer, but looking at their code they use a 2-layer
    network: 400 -> 256 -> 1. But unsure as well.

    """

    def __init__(self, config: CBPDConfig):
        super().__init__()

        self.config = config
        self.dense = nn.Linear(config.n_size, 1)

    def forward(self, n: torch.Tensor) -> torch.Tensor:
        r"""

        Calculate the content based popularity $\hat{p}_c$ based on the news
        embedding n.

        n is a tensor of shape (batch_size, n_size) that contains the news
        embeddings for each news article in the batch.

        """

        assert n.size(1) == self.config.n_size

        pc = self.dense(n)
        return pc


@dataclass
class CRGConfig:
    r_size: int
    n_size: int


class ContentRecencyGate(nn.Module):
    r"""

    Implementets the 'content-specific gate', the Gate in figure 4. This gate
    makes the decision whether the content based popularity $\hat{p}_c$ or the
    recency based popularity $\hat{p}_r$ is more important, in to calculate the
    final popularity score $\hat{p}$.

    Q: In the paper Wp is mentioned as a vector, and bp as a vector. But that
    does not make sense to me, since theta is a scalar? So I assume that Wp is
    a vector, and bp is a scalar. Should als Songga. Or should we add the

    """

    def __init__(self, config: CRGConfig):
        super().__init__()

        self.config = config

        # assumption: Wp is a vector, and bp is a scalar, check class docstring
        self.Wp = nn.Parameter(torch.rand(config.r_size + config.n_size))
        self.bp = nn.Parameter(torch.rand(1))

    def forward(self, r: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        r"""

        Calculate the content-specific gate value theta, which is a scalar that
        makes the decision whether the content based popularity $\hat{p}_c$ or
        the recency based popularity $\hat{p}_r$ is more important.

        r is a tensor of shape (batch_size, r_size) that contains the recency
        embeddings for each news article in the batch.

        n is a tensor of shape (batch_size, n_size) that contains the news
        embeddings for each news article in the batch.

        theta is a tensor of shape (batch_size) that contains the content-specific
        gate values for each news article in the batch.

        """

        assert r.size(1) == self.config.r_size
        assert n.size(1) == self.config.n_size

        rn = torch.cat([r, n], dim=1)  # (batch_size, r_size + n_size)
        assert rn.size(1) == self.config.r_size + self.config.n_size

        Wp_rn = torch.matmul(rn, self.Wp)  # (batch_size)
        assert len(Wp_rn.size()) == 1

        Wp_rn_bp = Wp_rn + self.bp  # (batch_size)
        assert len(Wp_rn_bp.size()) == 1

        theta = torch.sigmoid(Wp_rn_bp)

        return theta
