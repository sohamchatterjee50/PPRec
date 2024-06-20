"""

This file contains immplementations for the Knowledge Aware News Encoder,
described in Qi et al. (2021) in section 3.2, and visualized in figure 3.

At the moment there only is a shortcut implementation, which does not
follow the visualization and explanation from the paper, but Songha told us
to use the pre-made article embeddings in the data artifact as a news encoder
as a starting point. To get something up and running first. 

"""

from typing import Literal
from abc import ABC
import os
from dataclasses import dataclass

from torch import nn
import torch
import pandas as pd
import numpy as np

from ..utils import get_data_folder


class NEConfig(ABC):
    """

    All news encoder configurations should inherit from this class.
    So in the full PPRec module, we can use the same interface for
    all news encoders, and test them by just changing the configuration.

    For instance, the popularity prdictor needs to know what the size
    of the news embeddings it can expect, so it can create the correct layers.
    Same for the user encoder.

    """

    def get_size_n(self) -> int:
        """

        Returns the size of the news embeddings.

        """
        raise NotImplementedError()


@dataclass
class KANEConfig(NEConfig):
    # Default values from the paper section 4.1 are 20, 20
    n_attention_heads: int
    head_output_size: int

    def get_size_n(self) -> int:
        """

        Returns the size of the news embeddings.

        This result is a summation of the word and entity representations,
        which are the output of an attention layer with N heads, and M
        dimensions output per head (check 4.1). This is concatenated, so
        results in a N * M dimensional vector.

        The default values are N = 20 and M = 20, so 400.

        """

        return self.n_attention_heads * self.head_output_size


class KnowledgeAwareNewsEncoder(nn.Module):
    """

    The actual news encoder as implemented in the paper.

    """

    def __init__(self, config: NEConfig, device: torch.device = torch.device("cpu")):
        super().__init__()

        self.config = config

        raise NotImplementedError()

    def forward(self, texts: list[str], entities: list[str]) -> torch.Tensor:
        """

        Returns the news encodings for a batch of articles.

        texts and entities are a list both of length `batch_size`

        The news encodings are a tensor of size (batch_size, size_n),
        where size_n is the size of the news embeddings.

        """

        assert len(texts) == len(entities)

        raise NotImplementedError()


@dataclass
class CANEConfig(NEConfig):
    # Default values from the paper section 4.1 are 20, 20
    n_attention_heads: int
    head_output_size: int

    def get_size_n(self) -> int:
        """

        Returns the size of the news embeddings.

        This result is a summation of the word and entity representations,
        which are the output of an attention layer with N heads, and M
        dimensions output per head (check 4.1). This is concatenated, so
        results in a N * M dimensional vector.

        The default values are N = 20 and M = 20, so 400.

        """

        return self.n_attention_heads * self.head_output_size


class ContentAwareNewsEncoder(nn.Module):
    """

    As a second step, Songga told us to implement a news encoder that
    only incorporates the article text, and not the entities.

    """

    def __init__(self, config: CANEConfig):
        super().__init__()

        self.config = config

        raise NotImplementedError()

    def forward(self, texts: list[str]) -> torch.Tensor:
        """

        Returns the news encodings for a batch of articles.

        texts is a list of length `batch_size`

        The news encodings are a tensor of size (batch_size, size_n),
        where size_n is the size of the news embeddings.

        """

        raise NotImplementedError()


TextEncodeModel = Literal["bert", "roberta", "word2vec", "contrastive"]


@dataclass
class LNEConfig(NEConfig):
    size_n: int
    model: TextEncodeModel

    def get_size_n(self) -> int:
        return self.size_n


class LookupNewsEncoder(nn.Module):
    """

    Songga told us to first use the premade article embedding in one
    of the artifacts as a newencoder.

    So this news encoder lookes up the embeddings of the articles in
    the data artifact and convert them to the desired size, using a
    fully connected layer.

    """

    def __init__(
        self,
        config: LNEConfig,
        device: torch.device,
        data_folder: str | None = None,
    ):
        super().__init__()

        data_folder = get_data_folder(data_folder)

        parquet_file_path = _parquet_file_path_from_model(config.model, data_folder)

        self.data = pd.read_parquet(parquet_file_path)
        self.data.set_index("article_id", inplace=True)

        embeddings_column_name = _embeddings_column_name_from_model(config.model)
        self.data.rename(columns={embeddings_column_name: "embeddings"}, inplace=True)

        news_embedding_size = config.get_size_n()

        self.embedding_size = _embedding_size_from_model(config.model)

        self.fcout = nn.Linear(self.embedding_size, news_embedding_size)

        self.config = config
        self.device = device

    def forward(self, article_ids: np.ndarray) -> torch.Tensor:
        """

        Returns n (the news encodings) for a batch of articles.
        Fills embeddings that werent found with zeros

        article_ids is a numpy array of size (batch_size)

        n has shape (batch_size, embedding_size),

        """

        embeddings = self.get_embeddings_batch(article_ids)

        embeddings = torch.tensor(embeddings, dtype=torch.float32, device=self.device)

        n = self.fcout(embeddings)  # (batch_size, embedding_size)
        assert len(n.size()) == 2
        assert n.size(0) == len(article_ids)
        assert n.size(1) == self.config.get_size_n()

        return n

    def get_embeddings(self, article_id: int) -> np.ndarray:
        """

        Gets the embeddings given an article id.

        The embeddings are a numpy array of size (768,)
        except for the word2vec model where the size is (300,)

        """

        if article_id == 0:
            return np.zeros(self.embedding_size)

        embeddings = self.data.loc[article_id, "embeddings"]
        embeddings = np.array(embeddings)

        return embeddings

    def get_embeddings_batch(self, article_ids: np.ndarray) -> np.ndarray:
        """

        Gets the embeddings for a batch of article ids.

        The article ids are a numpy array of size (batch_size)

        The embeddings are a numpy array of size (batch_size, 768)
        except for the word2vec model where the size is (batch_size, 300)

        """

        embeddings = np.array(
            [self.get_embeddings(article_id) for article_id in article_ids]
        )

        return embeddings


def _embeddings_column_name_from_model(model: TextEncodeModel) -> str:
    """

    Returns the name of the column containing the embeddings of
    the articles in the dataframe for the given model.

    """
    match model:
        case "bert":
            return "google-bert/bert-base-multilingual-cased"
        case "roberta":
            return "FacebookAI/xlm-roberta-base"
        case "word2vec":
            return "document_vector"
        case "contrastive":
            return "contrastive_vector"

    raise ValueError(f"Unknown model: {model}")


def _parquet_file_path_from_model(model: TextEncodeModel, data_folder: str) -> str:
    """

    Returns the file path to the parquet file containing the
    embeddings of the articles for the given model.

    E.g. the bert model article embeddings are located in
    $DATA_FOLDER/google_bert_base_multilingual_cased/bert_base_multilingual_cased.parquet

    """

    match model:
        case "bert":
            return os.path.join(
                data_folder,
                "google_bert_base_multilingual_cased",
                "bert_base_multilingual_cased.parquet",
            )
        case "roberta":
            return os.path.join(
                data_folder, "FacebookAI_xlm_roberta_base", "xlm_roberta_base.parquet"
            )
        case "word2vec":
            return os.path.join(
                data_folder, "Ekstra_Bladet_word2vec", "document_vector.parquet"
            )
        case "contrastive":
            return os.path.join(
                data_folder,
                "Ekstra_Bladet_contrastive_vector",
                "contrastive_vector.parquet",
            )

    raise ValueError(f"Unknown model: {model}")


def _embedding_size_from_model(model: TextEncodeModel) -> int:
    """

    Not all the article embeddings in the data artifact have the same size.
    They have size 768, except for the word2vec model where the size is 300.

    This function returns the size of the embeddings for the given model.

    """

    match model:
        case "bert" | "roberta" | "contrastive":
            return 768
        case "word2vec":
            return 300

    raise ValueError(f"Unknown model: {model}")
