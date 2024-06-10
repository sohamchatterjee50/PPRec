from typing import Literal
import os
from dataclasses import dataclass

from torch import nn
import torch
import pandas as pd
import numpy as np

from ..utils import get_data_folder


@dataclass
class NEConfig:
    n_attention_heads: int = 20
    head_output_size: int = 20

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


TextEncodeModel = Literal["bert", "roberta", "word2vec", "contrastive"]


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
        model: TextEncodeModel,
        config: NEConfig,
        data_folder: str | None = None,
    ):
        super().__init__()

        data_folder = get_data_folder(data_folder)

        parquet_file_path = _parquet_file_path_from_model(model, data_folder)

        self.data = pd.read_parquet(parquet_file_path)
        self.data.set_index("article_id", inplace=True)

        embeddings_column_name = _embeddings_column_name_from_model(model)
        self.data.rename(columns={embeddings_column_name: "embeddings"}, inplace=True)

        news_embedding_size = config.get_size_n()

        self.fcout = nn.Linear(_embedding_size_from_model(model), news_embedding_size)

        self.config = config
        self.model = model

    def forward(self, article_ids: list[int]) -> torch.Tensor:
        """

        Returns n (the news encodings) for a batch of articles.

        n has shape (batch_size, embedding_size), where embedding_size
        is 400 by default.

        """

        embeddings = self.get_embeddings_batch(article_ids)
        embeddings = torch.tensor(embeddings, dtype=torch.float32)
        n = self.fcout(embeddings)

        return n

    def get_embeddings(self, article_id: int) -> np.ndarray:
        """

        Gets the embeddings given an article id.

        The embeddings are a numpy array of size (768,)
        except for the word2vec model where the size is (300,)

        """

        embeddings = self.data.loc[article_id, "embeddings"]

        return embeddings  # type: ignore

    def get_embeddings_batch(self, article_ids: list[int]) -> np.ndarray:
        """

        Gets the embeddings for a batch of article ids.

        The embeddings are a numpy array of size (batch_size, 768)
        except for the word2vec model where the size is (batch_size, 300)

        """

        embeddings = self.data.loc[article_ids, "embeddings"]
        embeddings = np.stack(embeddings)  # type: ignore

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
