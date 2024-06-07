from typing import Literal
import os

from torch.nn import Module
import pandas as pd
import numpy as np

from ..utils import get_data_folder

TextEncodeModel = Literal["bert", "roberta", "word2vec", "contrastive"]


class LookupNewsEncoder:
    """

    Songga told us to first use the data in one of the artifacts as a newencoder.
    No need to make this a PyTorch module, no trainable parameters.

    """

    def __init__(self, model: TextEncodeModel, data_folder: str | None = None):
        super().__init__()

        data_folder = get_data_folder(data_folder)

        parquet_file_path = _parquet_file_path_from_model(model, data_folder)

        self.data = pd.read_parquet(parquet_file_path)
        self.data.set_index("article_id", inplace=True)

        embeddings_column_name = _embeddings_column_name_from_model(model)
        self.data.rename(columns={embeddings_column_name: "embeddings"}, inplace=True)

    def get_embeddings(self, article_id: int) -> np.ndarray:
        """

        Gets the embeddings given an article id.

        The embeddings are a numpy array of size (768,)
        except for the word2vec model where the size is (300,)

        """

        embeddings = self.data.loc[article_id, "embeddings"]

        return embeddings # type: ignore

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
