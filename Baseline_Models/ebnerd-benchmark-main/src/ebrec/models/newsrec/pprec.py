# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from ebrec.models.newsrec.layers import AttLayer2, SelfAttention, AttentivePoolingQKY,Attention
import tensorflow as tf
import numpy as np


class PPRecModel:
    """PPRec model
    """

    def __init__(
        self,
        hparams: dict,
        word2vec_embedding: np.ndarray = None,
        word_emb_dim: int = 300,
        vocab_size: int = 32000,
        seed: int = None,
    ):
        """Initialization steps for PPRec."""
        self.hparams = hparams
        self.seed = seed

        # SET SEED:
        tf.random.set_seed(seed)
        np.random.seed(seed)

        # INIT THE WORD-EMBEDDINGS:
        if word2vec_embedding is None:
            self.word2vec_embedding = np.random.rand(vocab_size, word_emb_dim)
        else:
            self.word2vec_embedding = word2vec_embedding

        # BUILD AND COMPILE MODEL:
        self.model, self.scorer = self._build_graph()
        data_loss = self._get_loss(self.hparams.loss)
        train_optimizer = self._get_opt(
            optimizer=self.hparams.optimizer, lr=self.hparams.learning_rate
        )
        self.model.compile(loss=data_loss, optimizer=train_optimizer)

    def _get_loss(self, loss: str):
        """Make loss function, consists of data loss and regularization loss
        Returns:
            object: Loss function or loss function name
        """
        if loss == "cross_entropy_loss":
            data_loss = "categorical_crossentropy"
        elif loss == "log_loss":
            data_loss = "binary_crossentropy"
        else:
            raise ValueError(f"this loss not defined {loss}")
        return data_loss

    def _get_opt(self, optimizer: str, lr: float):
        """Get the optimizer according to configuration. Usually we will use Adam.
        Returns:
            object: An optimizer.
        """
        # TODO: shouldn't be a string input you should just set the optimizer, to avoid stuff like this:
        # => 'WARNING:absl:At this time, the v2.11+ optimizer `tf.keras.optimizers.Adam` runs slowly on M1/M2 Macs, please use the legacy Keras optimizer instead, located at `tf.keras.optimizers.legacy.Adam`.'
        if optimizer == "adam":
            train_opt = tf.keras.optimizers.legacy.Adam(learning_rate=lr)
        else:
            raise ValueError(f"this optimizer not defined {optimizer}")
        return train_opt

    def _build_graph(self):
        """Build PPRec model and scorer.

        Returns:
            object: a model used to train.
            object: a model used to evaluate and inference.
        """
        model, scorer = self._build_pprec()
        return model, scorer

    def _build_userencoder(self, titleencoder):
        """The main function to create user encoder of PPRec.

        Args:
            titleencoder (object): the news encoder of PPRec.

        Return:
            object: the user encoder of PPRec.
        """
        his_input_title = tf.keras.Input(
            shape=(self.hparams.history_size, self.hparams.title_size), dtype="int32"
        )

        click_title_presents = tf.keras.layers.TimeDistributed(titleencoder)(
            his_input_title
        )
        y = SelfAttention(self.hparams.head_num, self.hparams.head_dim, seed=self.seed)(
            [click_title_presents] * 3
        )
        user_present = AttLayer2(self.hparams.attention_hidden_dim, seed=self.seed)(y)

        model = tf.keras.Model(his_input_title, user_present, name="user_encoder")
        return model

    def _build_newsencoder(self):
        """The main function to create news encoder of PPRec.

        Args:
            embedding_layer (object): a word embedding layer.

        Return:
            object: the news encoder of PPRec.
        """
        embedding_layer = tf.keras.layers.Embedding(
            self.word2vec_embedding.shape[0],
            self.word2vec_embedding.shape[1],
            weights=[self.word2vec_embedding],
            trainable=True,
        )
        sequences_input_title = tf.keras.Input(
            shape=(self.hparams.title_size,), dtype="int32"
        )
        embedded_sequences_title = embedding_layer(sequences_input_title)

        y = tf.keras.layers.Dropout(self.hparams.dropout)(embedded_sequences_title)
        y = SelfAttention(self.hparams.head_num, self.hparams.head_dim, seed=self.seed)(
            [y, y, y]
        )
        y = tf.keras.layers.Dropout(self.hparams.dropout)(y)
        pred_title = AttLayer2(self.hparams.attention_hidden_dim, seed=self.seed)(y)

        model = tf.keras.Model(sequences_input_title, pred_title, name="news_encoder")
        return model

    def _build_pprec(self):
        """The main function to create PPRec's logic. The core of PPRec
        is a user encoder and a news encoder.

        Returns:
            object: a model used to train.
            object: a model used to evaluate and inference.
        """

        his_input_title = tf.keras.Input(
            shape=(self.hparams.history_size, self.hparams.title_size),
            dtype="int32",
        )
        pred_input_title = tf.keras.Input(
            # shape = (hparams.npratio + 1, hparams.title_size)
            shape=(None, self.hparams.title_size),
            dtype="int32",
        )
        pred_input_title_one = tf.keras.Input(
            shape=(
                1,
                self.hparams.title_size,
            ),
            dtype="int32",
        )
        pred_title_one_reshape = tf.keras.layers.Reshape((self.hparams.title_size,))(
            pred_input_title_one
        )
        titleencoder = self._build_newsencoder()
        self.userencoder = self._build_userencoder(titleencoder)
        self.newsencoder = titleencoder

        user_present = self.userencoder(his_input_title)
        news_present = tf.keras.layers.TimeDistributed(self.newsencoder)(
            pred_input_title
        )
        news_present_one = self.newsencoder(pred_title_one_reshape)
        
        # Add popularity to user modelling
        # popularity_embedding_layer =  tf.keras.layers.Embedding(200, 400,trainable=True)
        # clicked_ctr  = tf.keras.Input(shape=(50,),dtype='int32')
        # news_input_length = int(self.newsencoder.input.shape[1])
        # clicked_input = tf.keras.Input(shape=(50, news_input_length,), dtype='int32')
        # user_present = tf.keras.layers.TimeDistributed(self.newsencoder)(clicked_input)
        # popularity_embedding = popularity_embedding_layer(clicked_ctr)
        # MHSA = Attention(50,50)
        # user_present = MHSA([user_present,user_present,user_present,user_present,user_present])
        # user_vec_query = tf.keras.layers.Concatenate(axis=-1)([user_present,popularity_embedding])
        # user_vec_query = AttentivePoolingQKY(50,2900,2500)([user_vec_query,user_present])
        
        
        
        preds = tf.keras.layers.Dot(axes=-1)([news_present, user_present])
        
        preds = tf.keras.layers.Activation(activation="softmax")(preds)
        pred_one = tf.keras.layers.Dot(axes=-1)([news_present_one, user_present])
        pred_one = tf.keras.layers.Activation(activation="sigmoid")(pred_one)
        model = tf.keras.Model([his_input_title, pred_input_title], preds)
        scorer = tf.keras.Model([his_input_title, pred_input_title_one], pred_one)

        
        
        return model, scorer
