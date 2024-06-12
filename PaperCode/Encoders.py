from NewsContent import *
from UserContent import *
from preprocessing import *
from models import *
from utils import *

import os
import click
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from collections import Counter
import pickle
import random


import numpy
from sklearn.metrics import roc_auc_score
import keras
from keras.utils.np_utils import *
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding, concatenate
from keras.layers import Dense, Input, Flatten, average, Lambda

from keras.layers import *
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras import backend as K
import keras.layers as layers
from keras.engine.topology import Layer, InputSpec
from keras import initializers  # keras2
from keras.utils import plot_model
from keras.optimizers import *


# Used in the news encoder to encode the title and body of the news
def get_doc_encoder(config, text_length, embedding_layer):
    news_encoder = config["news_encoder_name"]
    sentence_input = Input(shape=(text_length,), dtype="int32")
    embedded_sequences = embedding_layer(sentence_input)
    d_et = Dropout(0.2)(embedded_sequences)
    if news_encoder == "CNN":
        l_cnnt = Conv1D(400, kernel_size=3, activation="relu")(d_et)
        # l_cnnt =Attention(20,20)([d_et,d_et,d_et])
    elif news_encoder == "SelfAtt":
        l_cnnt = Attention(20, 20)([d_et, d_et, d_et])
    elif news_encoder == "SelfAttPE":
        d_et = PositionEmbedding(title_length, 300)(d_et)
        l_cnnt = Attention(20, 20)([d_et, d_et, d_et])
    d_ct = Dropout(0.2)(l_cnnt)
    l_att = AttentivePooling(text_length, 400)(d_ct)
    sentEncodert = Model(sentence_input, l_att)
    return sentEncodert


# Pepijn: used in the news_encoder, dont know what vert and subvert are but yeah
def get_vert_encoder(
    config,
    vert_num,
):
    vert_input = Input(shape=(1,))
    embedding_layer = Embedding(vert_num + 1, 400, trainable=True)
    vert_emb = embedding_layer(vert_input)
    vert_emb = keras.layers.Reshape((400,))(vert_emb)
    vert_emb = Dropout(0.2)(vert_emb)
    model = Model(vert_input, vert_emb)
    return model


# Our KnowledgeAwareNewsEncoder module in news_encoder.py. It created news embeddings $n$
# based on the articles. But beware that this news encoder is used in three places in the
# overall architecture. To create the news embeddings used in the user encoder, to create
# the news embeddings used in the popularity predictor, and to create the news embeddings
# that are dot-producted with the user embeddings to create the final personalized matching
# scores. Its gets even more confusing, because the news encoder is used in the user encoder
# does share weights with the news encoder user to create the news embeddings that will be
# dot-producted with the user embeddings. But the news encoder used in the popularity predictor
# does not share their weights. Look at the create_pe_model function to see this in action.
# Over there they instantiate two news encoders, `news_encoder` (the one used twice) and
# `bias_news_encoder` (the one used once, for the popularity predictor).
def get_news_encoder(
    config,
    vert_num,
    subvert_num,
    word_num,
    word_embedding_matrix,
    entity_embedding_matrix,
):
    LengthTable = {
        "title": config["title_length"],
        "body": config["body_length"],
        "vert": 1,
        "subvert": 1,
        "entity": config["max_entity_num"],
    }
    input_length = 0
    PositionTable = {}
    for v in config["attrs"]:
        PositionTable[v] = (input_length, input_length + LengthTable[v])
        input_length += LengthTable[v]
    print(PositionTable)
    word_embedding_layer = Embedding(
        word_num + 1,
        word_embedding_matrix.shape[1],
        weights=[word_embedding_matrix],
        trainable=True,
    )

    news_input = Input((input_length,), dtype="int32")

    title_vec = None
    body_vec = None
    vert_vec = None
    subvert_vec = None
    entity_vec = None

    if "title" in config["attrs"]:
        title_input = keras.layers.Lambda(
            lambda x: x[:, PositionTable["title"][0] : PositionTable["title"][1]]
        )(news_input)
        title_encoder = get_doc_encoder(
            config, LengthTable["title"], word_embedding_layer
        )
        title_vec = title_encoder(title_input)

    if "body" in config["attrs"]:
        body_input = keras.layers.Lambda(
            lambda x: x[:, PositionTable["body"][0] : PositionTable["body"][1]]
        )(news_input)
        body_encoder = get_doc_encoder(
            config, LengthTable["body"], word_embedding_layer
        )
        body_vec = body_encoder(body_input)

    if "vert" in config["attrs"]:

        vert_input = keras.layers.Lambda(
            lambda x: x[:, PositionTable["vert"][0] : PositionTable["vert"][1]]
        )(news_input)
        vert_encoder = get_vert_encoder(config, vert_num)
        vert_vec = vert_encoder(vert_input)

    if "subvert" in config["attrs"]:
        subvert_input = keras.layers.Lambda(
            lambda x: x[:, PositionTable["subvert"][0] : PositionTable["subvert"][1]]
        )(news_input)
        subvert_encoder = get_vert_encoder(config, subvert_num)
        subvert_vec = subvert_encoder(subvert_input)

    if "entity" in config["attrs"]:
        entity_input = keras.layers.Lambda(
            lambda x: x[:, PositionTable["entity"][0] : PositionTable["entity"][1]]
        )(news_input)
        entity_embedding_layer = Embedding(
            entity_embedding_matrix.shape[0],
            entity_embedding_matrix.shape[1],
            trainable=False,
        )
        entity_emb = entity_embedding_layer(entity_input)
        entity_vecs = Attention(20, 20)([entity_emb, entity_emb, entity_emb])
        entity_vec = AttentivePooling(LengthTable["entity"], 400)(entity_vecs)

    vec_Table = {
        "title": title_vec,
        "body": body_vec,
        "vert": vert_vec,
        "subvert": subvert_vec,
        "entity": entity_vec,
    }
    feature = []
    for attr in config["attrs"]:
        feature.append(vec_Table[attr])
    if len(feature) == 1:
        news_vec = feature[0]
    else:
        for i in range(len(feature)):
            feature[i] = keras.layers.Reshape((1, 400))(feature[i])
        news_vecs = keras.layers.Concatenate(axis=1)(feature)
        news_vec = AttentivePooling(len(config["attrs"]), 400)(news_vecs)
    model = Model(news_input, news_vec)
    return model


# Pepijn: This isnt used anywhere
def create_model(config, News, word_embedding_matrix, entity_embedding_matrix):
    max_clicked_news = config["max_clicked_news"]

    news_encoder = get_news_encoder(
        config,
        len(News.category_dict),
        len(News.subcategory_dict),
        len(News.word_dict),
        word_embedding_matrix,
        entity_embedding_matrix,
    )
    news_input_length = int(news_encoder.input.shape[1])
    print(news_input_length)
    clicked_input = Input(
        shape=(
            max_clicked_news,
            news_input_length,
        ),
        dtype="int32",
    )
    print(clicked_input.shape)
    user_vecs = TimeDistributed(news_encoder)(clicked_input)

    if config["user_encoder_name"] == "SelfAtt":
        user_vecs = Attention(20, 20)([user_vecs, user_vecs, user_vecs])
        user_vecs = Dropout(0.2)(user_vecs)
        user_vec = AttentivePooling(max_clicked_news, 400)(user_vecs)
    elif config["user_encoder_name"] == "Att":
        user_vecs = Dropout(0.2)(user_vecs)
        user_vec = AttentivePooling(max_clicked_news, 400)(user_vecs)
    elif config["user_encoder_name"] == "GRU":
        user_vecs = Dropout(0.2)(user_vecs)
        user_vec = GRU(400, activation="tanh")(user_vecs)

    candidates = keras.Input(
        (
            1 + config["npratio"],
            news_input_length,
        ),
        dtype="int32",
    )
    candidate_vecs = TimeDistributed(news_encoder)(candidates)
    score = keras.layers.Dot(axes=-1)([user_vec, candidate_vecs])
    logits = keras.layers.Activation(keras.activations.softmax, name="recommend")(score)

    model = Model([candidates, clicked_input], [logits])

    model.compile(
        loss=["categorical_crossentropy"], optimizer=Adam(lr=0.0001), metrics=["acc"]
    )

    user_encoder = Model([clicked_input], user_vec)

    return model, user_encoder, news_encoder


# Pepijn: Alternative version for get_news_encoder in create_pe_model, settable in config. This one is default in Main.ipynb
def get_news_encoder_co1(
    config,
    vert_num,
    subvert_num,
    word_num,
    word_embedding_matrix,
    entity_embedding_matrix,
):
    LengthTable = {
        "title": config["title_length"],
        "vert": 1,
        "subvert": 1,
        "entity": config["max_entity_num"],
    }
    input_length = 0
    PositionTable = {}
    for v in config["attrs"]:
        PositionTable[v] = (input_length, input_length + LengthTable[v])
        input_length += LengthTable[v]
    print(PositionTable)
    word_embedding_layer = Embedding(
        word_num + 1,
        word_embedding_matrix.shape[1],
        weights=[word_embedding_matrix],
        trainable=True,
    )

    news_input = Input((input_length,), dtype="int32")

    vert_input = keras.layers.Lambda(
        lambda x: x[:, PositionTable["vert"][0] : PositionTable["vert"][1]]
    )(news_input)
    vert_embedding_layer = Embedding(vert_num + 1, 200, trainable=True)
    vert_emb = vert_embedding_layer(vert_input)
    vert_emb = keras.layers.Reshape((200,))(vert_emb)
    vert_vec = Dropout(0.2)(vert_emb)

    title_input = keras.layers.Lambda(
        lambda x: x[:, PositionTable["title"][0] : PositionTable["title"][1]]
    )(news_input)
    title_emb = word_embedding_layer(title_input)
    title_emb = Dropout(0.2)(title_emb)

    entity_input = keras.layers.Lambda(
        lambda x: x[:, PositionTable["entity"][0] : PositionTable["entity"][1]]
    )(news_input)
    entity_embedding_layer = Embedding(
        entity_embedding_matrix.shape[0],
        entity_embedding_matrix.shape[1],
        trainable=True,
    )
    entity_emb = entity_embedding_layer(entity_input)

    title_co_emb = Attention(5, 40)([title_emb, entity_emb, entity_emb])
    entity_co_emb = Attention(5, 40)([entity_emb, title_emb, title_emb])

    title_vecs = Attention(20, 20)([title_emb, title_emb, title_emb])
    title_vecs = keras.layers.Concatenate(axis=-1)([title_vecs, title_co_emb])
    title_vecs = Dense(400)(title_vecs)
    title_vecs = Dropout(0.2)(title_vecs)
    title_vec = AttentivePooling(config["title_length"], 400)(title_vecs)

    entity_vecs = Attention(20, 20)([entity_emb, entity_emb, entity_emb])
    entity_vecs = keras.layers.Concatenate(axis=-1)([entity_vecs, entity_co_emb])
    entity_vecs = Dense(400)(entity_vecs)
    entity_vecs = Dropout(0.2)(entity_vecs)
    entity_vec = AttentivePooling(LengthTable["entity"], 400)(entity_vecs)

    feature = [title_vec, entity_vec, vert_vec]

    news_vec = keras.layers.Concatenate(axis=-1)(feature)
    news_vec = Dense(400)(news_vec)
    model = Model(news_input, news_vec)
    return model


# Pepijn: not used anywhere, just co1 and the regular one
def get_news_encoder_co2(
    config,
    vert_num,
    subvert_num,
    word_num,
    word_embedding_matrix,
    entity_embedding_matrix,
):
    LengthTable = {
        "title": config["title_length"],
        "vert": 1,
        "subvert": 1,
        "entity": config["max_entity_num"],
    }
    input_length = 0
    PositionTable = {}
    for v in config["attrs"]:
        PositionTable[v] = (input_length, input_length + LengthTable[v])
        input_length += LengthTable[v]
    print(PositionTable)
    word_embedding_layer = Embedding(
        word_num + 1,
        word_embedding_matrix.shape[1],
        weights=[word_embedding_matrix],
        trainable=True,
    )

    news_input = Input((input_length,), dtype="int32")

    vert_input = keras.layers.Lambda(
        lambda x: x[:, PositionTable["vert"][0] : PositionTable["vert"][1]]
    )(news_input)
    vert_embedding_layer = Embedding(vert_num + 1, 200, trainable=True)
    vert_emb = vert_embedding_layer(vert_input)
    vert_emb = keras.layers.Reshape((200,))(vert_emb)
    vert_vec = Dropout(0.2)(vert_emb)

    title_input = keras.layers.Lambda(
        lambda x: x[:, PositionTable["title"][0] : PositionTable["title"][1]]
    )(news_input)
    title_emb = word_embedding_layer(title_input)
    title_emb = Dropout(0.2)(title_emb)
    title_vecs = Attention(20, 20)([title_emb, title_emb, title_emb])

    entity_input = keras.layers.Lambda(
        lambda x: x[:, PositionTable["entity"][0] : PositionTable["entity"][1]]
    )(news_input)
    entity_embedding_layer = Embedding(
        entity_embedding_matrix.shape[0],
        entity_embedding_matrix.shape[1],
        trainable=False,
    )
    entity_emb = entity_embedding_layer(entity_input)
    entity_vecs = Attention(20, 20)([entity_emb, entity_emb, entity_emb])

    title_co_emb = Attention(5, 40)([title_vecs, entity_vecs, entity_vecs])
    entity_co_emb = Attention(5, 40)([entity_vecs, title_vecs, title_vecs])

    title_vecs = keras.layers.Concatenate(axis=-1)([title_vecs, title_co_emb])
    title_vecs = Dense(400)(title_vecs)
    title_vecs = Dropout(0.2)(title_vecs)
    title_vec = AttentivePooling(config["title_length"], 400)(title_vecs)

    entity_vecs = keras.layers.Concatenate(axis=-1)([entity_vecs, entity_co_emb])
    entity_vecs = Dense(400)(entity_vecs)
    entity_vecs = Dropout(0.2)(entity_vecs)
    entity_vec = AttentivePooling(LengthTable["entity"], 400)(entity_vecs)

    feature = [title_vec, entity_vec, vert_vec]

    news_vec = keras.layers.Concatenate(axis=-1)(feature)
    news_vec = Dense(400)(news_vec)
    model = Model(news_input, news_vec)
    return model


# Pepijn: not used anywhere, just co1 and the regular one
def get_news_encoder_co3(
    config,
    vert_num,
    subvert_num,
    word_num,
    word_embedding_matrix,
    entity_embedding_matrix,
):
    LengthTable = {
        "title": config["title_length"],
        "vert": 1,
        "subvert": 1,
        "entity": config["max_entity_num"],
    }
    input_length = 0
    PositionTable = {}
    for v in config["attrs"]:
        PositionTable[v] = (input_length, input_length + LengthTable[v])
        input_length += LengthTable[v]
    print(PositionTable)
    word_embedding_layer = Embedding(
        word_num + 1,
        word_embedding_matrix.shape[1],
        weights=[word_embedding_matrix],
        trainable=True,
    )

    news_input = Input((input_length,), dtype="int32")

    vert_input = keras.layers.Lambda(
        lambda x: x[:, PositionTable["vert"][0] : PositionTable["vert"][1]]
    )(news_input)
    vert_embedding_layer = Embedding(vert_num + 1, 200, trainable=True)
    vert_emb = vert_embedding_layer(vert_input)
    vert_emb = keras.layers.Reshape((200,))(vert_emb)
    vert_vec = Dropout(0.2)(vert_emb)

    title_input = keras.layers.Lambda(
        lambda x: x[:, PositionTable["title"][0] : PositionTable["title"][1]]
    )(news_input)
    title_emb = word_embedding_layer(title_input)
    title_emb = Dropout(0.2)(title_emb)
    title_vecs = Attention(20, 20)([title_emb, title_emb, title_emb])
    title_vecs = Dropout(0.2)(title_vecs)
    title_vec = AttentivePooling(config["title_length"], 400)(title_vecs)

    entity_input = keras.layers.Lambda(
        lambda x: x[:, PositionTable["entity"][0] : PositionTable["entity"][1]]
    )(news_input)
    entity_embedding_layer = Embedding(
        entity_embedding_matrix.shape[0],
        entity_embedding_matrix.shape[1],
        trainable=False,
    )
    entity_emb = entity_embedding_layer(entity_input)
    entity_vecs = Attention(20, 20)([entity_emb, entity_emb, entity_emb])
    entity_vecs = Dropout(0.2)(entity_vecs)
    entity_vec = AttentivePooling(LengthTable["entity"], 400)(entity_vecs)

    title_query_vec = keras.layers.Reshape((1, 400))(title_vec)
    entity_query_vec = keras.layers.Reshape((1, 400))(entity_vec)
    title_co_vec = Attention(
        1,
        100,
    )([entity_query_vec, title_vecs, title_vecs])
    entity_co_vec = Attention(
        1,
        100,
    )([title_query_vec, entity_vecs, entity_vecs])
    title_co_vec = keras.layers.Reshape((100,))(title_co_vec)
    entity_co_vec = keras.layers.Reshape((100,))(entity_co_vec)

    title_vec = keras.layers.Concatenate(axis=-1)([title_vec, title_co_vec])
    entity_vec = keras.layers.Concatenate(axis=-1)([entity_vec, entity_co_vec])
    title_vec = Dense(400)(title_vec)
    entity_vec = Dense(400)(entity_vec)
    feature = [title_vec, entity_vec, vert_vec]

    news_vec = keras.layers.Concatenate(axis=-1)(feature)
    news_vec = Dense(400)(news_vec)
    model = Model(news_input, news_vec)
    return model


# Pepijn: This is not just the popularity encoder, or the popularity embedding module. This is the full PPRec model.
# Confusing fucking name hahah.
def create_pe_model(
    config, model_config, News, word_embedding_matrix, entity_embedding_matrix
):
    # Pepijn: max_clicked in our code, in user_encoder.py. Added a max_clicked_news config parameter to our code.
    max_clicked_news = config["max_clicked_news"]

    # Pepijn, news encoder 0 is the news encoder that uses all article attributes.
    # co1 doesnt use the body (the else statement) does not, and if for the w/o news
    # content bar in figure 10 in the paper.
    if model_config["news_encoder"] == 0:

        # Pepijn: A news encoder. But this one is used in two occasions. If you look at
        # figure 2 (the whole architecture), this covers the right side of the figure.
        # The Knowledge-aware News Encoder in the middle, and also in the Popularity-aware User
        # Encoder. If we zoom in on the Popularity-aware User Encoder by looking at figure 5,
        # We see it again as the News Encoder module. So what we have to rememeber is that
        # this news encoder is used in two places, and the weights are shared.
        news_encoder = get_news_encoder(
            config,
            len(News.category_dict),
            len(News.subcategory_dict),
            len(News.word_dict),
            word_embedding_matrix,
            entity_embedding_matrix,
        )

        # Pepijn: A second news encoder. This one is only used in the Time-aware Popularity Predictor,
        # in the left side of figure 2. If we zoom in on this popularity predictor, by looking at
        # figure 4, we see it again as the News Encoder module. A thing to note is that this `bias`
        # prefix is the authors slang to denote that its about the popularity predictor. They use
        # this prefix more often in this code.
        bias_news_encoder = get_news_encoder(
            config,
            len(News.category_dict),
            len(News.subcategory_dict),
            len(News.word_dict),
            word_embedding_matrix,
            entity_embedding_matrix,
        )

    # Pepijn: Like I stated above, this option is to not use the body of the article, for
    # the experiment in figure 10 in the paper. For the w/o news content bar.
    elif model_config["news_encoder"] == 1:

        # Pepijn: Check above for more info on the difference between these
        # two news encoders. To summarize, this one is used in the Popularity-aware User
        # and the Knowledge-aware News Encoder.
        news_encoder = get_news_encoder_co1(
            config,
            len(News.category_dict),
            len(News.subcategory_dict),
            len(News.word_dict),
            word_embedding_matrix,
            entity_embedding_matrix,
        )

        # Pepijn: Check above for more info. This one is only used in the Time-aware Popularity Predictor.
        bias_news_encoder = get_news_encoder_co1(
            config,
            len(News.category_dict),
            len(News.subcategory_dict),
            len(News.word_dict),
            word_embedding_matrix,
            entity_embedding_matrix,
        )

    # Pepijn: the articles vector size. This is the concatenated size of the title
    # length (in tokens or words I think), body length (in tokens as well) those
    # verts (no idea what this is yet) and lastly entity length (in number of entities),
    # in this order. In the co1 version there is no body though. Every token, entity
    # or vert is an int, some 'vocabulary' index. Note: I suspect its number of tokens,
    # but this could well be something else. The config in Main.ipynb uses a body_length
    # of 100, which seems quite small...
    #
    # When we implement the actual news encoder I think its clearer to keep these
    # separate, and not concatenate them before input? We can concatenate them in the
    # module right. So we can avoid this lookup table thing in `get_news_encoder` etc.
    news_input_length = int(news_encoder.input.shape[1])
    print(news_input_length)

    # Pepijn: Now comes the implementation of the user encoder. So our user_encoder.py.

    # Pepijn: the article vectors for all historically clicked inputs.
    clicked_input = Input(
        shape=(
            max_clicked_news,
            news_input_length,
        ),
        dtype="int32",
    )

    # Pepijn: in our user_encoder.py: ctr input to PopularityEmbedding?
    # We can see here ctr is an int, so maybe how many times the article
    # was clicked? But a click-through should be a percentage, the number
    # of clicks over the number of impressions. So therefore I feel
    # like this is just the ctr as a precentage, but scaled to the range
    # of 0-200. Since the popularity embedding layer they use has a vocab
    # size of 200. I checked their data loader, and yes, its the ctr scaled
    # to 0-200.
    clicked_ctr = Input(shape=(max_clicked_news,), dtype="int32")
    print(clicked_input.shape)

    # Pepijn: This TimeDistributed stuff they use everywhere, its
    # just a fancy way to have a second 'batch' dimension. For instance
    # our news encoder expects a batch of news articles, but the user
    # encoder needs to encode the news of all the clicked articles of a user,
    # for every user in the batch. So in pytorch there is no tool to do this,
    # we would just merge the batch and the clicked articles dimension, do
    # the encoding, and then split them again. Nothing special.

    # Pepijn: user_vecs are the news embeddings for all clicked articles by a user. So in our
    # code n, the input to the NewsSelfAttention module for the user encoder.
    # So they use TimeDistributes to apply the news encoder to all clicked articles.
    user_vecs = TimeDistributed(news_encoder)(clicked_input)

    # Pepijn: In our code PopularityEmbedding in the user encoder.
    popularity_embedding_layer = Embedding(200, 400, trainable=True)

    # Pepijn: The output from the PopularityEmbedding module in the user encoder,
    # creating the popularity embeddings p, based on the click through rate's ctr.
    popularity_embedding = popularity_embedding_layer(clicked_ctr)

    # Pepijn: For the experiment in figure 9. They can turn of the popularity embeddings
    # in the user encoder, to create the w/o news popularity bar in the plot. This first
    # if is the regular option, so with the popularity embeddings.
    if model_config["popularity_user_modeling"]:

        # Pepijn: For some reason these guys create it twice. At first I thought it was
        # bacause the popularity embedding was also used for something else, maybe the
        # popularity predictor (even though no stated in the paper though) but they dont.
        # So we can safely assume its just a mistake. No difference I think haha.
        # They probably just forgot they already it two lines above. Check those comments.
        popularity_embedding_layer = Embedding(200, 400, trainable=True)

        # Pepijn: Relevant to the comment above. Check the comments a few lines above.
        popularity_embedding = popularity_embedding_layer(clicked_ctr)

        # Pepijn: Our NewsSelfAttention module in the user encoder.
        MHSA = Attention(20, 20)

        # Pepijn: Like the forward function in our NewsSelfAttention module. So the output
        # are the m vectors, the called 'contextual news representation' in the paper.
        # Annoying they still call is user vecs haha.
        user_vecs = MHSA([user_vecs, user_vecs, user_vecs])

        # Pepijn: Mmmm why is this commented out.
        # user_vec_query = keras.layers.Add()([user_vecs,popularity_embedding])

        # Pepijn: This stuff under does the same as our Content-
        # PopularityJoinAttention module. I think they found it
        # more logical to implement their CPJA stuff by formulating
        # it in terms of AttentivePooling.
        user_vec_query = keras.layers.Concatenate(axis=-1)(
            [user_vecs, popularity_embedding]
        )

        # Pepijn: so this final user_vec is the output of the user encoder, the user embeddings
        # u. So the output of the PopularityAwareUserEncoer module in user_encoder.py. The output
        # of the Popularity-aware User Encoder in figure 2 in the paper.
        user_vec = AttentivePoolingQKY(50, 800, 400)([user_vec_query, user_vecs])

    # Pepijn: This is for w/o news popularity in figure 9.
    else:

        # Pepijn: Same as above, but without the popularity embeddings. ContentPopularityJoinAttention stuff.
        user_vecs = Attention(20, 20)([user_vecs, user_vecs, user_vecs])
        user_vecs = Dropout(0.2)(user_vecs)

        # Pepijn: The output of the user encoder, the user embeddings u. So the output of the
        # PopularityAwareUserEncoer module in user_encoder.py. The output of the Popularity-aware
        # User Encoder in figure 2 in the paper. But of course, without the popularity embeddings used
        # in the process.
        user_vec = AttentivePooling(max_clicked_news, 400)(user_vecs)

    # Pepijn: This is were the user encoder stuff ends, and the popularity
    # predictor begins. So our popularity_predictor.py.

    # Pepijn: The articles input vector for the candidate articles. The `npratio` is the negative
    # to positive ratio. So to be able to have two or more of them during training, one
    # positive and some negatives. So one that the user eventually clicked, and some that
    # the user didnt click. In the paper they don't formulate the loss function to have
    # this possibility, just one negative and one positive, but its possible of course.
    # But probably does not give better results.
    candidates = keras.Input(
        (
            1 + config["npratio"],
            news_input_length,
        ),
        dtype="int32",
    )

    # Pepijn: The click through rate of the candidate articles. This time the crt is a float, so
    # probably the actual click through rate. So the number of clicks over the number of
    # impressions. So not scaled to a range, like for the user encoder!
    #  So this is the crt input to our TimeAwarePopularityPredictor in
    # popularity_predictor.py. Again, note the npratio to facilitate one positive and some
    # negative examples.
    candidates_ctr = keras.Input((1 + config["npratio"],), dtype="float32")

    # Pepijn: The recencies of the candidate articles. So the time since the article was published,
    # in hours, rounded, I think. The `recencies`` input to our TimeAwarePopularityPredictor in
    # popularity_predictor.py. Again, note the npratio to facilitate one positive and
    # some negative examples. Ok, I just checked the data loader, and the recency is the
    # not the actual number of hours since the article was published. They use some function
    # compute_Q_publish to compute the final recency value, but what this function does is
    # `time_since_publishing // 2`. So the recency is the number of hours since the article
    # was published, but divided by two (and rounded). Its seems like this is a simple trick to
    # get it into the range of the Embedding layer `time_embedding_layer` you can find some lines
    # below. There probably is no data clipping needed, as the recency/2 is always less than 1500.
    # Or they checked manually and found 1500 was the maximum recency. But its probably not exactly
    # 1500, but something a little less, making this embedding layer not as powerfull as it can be,
    # as theres some unused slots this way. Think we can do better that just dividing by two choosing
    # finding the value needed for the input dimensions of the recency embedding layer.
    candidates_rece_emb_index = keras.Input((1 + config["npratio"],), dtype="int32")

    # Pepijn: Config to create w/o news recency in figure 10 in the paper. First the
    # regular 'with recency' case.
    if model_config["rece_emb"]:

        # Pepijn: The combined vector containing both the recency embeddings r, and the news
        # embeddings n (the inputs to our TimeAwarePopularityPredictor in popularity_predictor.py).
        # 400 for the news embeddings, and 100 for the recency embeddings, making 500
        bias_content_vec = Input(shape=(500,))

        # Pepijn: Here they split them up in what we call n and r in our code and in the Paper.
        # Why werent they split up in the first case? Makes the code a lot more unreadable.

        # Pepijn: Separating the news embeddings n, the first 400 elements of the combined vector.
        vec1 = keras.layers.Lambda(lambda x: x[:, :400])(bias_content_vec)

        # Pepijn: The recency embeddings r, the last 100 elements of the combined vector.
        # You might think, where is the recency embedding done? RecencyEmbedding module in
        # our code: well its after this if statement. 'Beauty' of keras, you can just
        # define stuff in random order.
        vec2 = keras.layers.Lambda(lambda x: x[:, 400:])(bias_content_vec)

        # Pepijn: Here the news embeddings are passed through the Dense network.
        # This is the right side Dense module in figure 4. In our code this Dense network
        # is called ContentBasedPopularityDense. In the paper they say the Dense networks
        # all have two layers, with 100 dimensional hidden units. But this one has 256/128
        # dimensional hidden units, and wouldnt you call this a four layer network as well?
        # Annoying. All we know is that the paper is lying.
        vec1 = Dense(256, activation="tanh")(vec1)
        vec1 = Dense(256, activation="tanh")(vec1)
        vec1 = Dense(
            128,
        )(vec1)

        # Pepijn: The output of ContentBasedPopularityDense in our code. So $\hat{p}_c$ or pc
        # This confirms the output should be a scalar.
        bias_content_score = Dense(1, use_bias=False)(vec1)

        # Pepijn: Similarly, a small neural network functioning as teh left side
        # Dense in figure 4. In our code this Dense network is called RecencyBasedPopularityDense.
        # Just like the content Dense network, their lying in the paper about dimensions.
        # Its a three (not two, or would you call this two? idk) layer network, with
        # 64 dimensional hidden units, not 100.
        vec2 = Dense(64, activation="tanh")(vec2)
        vec2 = Dense(64, activation="tanh")(vec2)

        # Pepijn: The output of RecencyBasedPopularityDense in our code. So $\hat{p}_r$ or pr
        bias_recency_score = Dense(1, use_bias=False)(vec2)

        # Pepijn: All under here is the gate, but it surely is different from the one they explain
        # in formula (1) in section 3.3 in the paper... This might explain why in this formula
        # Wp is denoted as a matrix, but should be a vector, since otherwise theta wouldnt be a
        # scalar. Here they use a three/two (idk anymore) layer network, instead of 'a single layer'.
        # At least, this small network is called ContentRecencyGate in our code.
        gate = Dense(128, activation="tanh")(bias_content_vec)
        gate = Dense(64, activation="tanh")(gate)

        # Pepijn: The output of this ContentRecencyGate network. So $\theta$ in the paper and our code.
        gate = Dense(1, activation="sigmoid")(gate)

        # Pepijn: This piece of code is in the top level TimeAwarePopularityPredictor in our code.
        # The output is the final popularity score $\hat{p}$, which is, as the paper states
        # the weighted sum of the content and recency popularity scores $\hat{p}_c$ and $\hat{p}_r$.
        bias_content_score = keras.layers.Lambda(
            lambda x: (1 - x[0]) * x[1] + x[0] * x[2]
        )([gate, bias_content_score, bias_recency_score])

        # Pepijn: Combining the inputs and outputs to create the model. So this is almost the
        # TimeAwarePopularityPredictor in our code, only that the ctrs arent incorporated yet.
        # The thing outputted by this model would be the popularity score $\hat{p}$ or p, not yet
        # $s_p$.
        bias_content_scorer = Model(bias_content_vec, bias_content_score)

    # Pepijn: This is the w/o news recency case in figure 10.
    else:

        # Pepijn: So then the vec is just the news embeddings n with length 400.
        bias_content_vec = Input(shape=(400,))

        # Pepijn: More info about this in the comments in the 'with recency' case above.
        # In our code this network is called ContentBasedPopularityDense. And the paper is
        # lying about dimensions!
        vec = Dense(256, activation="tanh")(bias_content_vec)
        vec = Dense(256, activation="tanh")(vec)
        vec = Dense(
            128,
        )(vec)

        # Pepijn: The output of ContentBasedPopularityDense in our code. So $\hat{p}_c$ or pc
        bias_content_score = Dense(1, use_bias=False)(vec)

        # Pepijn: Combining the inputs and outputs to create the model. So again, this is almost
        # the TimeAwarePopularityPredictor in our code, but no crts used yet. So the output of this model
        # would be the popularity score $\hat{p}$ or p, not yet $s_p$. And of course in this else statement
        # the recencies are skipped, making it just $\hat{p}_c$ or pc. So pc == p in this case.
        bias_content_scorer = Model(bias_content_vec, bias_content_score)

    # Pepijn: And finally, the RecencyEmbedding module in our code.
    time_embedding_layer = Embedding(1500, 100, trainable=True)

    # Pepijn: The recency embeddings r, the output of the RecencyEmbedding module in our code.
    time_embedding = time_embedding_layer(candidates_rece_emb_index)

    # Pepijn: These `candidate_vecs` are the news embeddings n that are going to be dot-producted
    # with the user embeddings u, to form the personalized matching scores. Take a look at figure 2
    # in the paper (the overall architecture), then it becomes clear. So from this we can see there
    candidate_vecs = TimeDistributed(news_encoder)(candidates)

    # Pepijn: These `bias_candidate_vecs` are the news embeddings n that are going to be used in the
    # Time-Aware Popularity Predictor when looking at figure 2. So this is the News Encoder module
    # when looking at figure 4.
    bias_candidate_vecs = TimeDistributed(bias_news_encoder)(candidates)

    # Pepijn: Remember from before, from the earlier rece_emb if statement, that their
    # bias_content_scorer (our TimeAwarePopularityPredictor) could expect either a 400 or 500
    # dimensional input. With or without the recency embeddings. For their experiment in figure 10.
    if model_config["rece_emb"]:
        bias_candidate_vecs = keras.layers.Concatenate(axis=-1)(
            [bias_candidate_vecs, time_embedding]
        )

    # Pepijn: `bias_candidate_score` would be $\hat{p}$ or p in the paper.
    # Not yet $s_p$! The full output of the Time-Aware Popularity Predictor.
    bias_candidate_score = TimeDistributed(bias_content_scorer)(bias_candidate_vecs)

    # Pepijn: Just some reshaping because the second dimension is now 1, so we can leave it out
    bias_candidate_score = keras.layers.Reshape((1 + config["npratio"],))(
        bias_candidate_score
    )

    # Pepijn: The personalized matching scores $s_m$ in the paper (look at figure 2).
    rel_scores = keras.layers.Dot(axes=-1)([user_vec, candidate_vecs])

    # Pepijn: Now they do some popularity predictor stuff again. This `scalar`, a
    # Dense layer with 1 unit, is just the scalar. weight for the crts in the
    # Time-Aware Popularity Predictor. So the $w_c$ in the paper, in figure 4.
    scaler = Dense(
        1, use_bias=False, kernel_initializer=keras.initializers.Constant(value=19)
    )

    # Pepijn: The click through rates, the crts, are scaled by the scalar weight $w_c$.
    # First some reshaping is needed to make it work with the overkill Dense layer.
    ctrs = keras.layers.Reshape((1 + config["npratio"], 1))(candidates_ctr)
    # Pepijn: Actually using the scalar weight $w_c$ on the crts.
    ctrs = scaler(ctrs)
    # Pepijn: And then they revert the reshaping that was needed.
    bias_ctr_score = keras.layers.Reshape((1 + config["npratio"],))(ctrs)

    # Pepijn: This is used later in creating the final `model`. This `model` is
    # returned, and in Main.ipynb, its only fitted/trained. Then the other stuff returned
    # is used for evaluation. Only the line right under `create_pe_model` in Main.ipynb uses
    # this `model`, namely: model.fit_generator(train_generator,epochs=2). So what I think has
    # happened, is that the train_generator also produces some 'activity' data, but this data
    # is not used in the model, as this `user_activity_input` thing you see here is not linked
    # to any outputs of the model, so its not used anywhere. The author was probably planning
    # to also use it, or was just too lazy to remove it from the train generator. In other words,
    # just ignore this line. If you look into the train generator, you see that the `activity`
    # is just the total number of clicks for the user. But yeah its not used anywhere.
    user_activity_input = keras.layers.Input((1,), dtype="int32")

    # Pepijn: This network here is used in the personalized aggregator gate they talk about in
    # 3.5 of the paper, used to produce $\eta$ in formula (3) based on the user embeddings u.
    # Just like for the gates in the Time-Aware Popularity Predictor, the authors lie about using
    # a two layer network, with 100 dimensional hidden units (they state this in section 4.1). Instead
    # they use this three layer network, with 128 and 64 dimensional hidden units you can see here.
    # In our code this is called PersonalizedAggregatorGate, which you can find in pprec.py.
    user_vec_input = keras.layers.Input(
        (400,),
    )
    activity_gate = Dense(128, activation="tanh")(user_vec_input)
    # Pepijn: And I think I even found an actual mistake by the authors here.
    # They abviously intended to input the `activity_gate` into the next layer,
    # but they accidentally input the `user_vec_input` into it again. So the netwerk
    # becomes 400 -> 64 -> 1, instead of 400 -> 128 -> 64 -> 1. Jeezzzz.
    activity_gate = Dense(64, activation="tanh")(user_vec_input)
    activity_gate = Dense(1, activation="sigmoid")(activity_gate)
    activity_gate = keras.layers.Reshape((1,))(activity_gate)
    activity_gater = Model(user_vec_input, activity_gate)
    # Pepijn: So this is the output of the gate, so $\eta$ in figure 2 in the paper.
    # Our output of the PersonalizedAggregatorGate in pprec.py.
    user_activtiy = activity_gater(user_vec)

    # Pepijn: In this part, the actual output of the full PPRec model is created.
    # So $s$, the ranking score. The result of formula (3) in section 3.5 in the paper.
    #
    # Some quick reminders make this clearer.
    #   rel_sores $s_m$ is the personalized matching score, how much the user embeddings $u$ match the news embeddings.
    #   bias_candidate_score $\hat{p}$ is content-based news popularity.
    #   bias_ctr_score these are the click through rates, already scaled by the scalar weight $w_c$.
    #
    # Something to note is that: The second scalar parameter/weight $w_p$ is missing.
    # Expanding on this topic: If you would try to get the output score just the regular way, the way
    # the authors describe in the paper, without all the experimenting options stuff. Then you would expect
    # the authors to first calculate the final popularity score $s_p$ by adding the content-based popularity
    # score $\hat{p}$ and the click through rates weighted by the parameter $w_c$. But then we're missing
    # something right? Namely, in figure 4 there also is a weight for the $\hat{p}$, called $w_p$. This second
    # weight is nowhere to be found in the code... I think this is because, in theory, the model has the same
    # expressability without this weight. The parameters for the part of the model creating the the personalized
    # matching scores $s_m$, and the other weight $w_c$, could just adjust to the absence of this weight. It does
    # not add any expressability to the model. So the authors probably just left it out to keep the number of
    # parameters lower, even if its just one. But it was probably still clearer to include it in the explanation
    # in the paper. To summarize this paragraph: the authors left out the $w_p$ weight which is shown in figure 4,
    # because it wouldnt improve the model.

    scores = []

    # Pepijn: This if adds scores if you want to have let the user encoder have influence on the scores.
    if model_config["rel"]:

        # Pepijn: This if adds scores if you cant to use use $\eta$ in the final score.
        # Same for the other two similar ifs. Aka you can turn of the influnce of the user
        # embeddings u on the personalized aggregator.
        if model_config["activity"]:
            print(user_activtiy.shape)
            print(rel_scores.shape)
            rel_scores = keras.layers.Lambda(lambda x: 2 * x[0] * x[1])(
                [rel_scores, user_activtiy]
            )
            print(rel_scores.shape)

        scores.append(rel_scores)

    # These two ifs add scores if you want to have the popularity encoder have influence on the scores.
    # This first if when you only if you want to have the content-based popularity score $\hat{p}$ in the final score.
    if model_config["content"]:
        if model_config["activity"]:
            bias_candidate_score = keras.layers.Lambda(lambda x: 2 * x[0] * (1 - x[1]))(
                [bias_candidate_score, user_activtiy]
            )
        scores.append(bias_candidate_score)

    # This second popularity encoder related if, when you only want to have the click through rates in the final score.
    if model_config["ctr"]:
        if model_config["activity"]:
            bias_ctr_score = keras.layers.Lambda(lambda x: 2 * x[0] * (1 - x[1]))(
                [bias_ctr_score, user_activtiy]
            )
        scores.append(bias_ctr_score)

    # Pepijn: Summing all these separate scores to get the final score $s$. Because this is Keras,
    # you cant just sum them, you have to use the Add layer.
    if len(scores) > 1:
        scores = keras.layers.Add()(scores)
    else:
        scores = scores[0]

    # Pepijn: And again something not stated in the paper. They softmax to make the logits of the
    # positive and negative article sum to one. This probably gives better results, maybe something
    # to do with the sigmoid in the loss function, making this a bit more stable.
    logits = keras.layers.Activation(keras.activations.softmax, name="recommend")(
        scores
    )

    # Pepijn: The full model used for training in Main.ipynb.
    model = Model(
        [
            candidates,  # The candidate articles (their content vectors)
            candidates_ctr,  # The click through rates of the candidate articles, floats probably between 0 and 1
            candidates_rece_emb_index,  # The recencies of the candidate articles
            user_activity_input,  # This one is not used in the model, so ignore it. Probably just a leftover from the train generator.
            clicked_input,  # The clicked articles (their content vectors)
            clicked_ctr,  # The click through rates of the clicked articles, ints probably between 0 and 200.
        ],
        [logits],
    )

    # Pepijn: This finds all computation steps in the model to map all the inputs we defined to outputs.
    # I the loss function in their paper can be rewritten as a categorical crossentropy loss, with two
    # classes, the positive and negative article. But yeah bit weird.
    model.compile(
        loss=["categorical_crossentropy"], optimizer=Adam(lr=0.0001), metrics=["acc"]
    )

    # Pepijn: they still formulate the final user encoder as a separe model, so they can use it doing
    # evaluation. In Keras, by training the `model` the weights of the `user_encoder`, as well as
    # all the other included and returned models are also trained. In our pytorch version, we could just
    # train an PPRec instance, lets call it `pprec_model`, and then use `ue = pprec_model.user_encoder` to
    # use and evaluate the user encoder `ue` separately.
    user_encoder = Model([clicked_input, clicked_ctr], user_vec)

    return (
        model,
        user_encoder,
        news_encoder,
        bias_news_encoder,
        bias_content_scorer,
        scaler,
        time_embedding_layer,
        activity_gater,
    )
