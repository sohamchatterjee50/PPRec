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


# Pepijn: This is not just the popularity encoder, or the popularity embedding module. This is the PPRec model.
def create_pe_model(
    config, model_config, News, word_embedding_matrix, entity_embedding_matrix
):
    # Pepijn: max_clicked in our code, in user_encoder.py. Added a max_clicked_news config parameter to our code.
    max_clicked_news = config["max_clicked_news"]

    # Pepijn, news encoder 0 is the news encoder that uses all article attributes. 1 doesnt use the body.
    # For the w/o news content bar in figure 10 in the paper.
    if model_config["news_encoder"] == 0:
        # Pepijn: Our news_encoder.py
        news_encoder = get_news_encoder(
            config,
            len(News.category_dict),
            len(News.subcategory_dict),
            len(News.word_dict),
            word_embedding_matrix,
            entity_embedding_matrix,
        )
        # Pepijn: Our news_encoder.py. And this is the one used in the popularity encoder it seems.
        # This means that the weights arent shared? Maybe only the embedding matrices, but I feel like
        # these are just starting weights, they dont share the same memory on gpu.
        bias_news_encoder = get_news_encoder(
            config,
            len(News.category_dict),
            len(News.subcategory_dict),
            len(News.word_dict),
            word_embedding_matrix,
            entity_embedding_matrix,
        )

    # Pepijn: Like I stated above, this option is to not use the body of the article, for
    # the experiment in figure 10 in the paper.
    elif model_config["news_encoder"] == 1:

        # Pepijn: our news encoder, but which doesnt use the body of the articles
        news_encoder = get_news_encoder_co1(
            config,
            len(News.category_dict),
            len(News.subcategory_dict),
            len(News.word_dict),
            word_embedding_matrix,
            entity_embedding_matrix,
        )

        # Pepijn: Check above
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
    # or vert is an int, some 'vocabulary' index.
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
    # size of 200. Maybe the answer is in the input data loading.
    clicked_ctr = Input(shape=(max_clicked_news,), dtype="int32")
    print(clicked_input.shape)

    # Pepijn: This TimeDistributed stuff they use everywhere, its
    # just a fancy way to have a second 'batch' dimension. For instance
    # our news encoder expects a batch of news articles, but the user
    # encoder needs to encode the news of all the clicked articles of a user,
    # for every user in the batch. So in pytorch there is no tool to do this,
    # we would just merge the batch and the clicked articles dimension, do
    # the encoding, and then split them again. Nothing special.

    # Pepijn: the news embeddings for all clicked articles by a user. So in our
    # code n, the input to the NewsSelfAttention module for the user encoder.
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
        user_vec = AttentivePoolingQKY(50, 800, 400)([user_vec_query, user_vecs])

    # Pepijn: This is for w/o news popularity in figure 9.
    else:

        # Pepijn: Same as above, but without the popularity embeddings.
        user_vecs = Attention(20, 20)([user_vecs, user_vecs, user_vecs])
        user_vecs = Dropout(0.2)(user_vecs)
        user_vec = AttentivePooling(max_clicked_news, 400)(user_vecs)

    # Pepijn: This is were the user encoder stuff ends, and the popularity 
    # predictor begins. So our popularity_predictor.py.


    candidates = keras.Input(
        (
            1 + config["npratio"],
            news_input_length,
        ),
        dtype="int32",
    )
    candidates_ctr = keras.Input((1 + config["npratio"],), dtype="float32")
    candidates_rece_emb_index = keras.Input((1 + config["npratio"],), dtype="int32")

    if model_config["rece_emb"]:
        bias_content_vec = Input(shape=(500,))
        vec1 = keras.layers.Lambda(lambda x: x[:, :400])(bias_content_vec)
        vec2 = keras.layers.Lambda(lambda x: x[:, 400:])(bias_content_vec)

        vec1 = Dense(256, activation="tanh")(vec1)
        vec1 = Dense(256, activation="tanh")(vec1)
        vec1 = Dense(
            128,
        )(vec1)
        bias_content_score = Dense(1, use_bias=False)(vec1)

        vec2 = Dense(64, activation="tanh")(vec2)
        vec2 = Dense(64, activation="tanh")(vec2)
        bias_recency_score = Dense(1, use_bias=False)(vec2)

        gate = Dense(128, activation="tanh")(bias_content_vec)
        gate = Dense(64, activation="tanh")(gate)
        gate = Dense(1, activation="sigmoid")(gate)

        bias_content_score = keras.layers.Lambda(
            lambda x: (1 - x[0]) * x[1] + x[0] * x[2]
        )([gate, bias_content_score, bias_recency_score])

        bias_content_scorer = Model(bias_content_vec, bias_content_score)

    else:
        bias_content_vec = Input(shape=(400,))
        vec = Dense(256, activation="tanh")(bias_content_vec)
        vec = Dense(256, activation="tanh")(vec)
        vec = Dense(
            128,
        )(vec)
        bias_content_score = Dense(1, use_bias=False)(vec)
        bias_content_scorer = Model(bias_content_vec, bias_content_score)

    time_embedding_layer = Embedding(1500, 100, trainable=True)
    time_embedding = time_embedding_layer(candidates_rece_emb_index)

    candidate_vecs = TimeDistributed(news_encoder)(candidates)
    bias_candidate_vecs = TimeDistributed(bias_news_encoder)(candidates)
    if model_config["rece_emb"]:
        bias_candidate_vecs = keras.layers.Concatenate(axis=-1)(
            [bias_candidate_vecs, time_embedding]
        )
    bias_candidate_score = TimeDistributed(bias_content_scorer)(bias_candidate_vecs)
    bias_candidate_score = keras.layers.Reshape((1 + config["npratio"],))(
        bias_candidate_score
    )

    rel_scores = keras.layers.Dot(axes=-1)([user_vec, candidate_vecs])

    scaler = Dense(
        1, use_bias=False, kernel_initializer=keras.initializers.Constant(value=19)
    )
    ctrs = keras.layers.Reshape((1 + config["npratio"], 1))(candidates_ctr)
    ctrs = scaler(ctrs)
    bias_ctr_score = keras.layers.Reshape((1 + config["npratio"],))(ctrs)

    user_activity_input = keras.layers.Input((1,), dtype="int32")

    user_vec_input = keras.layers.Input(
        (400,),
    )
    activity_gate = Dense(128, activation="tanh")(user_vec_input)
    activity_gate = Dense(64, activation="tanh")(user_vec_input)
    activity_gate = Dense(1, activation="sigmoid")(activity_gate)
    activity_gate = keras.layers.Reshape((1,))(activity_gate)
    activity_gater = Model(user_vec_input, activity_gate)

    user_activtiy = activity_gater(user_vec)

    scores = []
    if model_config["rel"]:
        if model_config["activity"]:
            print(user_activtiy.shape)
            print(rel_scores.shape)
            rel_scores = keras.layers.Lambda(lambda x: 2 * x[0] * x[1])(
                [rel_scores, user_activtiy]
            )
            print(rel_scores.shape)

        scores.append(rel_scores)
    if model_config["content"]:
        if model_config["activity"]:
            bias_candidate_score = keras.layers.Lambda(lambda x: 2 * x[0] * (1 - x[1]))(
                [bias_candidate_score, user_activtiy]
            )
        scores.append(bias_candidate_score)
    if model_config["ctr"]:
        if model_config["activity"]:
            bias_ctr_score = keras.layers.Lambda(lambda x: 2 * x[0] * (1 - x[1]))(
                [bias_ctr_score, user_activtiy]
            )
        scores.append(bias_ctr_score)

    if len(scores) > 1:
        scores = keras.layers.Add()(scores)
    else:
        scores = scores[0]
    logits = keras.layers.Activation(keras.activations.softmax, name="recommend")(
        scores
    )

    model = Model(
        [
            candidates,
            candidates_ctr,
            candidates_rece_emb_index,
            user_activity_input,
            clicked_input,
            clicked_ctr,
        ],
        [logits],
    )

    model.compile(
        loss=["categorical_crossentropy"], optimizer=Adam(lr=0.0001), metrics=["acc"]
    )

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
