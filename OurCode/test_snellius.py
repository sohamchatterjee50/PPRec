from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import torch 
import polars as pl

from ebrec.utils._constants import (
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_IMPRESSION_ID_COL,
    DEFAULT_SUBTITLE_COL,
    DEFAULT_LABELS_COL,
    DEFAULT_TITLE_COL,
    DEFAULT_USER_COL,
    DEFAULT_ARTICLE_MODIFIED_TIMESTAMP_COL,
    DEFAULT_IMPRESSION_TIMESTAMP_COL,
    DEFAULT_HISTORY_IMPRESSION_TIMESTAMP_COL
)

from ebrec.utils._behaviors import (
    create_binary_labels_column,
    sampling_strategy_wu2019,
    add_known_user_column,
    add_prediction_scores,
    truncate_history,
)
from ebrec.evaluation import MetricEvaluator, AucScore, NdcgScore, MrrScore
from ebrec.utils._articles import convert_text2encoding_with_transformers,concat_list_to_text
from ebrec.utils._polars import concat_str_columns, slice_join_dataframes
from ebrec.utils._articles import create_article_id_to_value_mapping
from ebrec.utils._nlp import get_transformers_word_embeddings
from ebrec.utils._python import write_submission_file, rank_predictions_by_score

# %load_ext autoreload
# %autoreload 2

from src.model.dataloader import PPRecDataLoader
from src.model.model_config import hparams_pprec
from src.model.pprec import PPRec
PATH = Path("/Users/sohamchatterjee/Documents/UvA/RecSYS/Project/ebnerd_data")
DATASPLIT = "ebnerd_demo"

import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


df_articles = pl.read_parquet(PATH.joinpath("articles.parquet"))
TRANSFORMER_MODEL_NAME = "FacebookAI/xlm-roberta-base"
TEXT_COLUMNS_TO_USE = [DEFAULT_SUBTITLE_COL, DEFAULT_TITLE_COL]
MAX_TITLE_LENGTH = 30

# LOAD HUGGINGFACE:
transformer_model = AutoModel.from_pretrained(TRANSFORMER_MODEL_NAME)
transformer_tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)

# We'll init the word embeddings using the
word2vec_embedding = get_transformers_word_embeddings(transformer_model)
#
df_articles, cat_cal = concat_str_columns(df_articles, columns=TEXT_COLUMNS_TO_USE)
df_articles, token_col_title = convert_text2encoding_with_transformers(
    df_articles, transformer_tokenizer, cat_cal, max_length=MAX_TITLE_LENGTH
)
# =>
article_mapping_title = create_article_id_to_value_mapping(
    df=df_articles, value_col=token_col_title
)


import pickle
article_mapping_title, article_mapping_entity, articles_ctr, popularity_mapping = {},{},{},{}
with open('article_mapping_title.pkl', 'rb') as handle:
    article_mapping_title = pickle.load(handle)
with open('article_mapping_entity.pkl', 'rb') as handle:
    article_mapping_entity = pickle.load(handle)
with open('articles_ctr.pkl', 'rb') as handle:
    articles_ctr = pickle.load(handle)
with open('popularity_mapping.pkl', 'rb') as handle:
    popularity_mapping = pickle.load(handle)

COLUMNS = [
   'user_id',
   'article_id_fixed',
   'article_ids_inview',
   'article_ids_clicked',
   'impression_id',
   'labels',
   'recency_inview',
   'recency_hist'  
]
df_train  = pl.scan_parquet("small_demo_train_all_features.parquet").select(COLUMNS).collect()

df_validation =  pl.scan_parquet("small_demo_val_all_features_with_sampling.parquet").select(COLUMNS).collect()



train_dataloader = PPRecDataLoader(
    behaviors=df_train,
    article_dict=article_mapping_title,
    entity_mapping=article_mapping_entity,
    ctr_mapping=articles_ctr,
    popularity_mapping = popularity_mapping,
    unknown_representation="zeros",
    history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
    history_recency = 'recency_hist',
    inview_recency = 'recency_inview',
    eval_mode=False,
    batch_size=4,
)


val_dataloader = PPRecDataLoader(
    behaviors=df_validation,
    article_dict=article_mapping_title,
    entity_mapping=article_mapping_entity,
    ctr_mapping=articles_ctr,
    popularity_mapping = popularity_mapping,
    unknown_representation="zeros",
    history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
    history_recency = 'recency_hist',
    inview_recency = 'recency_inview',
    eval_mode=True,
    batch_size=4,
)


from model.modules import PPRec
from model.modules import train_one_epoch
from datetime import datetime
from model.modules import BPELoss
from torch.utils.tensorboard import SummaryWriter
# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/pprec_check{}'.format(timestamp))
epoch_number = 0

EPOCHS = 5
loss_fn = BPELoss()
model = PPRec(hparams_pprec,word2vec_embedding)
model.to(device)
optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=0.01,
            weight_decay=1e-4
        )

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer,train_dataloader, optimizer,model,loss_fn,device)


    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(val_dataloader):
            vinputs, vlabels = vdata

            vtitle = vinputs[5]
            ventities = vinputs[6]
            vctr = vinputs[7]
            vrecency = vinputs[8]
            vhist_title = vinputs[0]
            vhist_popularity = vinputs[2]

            vtitle = torch.from_numpy(vtitle)
            ventities = torch.from_numpy(ventities)
            vctr = torch.from_numpy(vctr)
            vrecency = torch.from_numpy(vrecency)
            vhist_title = torch.from_numpy(vhist_title)
            vhist_popularity = torch.from_numpy(vhist_popularity)
            vlabels = torch.from_numpy(vlabels)
        
            vtitle = vtitle.to(device)
            ventities = ventities.to(device)
            vctr = vctr.to(device)
            vrecency = vrecency.to(device)
            vhist_title = vhist_title.to(device)
            vhist_popularity = vhist_popularity.to(device)
            vlabels = vlabels.to(device)


            voutputs = model(vtitle, ventities, vctr, vrecency , vhist_title, vhist_popularity )
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1