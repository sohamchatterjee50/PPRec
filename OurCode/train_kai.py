print("Adding path")
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
# import tensorflow as tf
import polars as pl
import math
#ADD THIS TO THE PATH
print("Adding path")
import os 
import os
cwd = os.getcwd()
print(cwd)
from testing_network import *
from ebrec.evaluation import MetricEvaluator, AucScore, NdcgScore, MrrScore
# sys.path.insert(0, '/home/apatra/Rec/contest/PPRec/OurCode/src/Ad_kai/ebnerd-benchmark/examples/00_quick_start')

# print("lets go")


from ebrec.utils._constants import (
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_IMPRESSION_ID_COL,
    DEFAULT_SUBTITLE_COL,
    DEFAULT_LABELS_COL,
    DEFAULT_TITLE_COL,
    DEFAULT_USER_COL,
)

from ebrec.utils._behaviors import (
    create_binary_labels_column,
    sampling_strategy_wu2019,
    add_known_user_column,
    add_prediction_scores,
    truncate_history,
)
from ebrec.evaluation import MetricEvaluator, AucScore, NdcgScore, MrrScore
from ebrec.utils._articles import convert_text2encoding_with_transformers
from ebrec.utils._polars import concat_str_columns, slice_join_dataframes
from ebrec.utils._articles import create_article_id_to_value_mapping
from ebrec.utils._nlp import get_transformers_word_embeddings
from ebrec.utils._python import write_submission_file, rank_predictions_by_score

from ebrec.models.newsrec.dataloader import NRMSDataLoader
# from ebrec.models.newsrec.model_config import hparams_nrms
# from ebrec.models.newsrec import NRMSModel

def ebnerd_from_path(path: Path, history_size: int = 30) -> pl.DataFrame:
    """
    Load ebnerd - function
    """
    df_history = (
        pl.scan_parquet(path.joinpath("history.parquet"))
        .select(DEFAULT_USER_COL, DEFAULT_HISTORY_ARTICLE_ID_COL)
        .pipe(
            truncate_history,
            column=DEFAULT_HISTORY_ARTICLE_ID_COL,
            history_size=history_size,
            padding_value=0,
            enable_warning=False,
        )
    )
    df_behaviors = (
        pl.scan_parquet(path.joinpath("behaviors.parquet"))
        .collect()
        .pipe(
            slice_join_dataframes,
            df2=df_history.collect(),
            on=DEFAULT_USER_COL,
            how="left",
        )
    )
    return df_behaviors


PATH = Path("~/ebnerd_data")
DATASPLIT = "ebnerd_small"
COLUMNS = [
    DEFAULT_USER_COL,
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_IMPRESSION_ID_COL,
]
HISTORY_SIZE = 10
FRACTION = 0.01

df_train = (
    ebnerd_from_path(PATH.joinpath(DATASPLIT, "train"), history_size=HISTORY_SIZE)
    .select(COLUMNS)
    .pipe(
        sampling_strategy_wu2019,
        npratio=4,
        shuffle=True,
        with_replacement=True,
        seed=123,
    )
    .pipe(create_binary_labels_column)
    .sample(fraction=FRACTION)
)
# =>
df_validation = (
    ebnerd_from_path(PATH.joinpath(DATASPLIT, "validation"), history_size=HISTORY_SIZE)
    .select(COLUMNS)
    .pipe(create_binary_labels_column)
    .sample(fraction=FRACTION)
)
df_articles = pl.read_parquet(PATH.joinpath("articles.parquet"))



df_articles=df_articles.with_columns((pl.col('total_pageviews') / pl.col('total_inviews') * 100).fill_null(0).alias('CTR'))

df_articles=df_articles.with_columns((pl.col('total_pageviews') / pl.col('total_inviews') * 100).fill_null(0).alias('Recency'))

# print(df_articles.select(['article_id', 'total_pageviews','total_inviews', 'CTR']))
# kai_behav=pl.read_parquet(PATH.joinpath("ebnerd_small/train/history.parquet"))
# print(kai_behav.head(2))
# kai_behav=ebnerd_from_path(PATH.joinpath(DATASPLIT, "train"), history_size=HISTORY_SIZE)
# # print(kai_behav.head(2))
# for col in df_articles.columns:
#     print(col)
# print(df_articles.head())


#MODEL KAI
TRANSFORMER_MODEL_NAME = "FacebookAI/xlm-roberta-base"
TEXT_COLUMNS_TO_USE = [DEFAULT_SUBTITLE_COL, DEFAULT_TITLE_COL]
MAX_TITLE_LENGTH = 30

# LOAD HUGGINGFACE:
transformer_model = AutoModel.from_pretrained(TRANSFORMER_MODEL_NAME)
transformer_tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)

# We'll init the word embeddings using the
word2vec_embedding = get_transformers_word_embeddings(transformer_model)
# print("help",word2vec_embedding)
#
df_articles, cat_cal = concat_str_columns(df_articles, columns=TEXT_COLUMNS_TO_USE)
# print("Hurricane",df_articles.head())
df_articles, token_col_title = convert_text2encoding_with_transformers(
    df_articles, transformer_tokenizer, cat_cal, max_length=MAX_TITLE_LENGTH
)
# =>
print("Hurricane2",token_col_title)

article_mapping = create_article_id_to_value_mapping(
    df=df_articles, value_col=token_col_title
)
article_id_to_ctr_mapping = dict(zip(df_articles['article_id'].to_list(), df_articles['CTR'].to_list()))
article_id_to_Recency_mapping = dict(zip(df_articles['article_id'].to_list(), df_articles['Recency'].to_list()))


for i in article_mapping.keys():
    data_list = article_mapping[i].to_list()  # Convert to list
    data_list.append(article_id_to_ctr_mapping[i])  # Append the number
    data_list.append(article_id_to_Recency_mapping[i])  # Append the number
    article_mapping[i] = pl.Series(data_list)



print("hurricane 3",word2vec_embedding.shape)



train_dataloader = NRMSDataLoader(
    behaviors=df_train,
    article_dict=article_mapping,
    unknown_representation="zeros",
    history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
    eval_mode=False,
    batch_size=64,
)
val_dataloader = NRMSDataLoader(
    behaviors=df_validation,
    article_dict=article_mapping,
    unknown_representation="zeros",
    history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
    eval_mode=True,
    batch_size=32,
)

config1=NewsEnc_config()
config2=Popularity_config()
config3=Userenc_config()
config4=General_Config()


model=PPREC(config4,config2,config3,config1,torch.tensor(word2vec_embedding)).cuda()







criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD([
                {'params': model.parameters(), 'lr': 1e-2},
            ], lr=1e-3, momentum=0.9)

for i in range(50):
    for x,y in train_dataloader:
        # print(x[0].shape,x[1].shape,y.shape)
        logits=model(torch.tensor(x[1]).long().cuda(),torch.tensor(x[0]).long().cuda())
        loss=criterion(logits,torch.argmax(torch.tensor(y),1).cuda())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        break

labels=[]
pred=[]
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Numberof Params",count_parameters(model))

model.PN_ratio=1
for x,y in val_dataloader:
    for z in range(math.ceil(x[0].shape[0]/73)):
        a0=x[0][z*73:(z+1)*73,:,:]
        a1=x[1][z*73:(z+1)*73,:,:]

        print(a0.shape,a1.shape)
        scores=model(torch.tensor(a1).long().cuda(),torch.tensor(a0).long().cuda())
        # scores=torch.zeros(x[0].shape[0],1)
        pred.append(scores.detach().cpu())
    # break

# labels=torch.cat(labels,0)
pred_validation=torch.cat(pred,0).squeeze()
print("hola",pred_validation.shape)

df_validation = add_prediction_scores(df_validation, pred_validation.tolist()).pipe(
    add_known_user_column, known_users=df_train[DEFAULT_USER_COL]
)
print(df_validation.head(2))

metrics = MetricEvaluator(
    labels=df_validation["labels"].to_list(),
    predictions=df_validation["scores"].to_list(),
    metric_functions=[AucScore(), MrrScore(), NdcgScore(k=5), NdcgScore(k=10)],
)
metrics.evaluate()

print(metrics)
