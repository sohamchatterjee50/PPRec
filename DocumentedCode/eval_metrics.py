import torch
from src.model.dataloader import PPRecDataLoader
from src.model.model_config import hparams_pprec
from model.modules import PPRec
from transformers import AutoTokenizer, AutoModel
from ebrec.evaluation import MetricEvaluator, AucScore, NdcgScore, MrrScore
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
from ebrec.utils._nlp import get_transformers_word_embeddings
import numpy as np


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
TRANSFORMER_MODEL_NAME = "FacebookAI/xlm-roberta-base"

transformer_model = AutoModel.from_pretrained(TRANSFORMER_MODEL_NAME)
transformer_tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)

word2vec_embedding = get_transformers_word_embeddings(transformer_model)

saved_model = PPRec(hparams_pprec,word2vec_embedding)

PATH = '/home/scur1575/RecSys/PP-Rec/OurCode/Checkpoints/User_Profiling/model_DEMO_User_Profiling_Log_Tensorboard__20240627_033837_0'
saved_model.load_state_dict(torch.load(PATH,weights_only=True))
saved_model.to(device)
import pickle
article_mapping_title, article_mapping_entity, articles_ctr, popularity_mapping = {},{},{},{}
with open('/home/scur1575/RecSys/PP-Rec/OurCode/demo_processed/article_mapping_title_DEMO.pkl', 'rb') as handle:
    article_mapping_title = pickle.load(handle)
with open('/home/scur1575/RecSys/PP-Rec/OurCode/demo_processed/article_mapping_entity_DEMO.pkl', 'rb') as handle:
    article_mapping_entity = pickle.load(handle)
with open('/home/scur1575/RecSys/PP-Rec/OurCode/demo_processed/articles_ctr_DEMO.pkl', 'rb') as handle:
    articles_ctr = pickle.load(handle)
with open('/home/scur1575/RecSys/PP-Rec/OurCode/demo_processed/popularity_mapping_DEMO.pkl', 'rb') as handle:
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


df_validation =  pl.scan_parquet("/home/scur1575/RecSys/PP-Rec/OurCode/demo_processed/val_DEMO.parquet").collect()



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
    batch_size=1024,
)


saved_model.eval()

predictions = np.empty(shape=(4,5))
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
            user_profile = torch.cat(
            (
            torch.from_numpy(vinputs[-4]),
            torch.from_numpy(vinputs[-3]),
            torch.from_numpy(vinputs[-2]),
            torch.from_numpy(vinputs[-1])),axis=1)
            
            
            vtitle = vtitle.to(device)
            ventities = ventities.to(device)
            vctr = vctr.to(device)
            vrecency = vrecency.to(device)
            vhist_title = vhist_title.to(device)
            vhist_popularity = vhist_popularity.to(device)
            vlabels = vlabels.to(device)
            user_profile = user_profile.to(device)
            

            outputs = saved_model(vtitle, ventities, vctr, vrecency , vhist_title, vhist_popularity, user_profile).cpu().detach().numpy()
            predictions = np.concatenate([predictions,outputs],axis=0)
            
           
            

predictions = predictions[4:]





df_validation = df_validation.with_columns(pl.Series(name="predicted_scores", values=predictions)) 

metrics = MetricEvaluator(
    labels=df_validation["labels"].to_list(),
    predictions=df_validation["predicted_scores"].to_list(),
    metric_functions=[AucScore(), MrrScore(), NdcgScore(k=5), NdcgScore(k=10)],
)
print(metrics.evaluate())
