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
# with open('/home/scur1575/RecSys/PP-Rec/OurCode/demo_processed/article_mapping_topics_DEMO.pkl', 'rb') as handle:
#     article_mapping_topics = pickle.load(handle)
# with open('/home/scur1575/RecSys/PP-Rec/OurCode/demo_processed/article_mapping_category_str_DEMO.pkl', 'rb') as handle:
#     article_mapping_category_str = pickle.load(handle)

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


df_validation =  pl.scan_parquet("/home/scur1575/RecSys/PP-Rec/OurCode/demo_processed/val_ALL_COLS_DEMO.parquet").collect()

# subscription = True
# df_validation_sub =  df_validation.filter(pl.col("is_subscriber") == True)


val_dataloader = PPRecDataLoader(
    behaviors=df_validation,
    article_dict=article_mapping_title,
    entity_mapping=article_mapping_entity,
    ctr_mapping=articles_ctr,
    popularity_mapping = popularity_mapping,
    # topic_mapping = article_mapping_topics,
    # category_mapping = article_mapping_category_str,
    unknown_representation="zeros",
    history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
    history_recency = 'recency_hist',
    inview_recency = 'recency_inview',
    eval_mode=True,
    batch_size=1024,
)

print("Length:",len(val_dataloader))

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
            # vtopics = vinputs[-2]
            # vcategories = vinputs[-1]


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
            # vtopics = torch.from_numpy(vtopics)
            # vcategories = torch.from_numpy(vcategories)
            
            vtitle = vtitle.to(device)
            ventities = ventities.to(device)
            vctr = vctr.to(device)
            vrecency = vrecency.to(device)
            vhist_title = vhist_title.to(device)
            vhist_popularity = vhist_popularity.to(device)
            vlabels = vlabels.to(device)
            user_profile = user_profile.to(device)
            # vtopics = vtopics.to(device)
            # vcategories = vcategories.to(device)


            outputs = saved_model(vtitle, ventities, vctr, vrecency , vhist_title, vhist_popularity, user_profile).cpu().detach().numpy()
            predictions = np.concatenate([predictions,outputs],axis=0)
            
           
            

predictions = predictions[4:]
print("Predictions:",predictions.shape)


print("df_validation.shape:",df_validation.shape)

df_validation = df_validation.with_columns(pl.Series(name="predicted_scores", values=predictions)) 

metrics = MetricEvaluator(
    labels=df_validation["labels"].to_list(),
    predictions=df_validation["predicted_scores"].to_list(),
    metric_functions=[AucScore(), MrrScore(), NdcgScore(k=5), NdcgScore(k=10)],
)
print(metrics.evaluate())
# labels=df_validation["labels"].to_list()
# predictions=df_validation["predicted_scores"].to_list()
# print(len(labels))
# print(len(predictions))
# print("MSE:",(np.square(np.array(labels) -  np.array(predictions))).mean())



#subscription = False
# df_validation_sub =  df_validation.filter(pl.col("is_subscriber") == False)


# val_dataloader = PPRecDataLoader(
#     behaviors=df_validation_sub,
#     article_dict=article_mapping_title,
#     entity_mapping=article_mapping_entity,
#     ctr_mapping=articles_ctr,
#     popularity_mapping = popularity_mapping,
#     # topic_mapping = article_mapping_topics,
#     # category_mapping = article_mapping_category_str,
#     unknown_representation="zeros",
#     history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
#     history_recency = 'recency_hist',
#     inview_recency = 'recency_inview',
#     eval_mode=True,
#     batch_size=1024,
# )

# print("Length:",len(val_dataloader))

# saved_model.eval()

# predictions = np.empty(shape=(4,5))
# with torch.no_grad():
#     for i, vdata in enumerate(val_dataloader):
#             vinputs, vlabels = vdata
#             vtitle = vinputs[5]
#             ventities = vinputs[6]
#             vctr = vinputs[7]
#             vrecency = vinputs[8]
#             vhist_title = vinputs[0]
#             vhist_popularity = vinputs[2]
#             # vtopics = vinputs[-2]
#             # vcategories = vinputs[-1]


#             vtitle = torch.from_numpy(vtitle)
#             ventities = torch.from_numpy(ventities)
#             vctr = torch.from_numpy(vctr)
#             vrecency = torch.from_numpy(vrecency)
#             vhist_title = torch.from_numpy(vhist_title)
#             vhist_popularity = torch.from_numpy(vhist_popularity)
#             vlabels = torch.from_numpy(vlabels)
#             # vtopics = torch.from_numpy(vtopics)
#             # vcategories = torch.from_numpy(vcategories)
            
#             vtitle = vtitle.to(device)
#             ventities = ventities.to(device)
#             vctr = vctr.to(device)
#             vrecency = vrecency.to(device)
#             vhist_title = vhist_title.to(device)
#             vhist_popularity = vhist_popularity.to(device)
#             vlabels = vlabels.to(device)
#             # vtopics = vtopics.to(device)
#             # vcategories = vcategories.to(device)


#             outputs = saved_model(vtitle, ventities, vctr, vrecency , vhist_title, vhist_popularity).cpu().detach().numpy()
#             predictions = np.concatenate([predictions,outputs],axis=0)
            
           
            

# predictions = predictions[4:]
# print("Predictions:",predictions.shape)


# print("df_validation.shape:",df_validation.shape)

# df_validation_sub = df_validation_sub.with_columns(pl.Series(name="predicted_scores", values=predictions)) 

# metrics = MetricEvaluator(
#     labels=df_validation_sub["labels"].to_list(),
#     predictions=df_validation_sub["predicted_scores"].to_list(),
#     metric_functions=[AucScore(), MrrScore(), NdcgScore(k=5), NdcgScore(k=10)],
# )
# print(metrics.evaluate())



# # Age into 3 brackets
# df_validation_sub =  df_validation.filter(pl.col("age") <=30)


# val_dataloader = PPRecDataLoader(
#     behaviors=df_validation_sub,
#     article_dict=article_mapping_title,
#     entity_mapping=article_mapping_entity,
#     ctr_mapping=articles_ctr,
#     popularity_mapping = popularity_mapping,
#     # topic_mapping = article_mapping_topics,
#     # category_mapping = article_mapping_category_str,
#     unknown_representation="zeros",
#     history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
#     history_recency = 'recency_hist',
#     inview_recency = 'recency_inview',
#     eval_mode=True,
#     batch_size=1024,
# )

# print("Length:",len(val_dataloader))

# saved_model.eval()

# predictions = np.empty(shape=(4,5))
# with torch.no_grad():
#     for i, vdata in enumerate(val_dataloader):
#             vinputs, vlabels = vdata
#             vtitle = vinputs[5]
#             ventities = vinputs[6]
#             vctr = vinputs[7]
#             vrecency = vinputs[8]
#             vhist_title = vinputs[0]
#             vhist_popularity = vinputs[2]
#             # vtopics = vinputs[-2]
#             # vcategories = vinputs[-1]


#             vtitle = torch.from_numpy(vtitle)
#             ventities = torch.from_numpy(ventities)
#             vctr = torch.from_numpy(vctr)
#             vrecency = torch.from_numpy(vrecency)
#             vhist_title = torch.from_numpy(vhist_title)
#             vhist_popularity = torch.from_numpy(vhist_popularity)
#             vlabels = torch.from_numpy(vlabels)
#             # vtopics = torch.from_numpy(vtopics)
#             # vcategories = torch.from_numpy(vcategories)
            
#             vtitle = vtitle.to(device)
#             ventities = ventities.to(device)
#             vctr = vctr.to(device)
#             vrecency = vrecency.to(device)
#             vhist_title = vhist_title.to(device)
#             vhist_popularity = vhist_popularity.to(device)
#             vlabels = vlabels.to(device)
#             # vtopics = vtopics.to(device)
#             # vcategories = vcategories.to(device)


#             outputs = saved_model(vtitle, ventities, vctr, vrecency , vhist_title, vhist_popularity).cpu().detach().numpy()
#             predictions = np.concatenate([predictions,outputs],axis=0)
            
           
            

# predictions = predictions[4:]
# print("Predictions:",predictions.shape)


# print("df_validation.shape:",df_validation.shape)

# df_validation_sub = df_validation_sub.with_columns(pl.Series(name="predicted_scores", values=predictions)) 

# metrics = MetricEvaluator(
#     labels=df_validation_sub["labels"].to_list(),
#     predictions=df_validation_sub["predicted_scores"].to_list(),
#     metric_functions=[AucScore(), MrrScore(), NdcgScore(k=5), NdcgScore(k=10)],
# )
# print(metrics.evaluate())


# # Age into 3 brackets
# df_validation_sub =  df_validation.filter(pl.col("age") >50)


# val_dataloader = PPRecDataLoader(
#     behaviors=df_validation_sub,
#     article_dict=article_mapping_title,
#     entity_mapping=article_mapping_entity,
#     ctr_mapping=articles_ctr,
#     popularity_mapping = popularity_mapping,
#     # topic_mapping = article_mapping_topics,
#     # category_mapping = article_mapping_category_str,
#     unknown_representation="zeros",
#     history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
#     history_recency = 'recency_hist',
#     inview_recency = 'recency_inview',
#     eval_mode=True,
#     batch_size=1024,
# )

# print("Length:",len(val_dataloader))

# saved_model.eval()

# predictions = np.empty(shape=(4,5))
# with torch.no_grad():
#     for i, vdata in enumerate(val_dataloader):
#             vinputs, vlabels = vdata
#             vtitle = vinputs[5]
#             ventities = vinputs[6]
#             vctr = vinputs[7]
#             vrecency = vinputs[8]
#             vhist_title = vinputs[0]
#             vhist_popularity = vinputs[2]
#             # vtopics = vinputs[-2]
#             # vcategories = vinputs[-1]


#             vtitle = torch.from_numpy(vtitle)
#             ventities = torch.from_numpy(ventities)
#             vctr = torch.from_numpy(vctr)
#             vrecency = torch.from_numpy(vrecency)
#             vhist_title = torch.from_numpy(vhist_title)
#             vhist_popularity = torch.from_numpy(vhist_popularity)
#             vlabels = torch.from_numpy(vlabels)
#             # vtopics = torch.from_numpy(vtopics)
#             # vcategories = torch.from_numpy(vcategories)
            
#             vtitle = vtitle.to(device)
#             ventities = ventities.to(device)
#             vctr = vctr.to(device)
#             vrecency = vrecency.to(device)
#             vhist_title = vhist_title.to(device)
#             vhist_popularity = vhist_popularity.to(device)
#             vlabels = vlabels.to(device)
#             # vtopics = vtopics.to(device)
#             # vcategories = vcategories.to(device)


#             outputs = saved_model(vtitle, ventities, vctr, vrecency , vhist_title, vhist_popularity).cpu().detach().numpy()
#             predictions = np.concatenate([predictions,outputs],axis=0)
            
           
            

# predictions = predictions[4:]
# print("Predictions:",predictions.shape)


# print("df_validation.shape:",df_validation.shape)

# df_validation_sub = df_validation_sub.with_columns(pl.Series(name="predicted_scores", values=predictions)) 

# metrics = MetricEvaluator(
#     labels=df_validation_sub["labels"].to_list(),
#     predictions=df_validation_sub["predicted_scores"].to_list(),
#     metric_functions=[AucScore(), MrrScore(), NdcgScore(k=5), NdcgScore(k=10)],
# )
# print(metrics.evaluate())


# # Age into 3 brackets
# df_validation_sub =  df_validation.filter(pl.col("age").is_between(30,50))


# val_dataloader = PPRecDataLoader(
#     behaviors=df_validation_sub,
#     article_dict=article_mapping_title,
#     entity_mapping=article_mapping_entity,
#     ctr_mapping=articles_ctr,
#     popularity_mapping = popularity_mapping,
#     # topic_mapping = article_mapping_topics,
#     # category_mapping = article_mapping_category_str,
#     unknown_representation="zeros",
#     history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
#     history_recency = 'recency_hist',
#     inview_recency = 'recency_inview',
#     eval_mode=True,
#     batch_size=1024,
# )

# print("Length:",len(val_dataloader))

# saved_model.eval()

# predictions = np.empty(shape=(4,5))
# with torch.no_grad():
#     for i, vdata in enumerate(val_dataloader):
#             vinputs, vlabels = vdata
#             vtitle = vinputs[5]
#             ventities = vinputs[6]
#             vctr = vinputs[7]
#             vrecency = vinputs[8]
#             vhist_title = vinputs[0]
#             vhist_popularity = vinputs[2]
#             # vtopics = vinputs[-2]
#             # vcategories = vinputs[-1]


#             vtitle = torch.from_numpy(vtitle)
#             ventities = torch.from_numpy(ventities)
#             vctr = torch.from_numpy(vctr)
#             vrecency = torch.from_numpy(vrecency)
#             vhist_title = torch.from_numpy(vhist_title)
#             vhist_popularity = torch.from_numpy(vhist_popularity)
#             vlabels = torch.from_numpy(vlabels)
#             # vtopics = torch.from_numpy(vtopics)
#             # vcategories = torch.from_numpy(vcategories)
            
#             vtitle = vtitle.to(device)
#             ventities = ventities.to(device)
#             vctr = vctr.to(device)
#             vrecency = vrecency.to(device)
#             vhist_title = vhist_title.to(device)
#             vhist_popularity = vhist_popularity.to(device)
#             vlabels = vlabels.to(device)
#             # vtopics = vtopics.to(device)
#             # vcategories = vcategories.to(device)


#             outputs = saved_model(vtitle, ventities, vctr, vrecency , vhist_title, vhist_popularity).cpu().detach().numpy()
#             predictions = np.concatenate([predictions,outputs],axis=0)
            
           
            

# predictions = predictions[4:]
# print("Predictions:",predictions.shape)


# print("df_validation.shape:",df_validation.shape)

# df_validation_sub = df_validation_sub.with_columns(pl.Series(name="predicted_scores", values=predictions)) 

# metrics = MetricEvaluator(
#     labels=df_validation_sub["labels"].to_list(),
#     predictions=df_validation_sub["predicted_scores"].to_list(),
#     metric_functions=[AucScore(), MrrScore(), NdcgScore(k=5), NdcgScore(k=10)],
# )
# print(metrics.evaluate())




# Gender = 0
# df_validation_sub =  df_validation.filter(pl.col("gender") == 0)


# val_dataloader = PPRecDataLoader(
#     behaviors=df_validation_sub,
#     article_dict=article_mapping_title,
#     entity_mapping=article_mapping_entity,
#     ctr_mapping=articles_ctr,
#     popularity_mapping = popularity_mapping,
#     # topic_mapping = article_mapping_topics,
#     # category_mapping = article_mapping_category_str,
#     unknown_representation="zeros",
#     history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
#     history_recency = 'recency_hist',
#     inview_recency = 'recency_inview',
#     eval_mode=True,
#     batch_size=1024,
# )

# print("Length:",len(val_dataloader))

# saved_model.eval()

# predictions = np.empty(shape=(4,5))
# with torch.no_grad():
#     for i, vdata in enumerate(val_dataloader):
#             vinputs, vlabels = vdata
#             vtitle = vinputs[5]
#             ventities = vinputs[6]
#             vctr = vinputs[7]
#             vrecency = vinputs[8]
#             vhist_title = vinputs[0]
#             vhist_popularity = vinputs[2]
#             # vtopics = vinputs[-2]
#             # vcategories = vinputs[-1]


#             vtitle = torch.from_numpy(vtitle)
#             ventities = torch.from_numpy(ventities)
#             vctr = torch.from_numpy(vctr)
#             vrecency = torch.from_numpy(vrecency)
#             vhist_title = torch.from_numpy(vhist_title)
#             vhist_popularity = torch.from_numpy(vhist_popularity)
#             vlabels = torch.from_numpy(vlabels)
#             # vtopics = torch.from_numpy(vtopics)
#             # vcategories = torch.from_numpy(vcategories)
            
#             vtitle = vtitle.to(device)
#             ventities = ventities.to(device)
#             vctr = vctr.to(device)
#             vrecency = vrecency.to(device)
#             vhist_title = vhist_title.to(device)
#             vhist_popularity = vhist_popularity.to(device)
#             vlabels = vlabels.to(device)
#             # vtopics = vtopics.to(device)
#             # vcategories = vcategories.to(device)


#             outputs = saved_model(vtitle, ventities, vctr, vrecency , vhist_title, vhist_popularity).cpu().detach().numpy()
#             predictions = np.concatenate([predictions,outputs],axis=0)
            
           
            

# predictions = predictions[4:]
# print("Predictions:",predictions.shape)


# print("df_validation.shape:",df_validation.shape)

# df_validation_sub = df_validation_sub.with_columns(pl.Series(name="predicted_scores", values=predictions)) 

# metrics = MetricEvaluator(
#     labels=df_validation_sub["labels"].to_list(),
#     predictions=df_validation_sub["predicted_scores"].to_list(),
#     metric_functions=[AucScore(), MrrScore(), NdcgScore(k=5), NdcgScore(k=10)],
# )
# print(metrics.evaluate())

# # Gender = 1
# df_validation_sub =  df_validation.filter(pl.col("gender") == 1)


# val_dataloader = PPRecDataLoader(
#     behaviors=df_validation_sub,
#     article_dict=article_mapping_title,
#     entity_mapping=article_mapping_entity,
#     ctr_mapping=articles_ctr,
#     popularity_mapping = popularity_mapping,
#     # topic_mapping = article_mapping_topics,
#     # category_mapping = article_mapping_category_str,
#     unknown_representation="zeros",
#     history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
#     history_recency = 'recency_hist',
#     inview_recency = 'recency_inview',
#     eval_mode=True,
#     batch_size=1024,
# )

# print("Length:",len(val_dataloader))

# saved_model.eval()

# predictions = np.empty(shape=(4,5))
# with torch.no_grad():
#     for i, vdata in enumerate(val_dataloader):
#             vinputs, vlabels = vdata
#             vtitle = vinputs[5]
#             ventities = vinputs[6]
#             vctr = vinputs[7]
#             vrecency = vinputs[8]
#             vhist_title = vinputs[0]
#             vhist_popularity = vinputs[2]
#             # vtopics = vinputs[-2]
#             # vcategories = vinputs[-1]


#             vtitle = torch.from_numpy(vtitle)
#             ventities = torch.from_numpy(ventities)
#             vctr = torch.from_numpy(vctr)
#             vrecency = torch.from_numpy(vrecency)
#             vhist_title = torch.from_numpy(vhist_title)
#             vhist_popularity = torch.from_numpy(vhist_popularity)
#             vlabels = torch.from_numpy(vlabels)
#             # vtopics = torch.from_numpy(vtopics)
#             # vcategories = torch.from_numpy(vcategories)
            
#             vtitle = vtitle.to(device)
#             ventities = ventities.to(device)
#             vctr = vctr.to(device)
#             vrecency = vrecency.to(device)
#             vhist_title = vhist_title.to(device)
#             vhist_popularity = vhist_popularity.to(device)
#             vlabels = vlabels.to(device)
#             # vtopics = vtopics.to(device)
#             # vcategories = vcategories.to(device)


#             outputs = saved_model(vtitle, ventities, vctr, vrecency , vhist_title, vhist_popularity).cpu().detach().numpy()
#             predictions = np.concatenate([predictions,outputs],axis=0)
            
           
            

# predictions = predictions[4:]
# print("Predictions:",predictions.shape)


# print("df_validation.shape:",df_validation.shape)

# df_validation_sub = df_validation_sub.with_columns(pl.Series(name="predicted_scores", values=predictions)) 

# metrics = MetricEvaluator(
#     labels=df_validation_sub["labels"].to_list(),
#     predictions=df_validation_sub["predicted_scores"].to_list(),
#     metric_functions=[AucScore(), MrrScore(), NdcgScore(k=5), NdcgScore(k=10)],
# )
# print(metrics.evaluate())

# On small VAL

# article_mapping_title, article_mapping_entity, articles_ctr, popularity_mapping = {},{},{},{}
# with open('/home/scur1575/RecSys/PP-Rec/OurCode/small_processed/article_mapping_title_SMALL.pkl', 'rb') as handle:
#     article_mapping_title = pickle.load(handle)
# with open('/home/scur1575/RecSys/PP-Rec/OurCode/small_processed/article_mapping_entity_SMALL.pkl', 'rb') as handle:
#     article_mapping_entity = pickle.load(handle)
# with open('/home/scur1575/RecSys/PP-Rec/OurCode/small_processed/articles_ctr_SMALL.pkl', 'rb') as handle:
#     articles_ctr = pickle.load(handle)
# with open('/home/scur1575/RecSys/PP-Rec/OurCode/small_processed/popularity_mapping_SMALL.pkl', 'rb') as handle:
#     popularity_mapping = pickle.load(handle)

# COLUMNS = [
#    'user_id',
#    'article_id_fixed',
#    'article_ids_inview',
#    'article_ids_clicked',
#    'impression_id',
#    'labels',
#    'recency_inview',
#    'recency_hist'  
# ]


# df_validation =  pl.scan_parquet("/home/scur1575/RecSys/PP-Rec/OurCode/small_processed/val_SMALL.parquet").select(COLUMNS).collect()

# val_dataloader = PPRecDataLoader(
#     behaviors=df_validation,
#     article_dict=article_mapping_title,
#     entity_mapping=article_mapping_entity,
#     ctr_mapping=articles_ctr,
#     popularity_mapping = popularity_mapping,
#     unknown_representation="zeros",
#     history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
#     history_recency = 'recency_hist',
#     inview_recency = 'recency_inview',
#     eval_mode=True,
#     batch_size=512,
# )

# print("Length:",len(val_dataloader))

# saved_model.eval()

# predictions = np.empty(shape=(4,5))
# with torch.no_grad():
#     for i, vdata in enumerate(val_dataloader):
#             vinputs, vlabels = vdata
#             vtitle = vinputs[5]
#             ventities = vinputs[6]
#             vctr = vinputs[7]
#             vrecency = vinputs[8]
#             vhist_title = vinputs[0]
#             vhist_popularity = vinputs[2]

#             vtitle = torch.from_numpy(vtitle)
#             ventities = torch.from_numpy(ventities)
#             vctr = torch.from_numpy(vctr)
#             vrecency = torch.from_numpy(vrecency)
#             vhist_title = torch.from_numpy(vhist_title)
#             vhist_popularity = torch.from_numpy(vhist_popularity)
#             vlabels = torch.from_numpy(vlabels)
            
#             vtitle = vtitle.to(device)
#             ventities = ventities.to(device)
#             vctr = vctr.to(device)
#             vrecency = vrecency.to(device)
#             vhist_title = vhist_title.to(device)
#             vhist_popularity = vhist_popularity.to(device)
#             vlabels = vlabels.to(device)


#             outputs = saved_model(vtitle, ventities, vctr, vrecency , vhist_title, vhist_popularity).cpu().detach().numpy()
#             predictions = np.concatenate([predictions,outputs],axis=0)
            
           
            

# predictions = predictions[4:]
# print("Predictions:",predictions.shape)


# print("df_validation.shape:",df_validation.shape)

# df_validation = df_validation.with_columns(pl.Series(name="predicted_scores", values=predictions)) 

# metrics = MetricEvaluator(
#     labels=df_validation["labels"].to_list(),
#     predictions=df_validation["predicted_scores"].to_list(),
#     metric_functions=[AucScore(), MrrScore(), NdcgScore(k=5), NdcgScore(k=10)],
# )
# print("DEMO checkpoint tested on SMALL VAL")
# print(metrics.evaluate())


# # On LARGE VAL

# article_mapping_title, article_mapping_entity, articles_ctr, popularity_mapping = {},{},{},{}
# with open('/home/scur1575/RecSys/PP-Rec/OurCode/large_processed/article_mapping_title_LARGE.pkl', 'rb') as handle:
#     article_mapping_title = pickle.load(handle)
# with open('/home/scur1575/RecSys/PP-Rec/OurCode/large_processed/article_mapping_entity_LARGE.pkl', 'rb') as handle:
#     article_mapping_entity = pickle.load(handle)
# with open('/home/scur1575/RecSys/PP-Rec/OurCode/large_processed/articles_ctr_LARGE.pkl', 'rb') as handle:
#     articles_ctr = pickle.load(handle)
# with open('/home/scur1575/RecSys/PP-Rec/OurCode/large_processed/popularity_mapping_LARGE.pkl', 'rb') as handle:
#     popularity_mapping = pickle.load(handle)

# COLUMNS = [
#    'user_id',
#    'article_id_fixed',
#    'article_ids_inview',
#    'article_ids_clicked',
#    'impression_id',
#    'labels',
#    'recency_inview',
#    'recency_hist'  
# ]


# df_validation =  pl.scan_parquet("/home/scur1575/RecSys/PP-Rec/OurCode/large_processed/val_LARGE_TEST.parquet").select(COLUMNS).collect()

# val_dataloader = PPRecDataLoader(
#     behaviors=df_validation,
#     article_dict=article_mapping_title,
#     entity_mapping=article_mapping_entity,
#     ctr_mapping=articles_ctr,
#     popularity_mapping = popularity_mapping,
#     unknown_representation="zeros",
#     history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
#     history_recency = 'recency_hist',
#     inview_recency = 'recency_inview',
#     eval_mode=True,
#     batch_size=1024,
# )

# print("Length:",len(val_dataloader))

# saved_model.eval()

# predictions = np.empty(shape=(4,5))
# with torch.no_grad():
#     for i, vdata in enumerate(val_dataloader):
#             vinputs, vlabels = vdata
#             vtitle = vinputs[5]
#             ventities = vinputs[6]
#             vctr = vinputs[7]
#             vrecency = vinputs[8]
#             vhist_title = vinputs[0]
#             vhist_popularity = vinputs[2]

#             vtitle = torch.from_numpy(vtitle)
#             ventities = torch.from_numpy(ventities)
#             vctr = torch.from_numpy(vctr)
#             vrecency = torch.from_numpy(vrecency)
#             vhist_title = torch.from_numpy(vhist_title)
#             vhist_popularity = torch.from_numpy(vhist_popularity)
#             vlabels = torch.from_numpy(vlabels)
            
#             vtitle = vtitle.to(device)
#             ventities = ventities.to(device)
#             vctr = vctr.to(device)
#             vrecency = vrecency.to(device)
#             vhist_title = vhist_title.to(device)
#             vhist_popularity = vhist_popularity.to(device)
#             vlabels = vlabels.to(device)


#             outputs = saved_model(vtitle, ventities, vctr, vrecency , vhist_title, vhist_popularity).cpu().detach().numpy()
#             predictions = np.concatenate([predictions,outputs],axis=0)
            
           
            

# predictions = predictions[4:]
# print("Predictions:",predictions.shape)


# print("df_validation.shape:",df_validation.shape)

# df_validation = df_validation.with_columns(pl.Series(name="predicted_scores", values=predictions)) 

# metrics = MetricEvaluator(
#     labels=df_validation["labels"].to_list(),
#     predictions=df_validation["predicted_scores"].to_list(),
#     metric_functions=[AucScore(), MrrScore(), NdcgScore(k=5), NdcgScore(k=10)],
# )
# print("DEMO checkpoint tested on LARGE VAL")
# print(metrics.evaluate())

