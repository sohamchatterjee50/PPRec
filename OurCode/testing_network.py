import torch
import torch.nn as nn
import math 
import torch.nn.functional as F

from ebrec.utils._behaviors import (
    create_binary_labels_column,
    sampling_strategy_wu2019,
    add_known_user_column,
    add_prediction_scores,
    truncate_history,
)

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention





def expand_mask(mask):
    assert mask.ndim >= 2, "Mask must be at least 2-dimensional with seq_length x seq_length"
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    while mask.ndim < 4:
        mask = mask.unsqueeze(0)
    return mask


class Attention(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, _ = x.size()
        if mask is not None:
            mask = expand_mask(mask)
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o

# def test():
#     v=Attention(30,60,5)
#     d=torch.zeros(42,20,30)
#     print(v(d).shape)


# test()


class AttentivePooling(nn.Module):
    def __init__(self,  dim2):
        super(AttentivePooling, self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.att_dense = nn.Linear(dim2, 200)
        self.tanh = nn.Tanh()
        self.dense = nn.Linear(200, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.dropout(x)  # (batch_size, 50, 400)
        att = self.att_dense(x)  # (batch_size, 50, 200)
        att = self.tanh(att)  # (batch_size, 50, 200)
        att = self.dense(att).squeeze(-1)  # (batch_size, 50)
        att = self.softmax(att)  # (batch_size, 50)
        att = att.unsqueeze(2)  # (batch_size, 50, 1)
        user_vec = torch.bmm(x.transpose(1, 2), att).squeeze(2)  # (batch_size, 400)
        return user_vec

# # Example usage
# dim1, dim2 = 50, 400
# model = AttentivePooling( dim2)
# input_data = torch.randn(32, dim1, dim2)  # batch_size is 32 for example
# output_data = model(input_data)
# print(output_data.shape)  # should be (32, 400)

class AttentivePoolingQKY(nn.Module):
    def __init__(self, dim1, dim2, dim3):
        super(AttentivePoolingQKY, self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.att_dense = nn.Linear(dim2, 200)
        self.tanh = nn.Tanh()
        self.dense = nn.Linear(200, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, vecs_input, value_input):
        vecs_input = self.dropout(vecs_input)  # (batch_size, dim1, dim2)
        user_att = self.att_dense(vecs_input)  # (batch_size, dim1, 200)
        user_att = self.tanh(user_att)  # (batch_size, dim1, 200)
        user_att = self.dense(user_att).squeeze(-1)  # (batch_size, dim1)
        user_att = self.softmax(user_att)  # (batch_size, dim1)
        user_att = user_att.unsqueeze(2)  # (batch_size, dim1, 1)
        user_vec = torch.bmm(value_input.transpose(1, 2), user_att).squeeze(2)  # (batch_size, dim3)
        return user_vec

# Example usage
# dim1, dim2, dim3 = 50, 400, 300
# model = AttentivePoolingQKY(dim1, dim2, dim3)
# vecs_input = torch.randn(32, dim1, dim2)  # batch_size is 32 for example
# value_input = torch.randn(32, dim1, dim3)  # batch_size is 32 for example
# output_data = model(vecs_input, value_input)
# print(output_data.shape)  # should be (32, dim3)


class NewsEncoder(nn.Module):
    def __init__(self, config):
        super(NewsEncoder, self).__init__()
        
        # self.att_dense = nn.Linear(dim2, 200)
        self.tanh = nn.Tanh()
        self.dense = nn.Linear(200, 1)
        self.softmax = nn.Softmax(dim=1)
        self.ATTN=Attention(config.input_dim, config.embed_dim, config.num_heads)

    def forward(self, x): #BATCH  X SEQ X FEATURES
        x=self.ATTN(x).mean(1)

        return x #BATCH  X FEATURES

class NewsEnc_config:
    input_dim=300
    embed_dim=300
    num_heads=20

# config1=NewsEnc_config
# ob=NewsEncoder(config1)
# print("NewEncTest-",ob(torch.zeros(6,56,300)).shape)


class TimeAwarePopularity(nn.Module):
    def __init__(self, config):
        super(TimeAwarePopularity, self).__init__()
        
        self.tanh = nn.Tanh()
        self.sigmoid=nn.Sigmoid()
        self.gate = nn.Linear(200, 1)
        self.softmax = nn.Softmax(dim=1)
        self.ATTN=Attention(config.input_dim, config.embed_dim, config.num_heads)
        self.fc_R=nn.Linear(config.recency_emb,config.inter)
        self.fc_C=nn.Linear(config.ctr_emb,config.inter)
        self.gate=nn.Linear(config.input_dim+config.ctr_emb,1)
        self.wc=nn.Linear(config.ctr_emb,config.feature_dim)
        self.wp=nn.Linear(config.inter,config.feature_dim)

    def forward(self, content_embedding,Ctr_Embedding,Recency_Embedding): #BATCH X FEATURES  Batch X ctr_emb_dim  Batch X Recency_emb_dim

        
        Gate=self.sigmoid(self.gate(torch.cat([content_embedding,Recency_Embedding],1)))
        p_hat_R=self.fc_R(self.tanh(Recency_Embedding))
        p_hat_C=self.fc_C(self.tanh(content_embedding))
        p_hat=(Gate*p_hat_C)+(1-Gate)*p_hat_R
        s_p=self.wc(self.tanh(Ctr_Embedding))+self.wp(self.tanh(p_hat))

        return s_p


class Popularity_config:
    input_dim=300
    embed_dim=300
    num_heads=20
    inter=600
    ctr_emb=300
    recency_emb=300
    feature_dim=300

# config2=Popularity_config
# ob=TimeAwarePopularity(config2)
# print("TimeAwarePopularityTest-",ob(torch.zeros(6,300),torch.zeros(6,300),torch.zeros(6,300)).shape)







class UserEncoder(nn.Module):
    def __init__(self,config ):
        super(UserEncoder, self).__init__()
        
        self.tanh = nn.Tanh()
        self.sigmoid=nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.ATTN=Attention(config.input_dim, config.embed_dim, config.num_heads)
        self.wu=nn.Linear(config.popularity_feat_dim+config.embed_dim,config.inter)
        self.alpha_linear=nn.Linear(config.inter,1)

    def forward(self, Popular_embedding,News_Embedding): #B x history X Feat  B x history X Feat

        Attend_News=self.ATTN(self.tanh(News_Embedding))
        alpha_i=F.softmax(self.alpha_linear(self.tanh(self.wu(torch.cat([Attend_News,Popular_embedding],2)))).squeeze(2),1) # B X History
        user_embedding=News_Embedding*alpha_i.unsqueeze(2)
        user_embedding=user_embedding.sum(1)
        return user_embedding  #B X Feat

class Userenc_config:
    input_dim=300
    embed_dim=300
    num_heads=20
    popularity_feat_dim=300
    inter=600


# config3=Userenc_config()
# ob=UserEncoder(config3)
# print("UserEncoderTest-",ob(torch.zeros(6,30,200),torch.zeros(6,30,300)).shape)








class PPREC(nn.Module):
    def __init__(self,General_Config,Popular_Config,UserENC_config,NewsEnc_config,word2vec ):
        super(PPREC, self).__init__()
        self.Pop_encoder=TimeAwarePopularity(Popular_Config)
        self.User_encoder=UserEncoder(UserENC_config)
        self.News_encoder=NewsEncoder(NewsEnc_config)
        self.history_size=General_Config.max_history_size
        self.max_history_size=General_Config.max_history_size
        # self.max_seq_length=General_Config.max_seq_length
        self.PN_ratio=General_Config.PN_ratio
        self.embed=nn.Embedding.from_pretrained(word2vec)#(General_Config.vocab,General_Config.word_emb_dim)#.from_pretrained(word2vec)

        self.embed_Rec=nn.Linear(1,General_Config.Rec_emb_len)
        self.embed_ctr=nn.Linear(1,General_Config.ctr_emb_len)

        self.fc_last=nn.Linear(UserENC_config.embed_dim+Popular_Config.feature_dim,1)
        self.fc_shorten=nn.Linear(word2vec.shape[1],General_Config.word_emb_dim)
        self.tanh=nn.Tanh()
        self.extra_fc=nn.Linear(300,768)

        
    def forward(self,candidate,history ): #B x PN_ratio X seqlength   B x history X seqlength
        batch=candidate.shape[0]
        # print(candidate.shape,history.shape)
        candidate=candidate.reshape(batch*self.PN_ratio,-1)[:,:-2]    #B_PN  X  seq_len
        history=history.reshape(batch*self.max_history_size,-1)[:,:-2] #B_MH X seqlen
        # print(candidate.shape,history.shape)
        ctr_cand=candidate.reshape(batch*self.PN_ratio,-1)[:,-2].unsqueeze(1)  #B_PN X 1
        Recency_cand=candidate.reshape(batch*self.PN_ratio,-1)[:,-1].unsqueeze(1) #B_PN X 1

        ctr_hist=history.reshape(batch*self.max_history_size,-1)[:,-2].unsqueeze(1)  #B_MH X 1
        Recency_hist=history.reshape(batch*self.max_history_size,-1)[:,-1].unsqueeze(1) #B_MH X 1

        # print(ctr_cand.shape,Recency_cand.shape,ctr_hist.shape,Recency_hist.shape)
        

        embed_candidate= self.fc_shorten(self.tanh(self.embed(candidate))) #B_PN x seqlength X Feat 
        embed_history= self.fc_shorten(self.tanh(self.embed(history))) #B_MH x seqlength X Feat 
        # print(embed_candidate.shape,embed_history.shape)

        embed_cand_Rec=self.embed_Rec(Recency_cand.float()) #B_PN  X Feat_rec
        embed_hist_Rec=self.embed_Rec(Recency_hist.float()) #B_MH  X Feat_rec

        embed_cand_ctr=self.embed_ctr(ctr_cand.float()) #B_PN  X Feat_ctr
        embed_hist_ctr=self.embed_ctr(ctr_hist.float()) #B_MH  X Feat_ctr
        # print("emb",embed_cand_ctr.shape,embed_hist_ctr.shape)

        news_embedding_candidate=self.tanh(self.News_encoder(self.tanh(embed_candidate))) #B_PN X Feat_News
        news_embedding_history=self.tanh(self.News_encoder(self.tanh(embed_history))) #B_MH X Feat_News

        # print("news",news_embedding_candidate.shape,news_embedding_history.shape)

        Pop_embedding_candidate=self.Pop_encoder(news_embedding_candidate,embed_cand_ctr,embed_cand_Rec) 
        Pop_embedding_history=self.Pop_encoder(news_embedding_history,embed_hist_ctr,embed_hist_Rec)

        # print("pop",Pop_embedding_candidate.shape,Pop_embedding_history.shape)

        s_p=Pop_embedding_candidate

        # print("s_p",s_p.shape)
        user_inp_POP=Pop_embedding_history.reshape(Pop_embedding_history.shape[0]//self.max_history_size,self.max_history_size,Pop_embedding_history.shape[1])
        user_inp_News=news_embedding_history.reshape(news_embedding_history.shape[0]//self.max_history_size,self.max_history_size,news_embedding_history.shape[1])
        # print("user_inp",user_inp_POP.shape,user_inp_News.shape)
        user_embedding=self.User_encoder(user_inp_POP,user_inp_News)
        # print("UserEMb",user_embedding.shape,news_embedding_candidate.reshape(batch,self.PN_ratio,-1).shape)
        s_m=user_embedding.unsqueeze(1)*news_embedding_candidate.reshape(batch,self.PN_ratio,-1)
        s_p=s_p.reshape(batch,self.PN_ratio,-1)
        # print('S',s_m.shape,s_p.shape)
        logits=self.fc_last(torch.cat([s_m,s_p],2)).squeeze(2)
        # print("logits",logits.shape)


        return logits,self.extra_fc(self.tanh(news_embedding_history))


class General_Config:
  max_history_size=10
  word_emb_dim= 300
  vocab=32000
  PN_ratio=5
  Rec_emb_len=300
  ctr_emb_len=300

# config4=General_Config()
# ob=PPREC(config4,config2,config3,config1)
# print("PPRECTest-",ob(torch.zeros(6,5,34).long(),torch.zeros(6,10,34).long()).shape)














