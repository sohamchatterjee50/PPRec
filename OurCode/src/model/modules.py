import torch
from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

# class SelfAttention(nn.Module):
#     """Multi-head self attention implementation.

#     Args:
#         multiheads (int): The number of heads.
#         head_dim (object): Dimension of each head.
#         mask_right (boolean): whether to mask right words.
#     """

#     def __init__(self, multiheads, head_dim, mask_right=False, seed=0):
#         """Initialization steps for SelfAttention.

#         Args:
#             multiheads (int): The number of heads.
#             head_dim (object): Dimension of each head.
#             mask_right (boolean): whether to mask right words.
#         """
#         super(SelfAttention, self).__init__()
#         self.multiheads = multiheads
#         self.head_dim = head_dim
#         self.output_dim = multiheads * head_dim
#         self.mask_right = mask_right
#         self.seed = seed

#         self.WQ = nn.Parameter(torch.randn(head_dim, self.output_dim))
#         self.WK = nn.Parameter(torch.randn(head_dim, self.output_dim))
#         self.WV = nn.Parameter(torch.randn(head_dim, self.output_dim))

#         nn.init.xavier_uniform_(self.WQ)
#         nn.init.xavier_uniform_(self.WK)
#         nn.init.xavier_uniform_(self.WV)

#     def mask(self, inputs, seq_len, mode="add"):
#         """Mask operation used in multi-head self attention

#         Args:
#             seq_len (object): sequence length of inputs.
#             mode (str): mode of mask.

#         Returns:
#             object: tensors after masking.
#         """
#         if seq_len is None:
#             return inputs
#         else:
#             mask = torch.arange(inputs.size(1), device=inputs.device).expand(len(seq_len), inputs.size(1)) < seq_len.unsqueeze(1)
#             mask = 1 - mask.float()
#             mask = mask.unsqueeze(1).unsqueeze(3) if len(inputs.shape) > 3 else mask.unsqueeze(1)

#             if mode == "mul":
#                 return inputs * mask
#             elif mode == "add":
#                 return inputs - (1 - mask) * 1e12

#     def forward(self, QKVs):
#         """Core logic of multi-head self attention.

#         Args:
#             QKVs (list): inputs of multi-head self attention i.e. query, key and value.

#         Returns:
#             object: output tensors.
#         """
#         if len(QKVs) == 3:
#             Q_seq, K_seq, V_seq = QKVs
#             Q_len, V_len = None, None
#         elif len(QKVs) == 5:
#             Q_seq, K_seq, V_seq, Q_len, V_len = QKVs
#         #print(self.WQ.shape)
#         Q_seq = Q_seq.matmul(self.WQ)
#         Q_seq = Q_seq.view(-1, Q_seq.size(1), self.multiheads, self.head_dim).permute(0, 2, 1, 3)

#         K_seq = K_seq.matmul(self.WK)
#         K_seq = K_seq.view(-1, K_seq.size(1), self.multiheads, self.head_dim).permute(0, 2, 1, 3)

#         V_seq = V_seq.matmul(self.WV)
#         V_seq = V_seq.view(-1, V_seq.size(1), self.multiheads, self.head_dim).permute(0, 2, 1, 3)

#         A = Q_seq.matmul(K_seq.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
#         A = A.permute(0, 3, 2, 1)
#         A = self.mask(A, V_len, "add")
#         A = A.permute(0, 3, 2, 1)

#         if self.mask_right:
#             mask = torch.tril(torch.ones(A.size(-2), A.size(-1), device=A.device)).unsqueeze(0).unsqueeze(0) * -1e12
#             A = A + mask

#         A = F.softmax(A, dim=-1)

#         O_seq = A.matmul(V_seq)
#         O_seq = O_seq.permute(0, 2, 1, 3).contiguous()
#         O_seq = O_seq.view(-1, O_seq.size(2), self.output_dim)
#         O_seq = self.mask(O_seq, Q_len, "mul")

#         return O_seq

# Example usage
# multiheads = 8
# head_dim = 64
# self_attention = SelfAttention(multiheads, head_dim)
# Q_seq = torch.randn(32, 10, 64)  # Batch of 32, sequence length 10, embedding size 64
# K_seq = torch.randn(32, 10, 64)
# V_seq = torch.randn(32, 10, 64)
# output = self_attention([Q_seq, K_seq, V_seq])

class KnowledgeAwareNewsEncoder(nn.Module):
    def __init__(self,hparams,
        word2vec_embedding=None,
        seed=None,
        **kwargs,):
        super().__init__()
        # self.word_self_attention = SelfAttention(hparams.head_num, hparams.head_dim)
        # self.entity_self_attention = SelfAttention(hparams.head_num, hparams.head_dim)
        # self.word_cross_attention = SelfAttention(hparams.head_num, hparams.head_dim)
        # self.entity_cross_attention = SelfAttention(hparams.head_num, hparams.head_dim)

        self.word_self_attention = torch.nn.MultiheadAttention(hparams.embed_dim,hparams.head_num,batch_first=True)
        self.entity_self_attention = torch.nn.MultiheadAttention(hparams.embed_dim,hparams.head_num,batch_first=True)
        self.word_cross_attention = torch.nn.MultiheadAttention(hparams.embed_dim,hparams.head_num, batch_first=True)
        self.entity_cross_attention = torch.nn.MultiheadAttention(hparams.embed_dim,hparams.head_num, batch_first=True)

        
        self.word2vec = nn.Embedding.from_pretrained(torch.tensor(word2vec_embedding))
        self.final_attention_layer = torch.nn.MultiheadAttention(hparams.embed_dim,hparams.head_num, batch_first=True)
        
        
        

    def forward(self, words, entities):
        words = torch.tensor(words)
        entities = torch.tensor(entities)
        # words = torch.reshape(words,(words.shape[0],words.shape[1]*words.shape[2]))
        # entities = torch.reshape(entities,(entities.shape[0],entities.shape[1]*entities.shape[2]))
        word_embeddings = self.word2vec(words)
        entity_embeddings =  self.word2vec(entities)
        # print(word_embeddings.shape)
        # print(entity_embeddings.shape)
        word_embeddings = torch.reshape(word_embeddings,(word_embeddings.shape[0],word_embeddings.shape[1]*word_embeddings.shape[2],word_embeddings.shape[3]))
        entity_embeddings = torch.reshape(entity_embeddings,(entity_embeddings.shape[0],entity_embeddings.shape[1]*entity_embeddings.shape[2],entity_embeddings.shape[3]))
        # print(word_embeddings.shape)
        # print(entity_embeddings.shape)

        word_self_attn_output,_ = self.word_self_attention(word_embeddings, word_embeddings, word_embeddings)
        #print("Shape after word attention:",word_self_attn_output.shape)
        entity_self_attn_output,_ = self.entity_self_attention(entity_embeddings, entity_embeddings, entity_embeddings)
        #print("Shape after entity self attention:",entity_self_attn_output.shape)

        word_cross_output,_ = self.word_cross_attention(word_embeddings,entity_embeddings,entity_embeddings)
        entity_cross_output,_ = self.word_cross_attention(entity_embeddings, word_embeddings, word_embeddings)
        #print("Word cross attention:",word_cross_output.shape)
        print("Entity cross attention:",entity_cross_output.shape)

        
        word_output = torch.add(word_self_attn_output,word_cross_output)
        entity_output = torch.add(entity_self_attn_output,entity_cross_output)
        #print(word_output.shape)
        #print(entity_output.shape)
        news_encoder,_ = self.final_attention_layer(word_output, entity_output, entity_output)
        return news_encoder



class TimeAwarePopularityEncoder(nn.Module):
    def __init__(self,word2vec_embedding=None,
        seed=None,
        **kwargs,):
        super(TimeAwarePopularityEncoder, self).__init__()
        self.word2vec = nn.Embedding.from_pretrained(torch.tensor(word2vec_embedding))
        self.news_model = nn.Sequential(
          nn.Linear(768,256),
          nn.Tanh(),
          nn.Linear(256,256),
          nn.Tanh(),
          nn.Linear(256,128),
          nn.Tanh(),
          nn.Linear(128,1,bias=False)
        )
        self.dense = nn.Linear(30,1)
        
        
        self.recency_model = nn.Sequential(
            nn.Linear(768,64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,1,bias=False)
        )
        self.gate = nn.Sequential(
            nn.Linear(31,128),
            nn.Tanh(),
            nn.Linear(128,64),
            nn.Tanh(),
            nn.Linear(64,1),
            nn.Sigmoid()
        )
        self.ctr_model = nn.Sigmoid()
        self.combined_embed = nn.Linear(1,1)


    def forward(self,news, recency, ctr):
        news_tensor = torch.tensor(news)
        recency_tensor = torch.tensor(recency)
        ctr_tensor = torch.tensor(ctr)
        news_embed = self.word2vec(news_tensor)
        recency_embed = self.word2vec(recency_tensor)
        ctr_embed = self.word2vec(ctr_tensor)
        content_score = self.news_model(news_embed)
        recency_score = self.recency_model(recency_embed)
        recency_tensor = recency_tensor.unsqueeze(-1)
        combined_input = torch.cat([news_tensor,recency_tensor],2)
        combined_input = combined_input.to(torch.float32)
        combined_score = self.gate(combined_input)
        final_content_score = content_score.squeeze(-1)
        final_content_score = self.dense(final_content_score)
        # print("ALL:",combined_score.shape)
        # print("CONTENT:",final_content_score.shape)
        # print("RECENCY:",recency_score.shape)
        combined_prefinal_score = (1-combined_score)*recency_score+combined_score*final_content_score
        ctr_score = self.ctr_model(ctr_embed)

        combined_final_score = self.combined_embed(combined_prefinal_score)
        return ctr_score+combined_final_score
    
class ContentPopularityJointAttention(nn.Module):
    """

    Implementation of the content-popularity joint attention module
    for the popularity-aware user encoder.

    This is based on formula (2) in 3.4 of the paper.

    """

    def __init__(self, max_clicked: int, m_size: int, p_size: int, weight_size: int):
        super().__init__()

        # Q: should the weights be initialized randomly?
        # Its not stated in the paper, as far as I can see.
        # I guess, there is a better way to initialize them.
        # Lets check in their code, or ask Songga, or do some
        # research on the topic.

        self.Wu = nn.Parameter(torch.rand(512, m_size + p_size))
        self.b = nn.Parameter(torch.rand(weight_size))

        self.weight_size = weight_size
        self.m_size = m_size
        self.p_size = p_size
        self.max_clicked = max_clicked

    def forward(self, m: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """

        Calculates the user interest embeddings u, based on the
        the popularity embeddings p, and the contextual news
        representations m.

        m is a tensor of shape (batch_size, max_clicked, m_size)
        p is a tensor of shape (batch_size, max_clicked, p_size)
        u is a tensor of shape (batch_size, m_size)
        where max_clicked is the number of clicked articles by the user.

        """
        
        # assert len(m.size()) == 3
        # print(m.shape)
        # print(p.shape)
        batch_size, max_clicked, m_size = m.size()
        #print(m.size())
        # print("m_size:",m_size)
        # print("self.m_size",self.m_size)
        # assert m_size == self.m_size
        # print(max_clicked)
        # print(self.max_clicked)
        # assert max_clicked == self.max_clicked
        #print("HHHh")
        # assert len(p.size()) == 3
        # assert p.size(0) == batch_size
        # assert p.size(1) == max_clicked
        # assert p.size(2) == self.p_size
        
       
        
        mp = torch.cat((m, p), dim=2)  # (batch_size, max_clicked, m_size + p_size)
        assert len(mp.size()) == 3
        assert mp.size(0) == batch_size
        assert mp.size(1) == max_clicked
        #assert mp.size(2) == self.m_size + self.p_size
        # print(mp.shape)
        # print(self.Wu.T.shape)
        Wu_mp = torch.matmul(mp, self.Wu)  # (batch_size, max_clicked, weight_size)
        assert len(Wu_mp.size()) == 3
        assert Wu_mp.size(0) == batch_size
        assert Wu_mp.size(1) == max_clicked
        #print(Wu_mp.size(2))
        #assert Wu_mp.size(2) == self.weight_size

        tanh_Wu_mp = torch.tanh(Wu_mp)  # (batch_size, max_clicked, weight_size)
        assert len(tanh_Wu_mp.size()) == 3
        assert tanh_Wu_mp.size(0) == batch_size
        assert tanh_Wu_mp.size(1) == max_clicked
        # assert tanh_Wu_mp.size(2) == self.weight_size
        # print(tanh_Wu_mp.shape)
        # print(self.b.shape)
        b_tanh_Wu_mp = torch.matmul(tanh_Wu_mp, self.b)  # (batch_size, max_clicked)
        assert len(b_tanh_Wu_mp.size()) == 2
        assert b_tanh_Wu_mp.size(0) == batch_size
        assert b_tanh_Wu_mp.size(1) == max_clicked

        sum_b_tanh_Wu_mp = torch.sum(b_tanh_Wu_mp, dim=1)  # (batch_size)
        assert len(sum_b_tanh_Wu_mp.size()) == 1
        assert sum_b_tanh_Wu_mp.size(0) == batch_size

        a = torch.div(
            b_tanh_Wu_mp, sum_b_tanh_Wu_mp.unsqueeze(1)
        )  # (batch_size, max_clicked)
        assert len(a.size()) == 2
        assert a.size(0) == batch_size
        assert a.size(1) == max_clicked

        am = torch.mul(a.unsqueeze(2), m)  # (batch_size, max_clicked, m_size)
        assert len(am.size()) == 3
        assert am.size(0) == batch_size
        assert am.size(1) == max_clicked
        # print(am.shape)
        # print(self.m_size)
        assert am.size(2) == self.m_size

        u = torch.sum(am, dim=1)  # (batch_size, 
        
        assert len(u.size()) == 2
        assert u.size(0) == batch_size
        assert u.size(1) == self.m_size

        return u


class PopularityAwareUserEncoder(nn.Module):
    def __init__(self,
                 hparams,
        word2vec_embedding=None,
                 seed=None,
                **kwargs,):
                 super().__init__()

                 self.word2vec = nn.Embedding.from_pretrained(torch.tensor(word2vec_embedding))
                 self.pop_embed = nn.Sequential(
            nn.Linear(1,256),
            nn.Tanh(),
            nn.Linear(256,256),
            nn.Tanh()
        )
                 self.news_self_attention = SelfAttention(hparams.head_num, hparams.head_dim)
                 self.cpja = ContentPopularityJointAttention(hparams.max_clicked, hparams.m_size, hparams.p_size,hparams.weight_size)
        

    def forward(self,news,popularity):
        pop_tensor = torch.tensor(popularity)
        news_tensor = torch.tensor(news)
        pop_tensor = pop_tensor.to(torch.float32)
        popularity_embedding = self.pop_embed(pop_tensor)
        news_embedding = self.word2vec(news_tensor)
        news_attention_embedding = self.news_self_attention([news_embedding,news_embedding,news_embedding])
        # print(news_attention_embedding.shape)
        # print(popularity_embedding.shape)
        news_attention_embedding = news_attention_embedding.view(-1)
        target_shape=popularity_embedding.shape
        news_attention_embedding = news_attention_embedding[:torch.prod(torch.tensor(target_shape))].view(*target_shape, -1)
        news_attention_embedding = news_attention_embedding.squeeze(-1)
        pop_aware_user_encoder = self.cpja(news_attention_embedding,popularity_embedding)
        return pop_aware_user_encoder
    


class PPRec(nn.Module):
    """

    Implementation of PPRec. Figure 2 in the paper shows the architecture.
    Outputs a ranking score for some candidate news articles.

    """

    def __init__(
        self,
        hparams_pprec,
        word2vec_embedding= None 
    ):

        super().__init__()

       
      
        self.knowledge_news_model  = KnowledgeAwareNewsEncoder(hparams_pprec,word2vec_embedding,seed=123)
        self.user_model = PopularityAwareUserEncoder(hparams_pprec, word2vec_embedding=word2vec_embedding, seed=123)
        self.time_news_model = TimeAwarePopularityEncoder(word2vec_embedding=word2vec_embedding, seed=123)

        
        self.aggregator_gate = nn.Sequential(
            nn.Linear(1,1),
            nn.Sigmoid()
        )

    def forward(
        self,
        title, entities, ctr, recency, popularity
    ):
        """

        Returns the ranking scores for a batch of candidate news articles, given the user's
        past click history.

        """

        
        knowledge_news_embed = self.knowledge_news_model(title, entities)
        print("Knowldege aware news:",knowledge_news_embed.shape)
        news_pop_score = self.time_news_model(title, recency, ctr)
        print("Time aware news:",news_pop_score.shape)
        user_embed = self.user_model(title,popularity)
        print("User embed:",user_embed.shape)
        personalized_score = torch.dot(knowledge_news_embed,user_embed)
        return self.aggregator_gate(news_pop_score) + (1-self.aggregator_gate(personalized_score))

        

        







