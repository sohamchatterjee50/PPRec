import torch
from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class SelfAttention(nn.Module):
    """Multi-head self attention implementation.

    Args:
        multiheads (int): The number of heads.
        head_dim (object): Dimension of each head.
        mask_right (boolean): whether to mask right words.
    """

    def __init__(self, multiheads, head_dim, mask_right=False, seed=0):
        """Initialization steps for SelfAttention.

        Args:
            multiheads (int): The number of heads.
            head_dim (object): Dimension of each head.
            mask_right (boolean): whether to mask right words.
        """
        super(SelfAttention, self).__init__()
        self.multiheads = multiheads
        self.head_dim = head_dim
        self.output_dim = multiheads * head_dim
        self.mask_right = mask_right
        self.seed = seed

        self.WQ = nn.Parameter(torch.randn(head_dim, self.output_dim))
        self.WK = nn.Parameter(torch.randn(head_dim, self.output_dim))
        self.WV = nn.Parameter(torch.randn(head_dim, self.output_dim))

        nn.init.xavier_uniform_(self.WQ)
        nn.init.xavier_uniform_(self.WK)
        nn.init.xavier_uniform_(self.WV)

    def mask(self, inputs, seq_len, mode="add"):
        """Mask operation used in multi-head self attention

        Args:
            seq_len (object): sequence length of inputs.
            mode (str): mode of mask.

        Returns:
            object: tensors after masking.
        """
        if seq_len is None:
            return inputs
        else:
            mask = torch.arange(inputs.size(1), device=inputs.device).expand(len(seq_len), inputs.size(1)) < seq_len.unsqueeze(1)
            mask = 1 - mask.float()
            mask = mask.unsqueeze(1).unsqueeze(3) if len(inputs.shape) > 3 else mask.unsqueeze(1)

            if mode == "mul":
                return inputs * mask
            elif mode == "add":
                return inputs - (1 - mask) * 1e12

    def forward(self, QKVs):
        """Core logic of multi-head self attention.

        Args:
            QKVs (list): inputs of multi-head self attention i.e. query, key and value.

        Returns:
            object: output tensors.
        """
        if len(QKVs) == 3:
            Q_seq, K_seq, V_seq = QKVs
            Q_len, V_len = None, None
        elif len(QKVs) == 5:
            Q_seq, K_seq, V_seq, Q_len, V_len = QKVs

        Q_seq = Q_seq.matmul(self.WQ)
        Q_seq = Q_seq.view(-1, Q_seq.size(1), self.multiheads, self.head_dim).permute(0, 2, 1, 3)

        K_seq = K_seq.matmul(self.WK)
        K_seq = K_seq.view(-1, K_seq.size(1), self.multiheads, self.head_dim).permute(0, 2, 1, 3)

        V_seq = V_seq.matmul(self.WV)
        V_seq = V_seq.view(-1, V_seq.size(1), self.multiheads, self.head_dim).permute(0, 2, 1, 3)

        A = Q_seq.matmul(K_seq.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        A = A.permute(0, 3, 2, 1)
        A = self.mask(A, V_len, "add")
        A = A.permute(0, 3, 2, 1)

        if self.mask_right:
            mask = torch.tril(torch.ones(A.size(-2), A.size(-1), device=A.device)).unsqueeze(0).unsqueeze(0) * -1e12
            A = A + mask

        A = F.softmax(A, dim=-1)

        O_seq = A.matmul(V_seq)
        O_seq = O_seq.permute(0, 2, 1, 3).contiguous()
        O_seq = O_seq.view(-1, O_seq.size(2), self.output_dim)
        O_seq = self.mask(O_seq, Q_len, "mul")

        return O_seq

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
        self.word_self_attention = SelfAttention(hparams.multiheads, hparams.head_dim)
        self.entity_self_attention = SelfAttention(hparams.multiheads, hparams.head_dim)
        self.word_cross_attention = SelfAttention(hparams.multiheads, hparams.head_dim)
        self.entity_cross_attention = SelfAttention(hparams.multiheads, hparams.head_dim)
        self.word2vec = word2vec_embedding
        self.final_attention_layer = torch.nn.MultiheadAttention(hparams.embed_dim,hparams.num_heads)
        
        
        

    def forward(self, words, entities):
        word_embeddings = self.word2vec(words)
        entity_embeddings =  self.word2vec(entities)

        word_self_attn_output, word_attn_output_weights = self.word_self_attention(word_embeddings, word_embeddings, word_embeddings)
        entity_self_attn_output, entity_attn_output_weights = self.entity_self_attention(entity_embeddings, entity_embeddings, entity_embeddings)

        word_cross_output, word_attn_outptut_weights = self.word_cross_attention(word_embeddings,entity_embeddings,entity_embeddings)
        entity_cross_output, entity_cross_output_weights = self.word_cross_attention(entity_embeddings, word_embeddings, word_embeddings)

        
        word_output = torch.add(word_self_attn_output,word_cross_output)
        entity_output = torch.add(entity_self_attn_output,entity_cross_output)

        news_encoder = self.final_attention_layer(word_output, entity_output, entity_output)
        return news_encoder


        



class TimeAwarePopularityEncoderder(nn.Module):
    def __init__(self,news_input_shape,
                 recency_input_shape,
        seed=None,
        **kwargs,):
        self.news_embed = nn.Sequential(
          nn.Linear(news_input_shape,256),
          nn.Tanh(),
          nn.Linear(256,256),
          nn.Tanh(),
          nn.Linear(256,128),
          nn.Tanh(),
          nn.Linear(128,1,bias=False)
        )
        
        
        self.recency_embed = nn.Sequential(
            nn.Linear(recency_input_shape,64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,1,bias=False)
        )
        self.gate = nn.Sequential(
            nn.Linear(recency_input_shape+news_input_shape,128),
            nn.Tanh(),
            nn.Linear(128,64),
            nn.Tanh()
            nn.Linear(64,1),
            nn.Sigmoid()
        )
        self.ctr_embed = nn.Sequential(
        nn.Parameter(torch.randn_like(768)),
        nn.Sigmoid()
      )
        self.combined_embed = nn.Parameter(torch.randn_like(1))


    def forward(self,news, recency, ctr):
        content_score = self.news_embed(news)
        recency_score = self.recency_embed(recency)
        combined_input = torch.concat([news,recency])
        combined_score = self.gate(combined_input)
        combined_prefinal_score = (1-combined_score)*content_score+combined_score*recency_score
        ctr_score = self.ctr_embed(ctr)
        combined_final_score = self.combined_embed(combined_prefinal_score)
        return ctr_score+combined_final_score

class PopularityAwareUserEncoder(nn.Module):
    



