import torch
from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


"""  Knowledge Aware News Encoder which uses self attention and cross attention modules    """
class KnowledgeAwareNewsEncoder(nn.Module):
    def __init__(self,hparams,
        word2vec_embedding=None,
        seed=None,
        **kwargs,):
        super().__init__()
        

        self.word_self_attention = torch.nn.MultiheadAttention(hparams.embed_dim,hparams.head_num,batch_first=True)
        self.entity_self_attention = torch.nn.MultiheadAttention(hparams.embed_dim,hparams.head_num,batch_first=True)
        self.word_cross_attention = torch.nn.MultiheadAttention(hparams.embed_dim,hparams.head_num, batch_first=True)
        self.entity_cross_attention = torch.nn.MultiheadAttention(hparams.embed_dim,hparams.head_num, batch_first=True)

        
        self.word2vec = nn.Embedding.from_pretrained(word2vec_embedding)
        self.entity2vec = nn.Embedding.from_pretrained(word2vec_embedding)
        self.final_attention_layer = torch.nn.MultiheadAttention(hparams.embed_dim,hparams.head_num, batch_first=True)
        
        
        

    def forward(self, words, entities):
        word_embeddings = self.word2vec(words)
        entity_embeddings =  self.entity2vec(entities)
        word_embeddings = torch.reshape(word_embeddings,(word_embeddings.shape[0],word_embeddings.shape[1]*word_embeddings.shape[2],word_embeddings.shape[3]))
        entity_embeddings = torch.reshape(entity_embeddings,(entity_embeddings.shape[0],entity_embeddings.shape[1]*entity_embeddings.shape[2],entity_embeddings.shape[3]))
        
        """ Word level self attention """
        word_self_attn_output,_ = self.word_self_attention(word_embeddings, word_embeddings, word_embeddings)
        
        """ Entity(in this case NER clusters) level self attention """
        entity_self_attn_output,_ = self.entity_self_attention(entity_embeddings, entity_embeddings, entity_embeddings)
        
        """ Cross attention between words and entities   """
        word_cross_output,_ = self.word_cross_attention(word_embeddings,entity_embeddings,entity_embeddings)
        entity_cross_output,_ = self.word_cross_attention(entity_embeddings, word_embeddings, word_embeddings)
        

        word_output = torch.add(word_self_attn_output,word_cross_output)
        entity_output = torch.add(entity_self_attn_output,entity_cross_output)
        news_encoder,_ = self.final_attention_layer(word_output, entity_output, entity_output)
        return news_encoder



class TimeAwarePopularityEncoder(nn.Module):
    def __init__(self,word2vec_embedding=None,
        seed=None,
        **kwargs,):
        super(TimeAwarePopularityEncoder, self).__init__()
        self.word2vec = nn.Embedding.from_pretrained(word2vec_embedding)
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
        news_embed = self.word2vec(news)
        recency_embed = self.word2vec(recency)
        ctr_embed = self.word2vec(ctr)
        content_score = self.news_model(news_embed)
        recency_score = self.recency_model(recency_embed)
        recency_tensor = recency.unsqueeze(-1)

        combined_input = torch.cat([news,recency_tensor],2)
        combined_input = combined_input.to(torch.float32)

        combined_score = self.gate(combined_input)
        final_content_score = content_score.squeeze(-1)
        final_content_score = self.dense(final_content_score)
        
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
        self.Wu = nn.Parameter(torch.rand(weight_size, m_size + p_size))
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

        assert len(m.size()) == 3
        batch_size, max_clicked, m_size = m.size()
        assert m_size == self.m_size
        assert max_clicked == self.max_clicked

        assert len(p.size()) == 3
        assert p.size(0) == batch_size
        assert p.size(1) == max_clicked
        assert p.size(2) == self.p_size

        mp = torch.cat((m, p), dim=2)  # (batch_size, max_clicked, m_size + p_size)
        assert len(mp.size()) == 3
        assert mp.size(0) == batch_size
        assert mp.size(1) == max_clicked
        assert mp.size(2) == self.m_size + self.p_size

        Wu_mp = torch.matmul(mp, self.Wu.T)  # (batch_size, max_clicked, weight_size)
        assert len(Wu_mp.size()) == 3
        assert Wu_mp.size(0) == batch_size
        assert Wu_mp.size(1) == max_clicked
        assert Wu_mp.size(2) == self.weight_size

        tanh_Wu_mp = torch.tanh(Wu_mp)  # (batch_size, max_clicked, weight_size)
        assert len(tanh_Wu_mp.size()) == 3
        assert tanh_Wu_mp.size(0) == batch_size
        assert tanh_Wu_mp.size(1) == max_clicked
        assert tanh_Wu_mp.size(2) == self.weight_size

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
        assert am.size(2) == self.m_size

        u = torch.sum(am, dim=1)  # (batch_size, m_size)
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

                 self.word2vec = nn.Embedding.from_pretrained(word2vec_embedding)
        
                 self.pop_embed = nn.Embedding.from_pretrained(word2vec_embedding)
                 self.news_self_attention = torch.nn.MultiheadAttention(hparams.embed_dim,hparams.head_num, batch_first=True)
                 self.cpja = ContentPopularityJointAttention(hparams.max_clicked, hparams.m_size, hparams.p_size,hparams.weight_size)
                 self.max_clicked = hparams.max_clicked
                 self.title_length = hparams.title_size

    def forward(self,news,popularity):
        
        
        popularity_embedding = self.pop_embed(popularity)
        popularity_embedding = popularity_embedding.squeeze(axis=2)
        news_embedding = self.word2vec(news)
        news_embedding = torch.reshape(news_embedding,(news_embedding.shape[0],news_embedding.shape[1]*news_embedding.shape[2],news_embedding.shape[3]))
        news_attention_embedding,_ = self.news_self_attention(news_embedding,news_embedding,news_embedding)

        news_attention_embedding = torch.reshape(news_attention_embedding,(news_attention_embedding.shape[0],self.max_clicked,self.title_length,news_attention_embedding.shape[2]))
        news_attention_embedding = torch.mean(news_attention_embedding, dim=2, keepdim=False)
        
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

        self.knowledge_news_model  = KnowledgeAwareNewsEncoder(hparams_pprec,torch.from_numpy(word2vec_embedding),seed=123)
        self.user_model = PopularityAwareUserEncoder(hparams_pprec, word2vec_embedding=torch.from_numpy(word2vec_embedding), seed=123)
        self.time_news_model = TimeAwarePopularityEncoder(word2vec_embedding=torch.from_numpy(word2vec_embedding), seed=123)

        
        self.aggregator_gate = nn.Sequential(
            nn.Linear(5,5),
            nn.Sigmoid()
        )
        self.title_size = hparams_pprec.title_size
        self.softmax = nn.Softmax(dim=1)

    def forward(
        self,
        title, entities, ctr, recency, hist_title, hist_popularity
    ):
        """

        Returns the ranking scores for a batch of candidate news articles, given the user's
        past click history.

        """

        
        knowledge_news_embed = self.knowledge_news_model(title, entities)
        
        time_aware_pop = self.time_news_model(title, recency, ctr)
        
        user_embed = self.user_model(hist_title,hist_popularity)
        
        time_aware_pop = torch.mean(time_aware_pop, dim=2, keepdim=False)
        knowledge_news_embed = torch.reshape(knowledge_news_embed, (knowledge_news_embed.shape[0],int(knowledge_news_embed.shape[1]/self.title_size), self.title_size,knowledge_news_embed.shape[2]))
        knowledge_news_embed = torch.mean(knowledge_news_embed, dim=2, keepdim=False)
        
        personalized_score = torch.matmul(knowledge_news_embed,user_embed.T)
        
        score1 =  self.aggregator_gate(time_aware_pop) 
        
        personalized_score = torch.mean(personalized_score,dim=2,keepdim=False)
        score2 =  (1-self.aggregator_gate(personalized_score))
        
        score = score1 + score2
        return self.softmax(score)

        
class BPELoss(nn.Module):
    def __init__(self):
        super(BPELoss, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, output, target): 
         
         batch_size = target.shape[0]
         total_no_samples = target.shape[1]
         
         
         mask = target > 0
         postive_index_select = torch.masked_select(output, mask)
         
         neg_mask = target == 0
         negative_index_select = torch.masked_select(output, neg_mask)
         negative_index_select = torch.reshape(negative_index_select,(batch_size,total_no_samples-1))
         
         negative_index_select,_ = torch.min(negative_index_select, dim=1, keepdim = True)
         diff = torch.sub(postive_index_select, negative_index_select)
         diff_sig = self.sigmoid(diff)
         diff_log = torch.log(diff_sig)
         return - torch.mean(diff_log)
         


def train_one_epoch(epoch_index, tb_writer, train_dataloader,optimizer,model,loss_fn,device):
    running_loss = 0.
    last_loss = 0.

    
    for i, data in enumerate(train_dataloader):
        # Every data instance is an input + label pair
        
        inputs, labels = data
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        title = inputs[5]
        entities = inputs[6]
        ctr = inputs[7]
        recency = inputs[8]
        hist_title = inputs[0]
        hist_popularity = inputs[2]
        
        title = torch.from_numpy(title)
        entities = torch.from_numpy(entities)
        ctr = torch.from_numpy(ctr)
        recency = torch.from_numpy(recency)
        hist_title = torch.from_numpy(hist_title)
        hist_popularity = torch.from_numpy(hist_popularity)
        labels = torch.from_numpy(labels)
        
        title = title.to(device)
        entities = entities.to(device)
        ctr = ctr.to(device)
        recency = recency.to(device)
        hist_title = hist_title.to(device)
        hist_popularity = hist_popularity.to(device)
        labels = labels.to(device)

        outputs = model(title, entities, ctr, recency ,hist_title, hist_popularity )
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        
        
        print('  batch {} loss: {}'.format(i + 1, running_loss))
        tb_x = epoch_index * len(train_dataloader) + i + 1
        tb_writer.add_scalar('Loss/train', running_loss, tb_x)

    return running_loss, i+1       