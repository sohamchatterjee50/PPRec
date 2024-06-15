import torch
import torch.nn as nn
import math 
import torch.nn.functional as F

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

def test():
    v=Attention(30,60,5)
    d=torch.zeros(42,20,30)
    print(v(d).shape)


test()


class AttentivePooling(nn.Module):
    def __init__(self, dim1, dim2):
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
# model = AttentivePooling(dim1, dim2)
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






















# class Attention(torch.nn.Module):
#     def __init__(self, nb_head, size_per_head, **kwargs):
#         self.nb_head = nb_head
#         self.size_per_head = size_per_head
#         self.output_dim = nb_head*size_per_head
#         super(Attention, self).__init__(**kwargs)
 
#     def build(self, input_shape):
#         # self.WQ = self.add_weight(name='WQ',
#         #                           shape=(input_shape[0][-1], self.output_dim),
#         #                           initializer='glorot_uniform',
#         #                           trainable=True)

#         self.WQ= Parameter(torch.zeros(input_shape[0][-1],self.output_dim))
#         # self.WK = self.add_weight(name='WK',
#         #                           shape=(input_shape[1][-1], self.output_dim),
#         #                           initializer='glorot_uniform',
#         #                           trainable=True)
#         self.WK= Parameter(torch.zeros(input_shape[1][-1],self.output_dim))
#         # self.WV = self.add_weight(name='WV',
#         #                           shape=(input_shape[2][-1], self.output_dim),
#         #                           initializer='glorot_uniform',
#         #                           trainable=True)
#         self.WV=Parameter(torch.zeros(input_shape[2][-1],self.output_dim))

#         self.reset_parameters()
#         super(Attention, self).build(input_shape)
 
#     def Mask(self, inputs, seq_len, mode='mul'):
#         if seq_len == None:
#             return inputs
#         else:
#             mask = K.one_hot(seq_len[:,0], K.shape(inputs)[1])
#             mask = 1 - K.cumsum(mask, 1)
#             for _ in range(len(inputs.shape)-2):
#                 mask = K.expand_dims(mask, 2)
#             if mode == 'mul':
#                 return inputs * mask
#             if mode == 'add':
#                 return inputs - (1 - mask) * 1e12
 
#     def call(self, x):
#         if len(x) == 3:
#             Q_seq,K_seq,V_seq = x
#             Q_len,V_len = None,None
#         elif len(x) == 5:
#             Q_seq,K_seq,V_seq,Q_len,V_len = x
#         Q_seq = K.dot(Q_seq, self.WQ)
#         Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
#         Q_seq = K.permute_dimensions(Q_seq, (0,2,1,3))
#         K_seq = K.dot(K_seq, self.WK)
#         K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
#         K_seq = K.permute_dimensions(K_seq, (0,2,1,3))
#         V_seq = K.dot(V_seq, self.WV)
#         V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
#         V_seq = K.permute_dimensions(V_seq, (0,2,1,3))

#         A = K.batch_dot(Q_seq, K_seq, axes=[3,3]) / self.size_per_head**0.5
#         A = K.permute_dimensions(A, (0,3,2,1))
#         A = self.Mask(A, V_len, 'add')
#         A = K.permute_dimensions(A, (0,3,2,1))
#         A = K.softmax(A)

#         O_seq = K.batch_dot(A, V_seq, axes=[3,2])
#         O_seq = K.permute_dimensions(O_seq, (0,2,1,3))
#         O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
#         O_seq = self.Mask(O_seq, Q_len, 'mul')
#         return O_seq
 
#     def compute_output_shape(self, input_shape):
#         return (input_shape[0][0], input_shape[0][1], self.output_dim)

# def AttentivePooling(dim1,dim2):
#     vecs_input = Input(shape=(dim1,dim2),dtype='float32') #(50,400)
#     user_vecs =Dropout(0.2)(vecs_input)
#     user_att = Dense(200,activation='tanh')(user_vecs) # (50,200)
#     user_att = Flatten()(Dense(1)(user_att)) # (50,)
#     user_att = Activation('softmax')(user_att)  # (50,)
#     user_vec = keras.layers.Dot((1,1))([user_vecs,user_att])  # (400,)
#     model = Model(vecs_input,user_vec)
#     return model

# def AttentivePoolingQKY(dim1,dim2,dim3):
#     vecs_input = Input(shape=(dim1,dim2),dtype='float32')
#     value_input = Input(shape=(dim1,dim3),dtype='float32')
#     user_vecs =Dropout(0.2)(vecs_input)
#     user_att = Dense(200,activation='tanh')(user_vecs)
#     user_att = Flatten()(Dense(1)(user_att))
#     user_att = Activation('softmax')(user_att)
#     user_vec = keras.layers.Dot((1,1))([value_input,user_att])
#     model = Model([vecs_input,value_input],user_vec)
#     return model

# def AttentivePooling_bias(dim1,dim2,dim3):
#     bias_input = Input(shape=(dim1,dim2),dtype='float32')
#     value_input = Input(shape=(dim1,dim3),dtype='float32')
    
#     bias_vecs =Dropout(0.2)(bias_input)
#     user_att = Dense(200,activation='tanh')(user_vecs)
#     user_att = Flatten()(Dense(1)(user_att))
#     user_att = Activation('softmax')(user_att)
#     user_vec = keras.layers.Dot((1,1))([value_input,user_att])
#     model = Model([vecs_input,value_input],user_vec)
#     return model