import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, nb_head, size_per_head):
        super(Attention, self).__init__()
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head * size_per_head

        self.WQ = nn.Linear(nb_head, self.output_dim)
        self.WK = nn.Linear(nb_head, self.output_dim)
        self.WV = nn.Linear(nb_head, self.output_dim)

    def mask(self, inputs, seq_len, mode='mul'):
        if seq_len is None:
            return inputs
        else:
            mask = torch.nn.functional.one_hot(seq_len[:, 0], inputs.size(1)).float()
            mask = 1 - torch.cumsum(mask, dim=1)
            mask = mask.unsqueeze(2).expand(-1, -1, inputs.size(2))
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12

    def forward(self, x):
        if len(x) == 3:
            Q_seq, K_seq, V_seq = x
            Q_len, V_len = None, None
        elif len(x) == 5:
            Q_seq, K_seq, V_seq, Q_len, V_len = x

        Q_seq = self.WQ(Q_seq)
        Q_seq = Q_seq.view(-1, Q_seq.size(1), self.nb_head, self.size_per_head).permute(0, 2, 1, 3)
        K_seq = self.WK(K_seq)
        K_seq = K_seq.view(-1, K_seq.size(1), self.nb_head, self.size_per_head).permute(0, 2, 1, 3)
        V_seq = self.WV(V_seq)
        V_seq = V_seq.view(-1, V_seq.size(1), self.nb_head, self.size_per_head).permute(0, 2, 1, 3)

        A = torch.matmul(Q_seq, K_seq.transpose(-1, -2)) / (self.size_per_head ** 0.5)
        A = self.mask(A.permute(0, 3, 2, 1), V_len, 'add').permute(0, 3, 2, 1)
        A = F.softmax(A, dim=-1)

        O_seq = torch.matmul(A, V_seq).permute(0, 2, 1, 3).contiguous()
        O_seq = O_seq.view(-1, O_seq.size(1), self.output_dim)
        O_seq = self.mask(O_seq, Q_len, 'mul')
        return O_seq

class AttentivePooling(nn.Module):
    def __init__(self, dim1, dim2):
        super(AttentivePooling, self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.attention = nn.Sequential(
            nn.Linear(dim2, 200),
            nn.Tanh(),
            nn.Linear(200, 1),
            nn.Flatten(),
            nn.Softmax(dim=1)
        )

    def forward(self, vecs_input):
        user_vecs = self.dropout(vecs_input)
        user_att = self.attention(user_vecs)
        user_vec = torch.matmul(user_att.unsqueeze(1), user_vecs).squeeze(1)
        return user_vec

class AttentivePoolingQKY(nn.Module):
    def __init__(self, dim1, dim2, dim3):
        super(AttentivePoolingQKY, self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.attention = nn.Sequential(
            nn.Linear(dim2, 200),
            nn.Tanh(),
            nn.Linear(200, 1),
            nn.Flatten(),
            nn.Softmax(dim=1)
        )

    def forward(self, vecs_input, value_input):
        user_vecs = self.dropout(vecs_input)
        user_att = self.attention(user_vecs)
        user_vec = torch.matmul(user_att.unsqueeze(1), value_input).squeeze(1)
        return user_vec

class AttentivePoolingBias(nn.Module):
    def __init__(self, dim1, dim2, dim3):
        super(AttentivePoolingBias, self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.attention = nn.Sequential(
            nn.Linear(dim2, 200),
            nn.Tanh(),
            nn.Linear(200, 1),
            nn.Flatten(),
            nn.Softmax(dim=1)
        )

    def forward(self, bias_input, value_input):
        bias_vecs = self.dropout(bias_input)
        user_att = self.attention(bias_vecs)
        user_vec = torch.matmul(user_att.unsqueeze(1), value_input).squeeze(1)
        return user_vec