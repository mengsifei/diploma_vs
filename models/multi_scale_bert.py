import torch
from torch import nn
from transformers import AutoModel


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class mainplm(nn.Module):
    def __init__(self):
        super(mainplm, self).__init__()
        self.plm = AutoModel.from_pretrained('bert-base-cased')
        self.mlp = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(self.plm.config.hidden_size, 4)
        )
        self.mlp.apply(init_weights)

    def forward(self, input_ids, attention_mask, token_type_ids):
        batch_size, num_chunks, seq_length = input_ids.size()
        cls_output = torch.zeros((batch_size, self.plm.config.hidden_size), device=input_ids.device)
        for i in range(num_chunks):
            chunk_input_ids = input_ids[:, i, :]
            chunk_attention_mask = attention_mask[:, i, :]
            chunk_token_type_ids = token_type_ids[:, i, :]

            outputs = self.plm(input_ids=chunk_input_ids, token_type_ids=chunk_token_type_ids, attention_mask=chunk_attention_mask)
            chunk_cls_output = outputs.last_hidden_state[:, 0, :]
            cls_output += chunk_cls_output
        cls_output /= num_chunks

        prediction = self.mlp(cls_output)
        return prediction 


import torch
from torch import nn
from transformers import AutoModel
import torch.nn.functional as F

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class chunkplm(nn.Module):
    def __init__(self):
        super(chunkplm, self).__init__()
        self.plm = AutoModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(p=0.1)
        self.lstm = nn.LSTM(self.plm.config.hidden_size, self.plm.config.hidden_size, batch_first=True)
        self.fc = nn.Linear(self.plm.config.hidden_size, 4) 
        self.fc.apply(init_weights)

        self.w_omega = nn.Parameter(torch.Tensor(self.plm.config.hidden_size, self.plm.config.hidden_size))
        self.b_omega = nn.Parameter(torch.Tensor(1, self.plm.config.hidden_size))
        self.u_omega = nn.Parameter(torch.Tensor(self.plm.config.hidden_size, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)
        nn.init.uniform_(self.b_omega, -0.1, 0.1)

    def forward(self, input_ids, attention_mask, token_type_ids, device='cpu', bert_batch_size=1):
        batch_size, num_chunks, seq_length = input_ids.size()
        bert_output = torch.zeros((batch_size, num_chunks, self.plm.config.hidden_size), device=device)
        for chunk_idx in range(num_chunks):
            chunk_input_ids = input_ids[:, chunk_idx, :]
            chunk_attention_mask = attention_mask[:, chunk_idx, :]
            chunk_token_type_ids = token_type_ids[:, chunk_idx, :]

            plm_output = self.plm(input_ids=chunk_input_ids, attention_mask=chunk_attention_mask, token_type_ids=chunk_token_type_ids).last_hidden_state
            bert_output[:, chunk_idx, :] = plm_output[:, 0, :] 

        bert_output = bert_output.view(batch_size, num_chunks, -1)
        lstm_output, _ = self.lstm(bert_output)
        attention_w = torch.tanh(torch.matmul(lstm_output, self.w_omega) + self.b_omega)
        attention_score = F.softmax(torch.matmul(attention_w, self.u_omega), dim=1)
        attention_output = torch.sum(lstm_output * attention_score, dim=1)

        prediction = self.fc(attention_output)
        return prediction
