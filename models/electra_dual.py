import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ElectraModel, BertModel
from models.poolings import *

class ResponseAttention(nn.Module):
    def __init__(self, hidden_size):
        super(ResponseAttention, self).__init__()
        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(hidden_size, hidden_size)
        self.W_v = nn.Linear(hidden_size, hidden_size)
        self.scale = torch.sqrt(torch.tensor(hidden_size, dtype=torch.float))

    def forward(self, x):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        weights = F.softmax(scores, dim=-1)
        return torch.bmm(weights, V)

class ResponsePromptAttention(nn.Module):
    def __init__(self, hidden_size):
        super(ResponsePromptAttention, self).__init__()
        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(hidden_size, hidden_size)
        self.W_v = nn.Linear(hidden_size, hidden_size)

    def forward(self, response, prompt):
        z = self.W_q(response)
        p = self.W_k(prompt)
        v = self.W_v(prompt)
        
        scores = torch.bmm(z, p.transpose(1, 2)) / torch.sqrt(torch.tensor(response.size(-1), dtype=torch.float))
        weights = F.softmax(scores, dim=-1)
        return torch.bmm(weights, v)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = BertModel.from_pretrained('bert-base-cased')
        self.hidden_size = self.model.config.hidden_size
        self.response_attention = ResponseAttention(self.hidden_size)
        self.response_prompt_attention = ResponsePromptAttention(self.hidden_size)
        self.regression = nn.Linear(self.hidden_size * 2, 4)
        self.dropout = nn.Dropout(0.2)

    def forward(self, essay_input_ids, essay_attention_mask, essay_token_type_ids, topic_input_ids, topic_attention_mask, topic_token_type_ids):
        essay_emb = self.model(input_ids=essay_input_ids, attention_mask=essay_attention_mask, token_type_ids=essay_token_type_ids).last_hidden_state 
        prompt_emb = self.model(input_ids=topic_input_ids, attention_mask=topic_attention_mask, token_type_ids=topic_token_type_ids).last_hidden_state
        response_vector = self.response_attention(essay_emb)
        prompt_attention_vector = self.response_prompt_attention(response_vector[:, 0, :].unsqueeze(1), prompt_emb)[:, 0, :]
        score = self.regression(self.dropout(torch.concat([prompt_attention_vector, response_vector[:,0,:]], dim=-1)))
        return score

class EssayScoringModel(nn.Module):
    def __init__(self, hidden_size=256):
        super(EssayScoringModel, self).__init__()
        self.model = ElectraModel.from_pretrained('bert-base-cased')
        self.response_attention = ResponseAttention(hidden_size)
        self.response_prompt_attention = ResponsePromptAttention(hidden_size)
        self.regression = nn.Linear(hidden_size * 2, 4)
        self.dropout = nn.Dropout(0.2)
    def forward(self, essay_input_ids, essay_attention_mask, essay_token_type_ids, topic_input_ids, topic_attention_mask, topic_token_type_ids):
        essay_emb = self.model(input_ids=essay_input_ids, attention_mask=essay_attention_mask, token_type_ids=essay_token_type_ids)[0][:, 0, :]
        prompt_emb = self.model(input_ids=topic_input_ids, attention_mask=topic_attention_mask, token_type_ids=topic_token_type_ids)[0][:, 0, :]
        response_vector = self.response_attention(essay_emb, essay_emb)[:, 0, :] 
        prompt_attention_vector = self.response_prompt_attention(response_vector.unsqueeze(1), prompt_emb)[:, 0, :]
        combined_vector = torch.cat([response_vector, prompt_attention_vector], dim=1)
        score = self.regression(self.dropout(combined_vector))
        return score