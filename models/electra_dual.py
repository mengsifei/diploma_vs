import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ElectraModel

class ResponseAttention(nn.Module):
    def __init__(self, hidden_size):
        super(ResponseAttention, self).__init__()
        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(hidden_size, hidden_size)
        self.W_v = nn.Linear(hidden_size, hidden_size)
        self.scale = torch.sqrt(torch.tensor(hidden_size, dtype=torch.float))

    def forward(self, x, relative_pos):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Calculate attention scores with relative position embeddings
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale + torch.bmm(Q, self.W_k(relative_pos).transpose(1, 2)) / self.scale
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

class EssayScoringModel(nn.Module):
    def __init__(self, hidden_size=256):
        super(EssayScoringModel, self).__init__()
        self.model = ElectraModel.from_pretrained('google/electra-small-discriminator')
        self.response_attention = ResponseAttention(hidden_size)
        self.response_prompt_attention = ResponsePromptAttention(hidden_size)
        self.regression = nn.Linear(hidden_size * 2, 4)

    def forward(self, essay_input_ids, essay_attention_mask, essay_token_type_ids, topic_input_ids, topic_attention_mask, topic_token_type_ids):
        # Handle essay and prompts with model
        essay_emb = self.model(input_ids=essay_input_ids, attention_mask=essay_attention_mask, token_type_ids=essay_token_type_ids)[0]
        prompt_emb = self.model(input_ids=topic_input_ids, attention_mask=topic_attention_mask, token_type_ids=topic_token_type_ids)[0]
        
        # Apply self-attention and response-prompt attention
        response_vector = self.response_attention(essay_emb, essay_emb)[:, 0, :]  # Assume using the CLS token
        prompt_attention_vector = self.response_prompt_attention(response_vector.unsqueeze(1), prompt_emb)[:, 0, :]
        
        # Regression for scoring
        combined_vector = torch.cat([response_vector, prompt_attention_vector], dim=1)
        score = self.regression(combined_vector)
        return score
