from torch.nn import LayerNorm
import math
import torch.nn.functional as F
from transformers import ElectraModel
import torch.nn as nn
import torch
from models.poolings import *

class CrossAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.scale = math.sqrt(hidden_size)
        self.norm = LayerNorm(hidden_size)

    def forward(self, queries, keys, values, mask=None):
        Q = self.query(queries)
        K = self.key(keys)
        V = self.value(values)

        attention_scores = torch.einsum('bik,bjk->bij', Q, K) / self.scale
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        attention_probs = F.softmax(attention_scores, dim=-1)
        context = torch.matmul(attention_probs, V)
        context = self.norm(context)  # Apply LayerNorm to the output context
        return context


class CustomELECTRA(nn.Module):
    def __init__(self, hidden_size=256):
        super(CustomELECTRA, self).__init__()
        self.electra = ElectraModel.from_pretrained('google/electra-small-discriminator')
        self.cross_attention = CrossAttention(hidden_size)
        self.pooler = MeanPooling()
        self.dropout = nn.Dropout(0.2)
        # Output heads for each criterion
        self.task_response_head = nn.Linear(hidden_size, 1)
        self.coherence_head = nn.Linear(hidden_size, 1)
        self.lexical_resource_head = nn.Linear(hidden_size, 1)
        self.grammatical_range_head = nn.Linear(hidden_size, 1)

    def forward(self, essay_input_ids, essay_attention_mask, essay_token_type_ids, topic_input_ids, topic_attention_mask, topic_token_type_ids):
        essay_outputs = self.electra(input_ids=essay_input_ids, attention_mask=essay_attention_mask, token_type_ids=essay_token_type_ids).last_hidden_state
        topic_outputs = self.electra(input_ids=topic_input_ids, attention_mask=topic_attention_mask, token_type_ids=topic_token_type_ids).last_hidden_state

        task_response = self.cross_attention(essay_outputs, topic_outputs, topic_outputs)
        pooled_task_response = self.pooler(task_response, essay_attention_mask)
        pooled_essay = self.pooler(essay_outputs, essay_attention_mask)
        dropout_essay = self.dropout(pooled_essay)
        dropout_task_response = self.dropout(pooled_task_response)
        task_response_score = self.task_response_head(dropout_task_response)
        coherence_score = self.coherence_head(dropout_essay)
        lexical_resource_score = self.lexical_resource_head(dropout_essay)
        grammatical_range_score = self.grammatical_range_head(dropout_essay)
        
        return torch.cat([task_response_score, coherence_score, lexical_resource_score, grammatical_range_score], dim=1)