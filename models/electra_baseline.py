from models.poolings import *
import torch
from transformers import ElectraModel
import torch.nn as nn

# class ELECTRA(torch.nn.Module):
#     def __init__(self):
#         super(ELECTRA, self).__init__()
#         self.electra = ElectraModel.from_pretrained('google/electra-small-discriminator')
#         self.dropout = nn.Dropout(0.2)
#         self.pooler = MeanPooling()
#         self.out = nn.Linear(self.electra.config.hidden_size, 4)  # Ensure hidden_size matches
#     def resize_token_embeddings(self, new_num_tokens):
#         self.electra.resize_token_embeddings(new_num_tokens)
#     def forward(self, input_ids, attention_mask, token_type_ids):
#         outputs = self.electra(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
#         last_hidden_state = outputs.last_hidden_state
#         pooled_output = self.pooler(last_hidden_state, attention_mask)
#         dropout_output = self.dropout(pooled_output)
#         outputs = self.out(dropout_output)
#         return outputs


class ELECTRA(nn.Module):
    def __init__(self, hidden_size=256, bidirectional=True):
        super(ELECTRA, self).__init__()
        self.electra = ElectraModel.from_pretrained('google/electra-small-discriminator')
        # self.soft_attention = AttentionPooling(hidden_dim=hidden_size)
        self.meanpooling = MeanPooling()
        self.drop_rate = 0.1
        self.task_response_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(hidden_size, 1)
        )
        self.coherence_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(hidden_size, 1)
        )
        self.lexical_resource_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(hidden_size, 1)
        )
        self.grammatical_range_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.electra(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs.last_hidden_state

        
        # Apply attention to each set of LSTM features
        # task_response_attended = self.soft_attention(sequence_output)
        # coherence_attended = self.soft_attention(sequence_output)
        # lexical_resource_attended = self.soft_attention(sequence_output)
        # grammatical_range_attended = self.soft_attention(sequence_output)
        task_response_attended = self.meanpooling(sequence_output, attention_mask)
        coherence_attended = self.meanpooling(sequence_output, attention_mask)
        lexical_resource_attended = self.meanpooling(sequence_output, attention_mask)
        grammatical_range_attended = self.meanpooling(sequence_output, attention_mask)
        
        # Compute the outputs for each task
        task_response_output = self.task_response_head(task_response_attended)
        coherence_output = self.coherence_head(coherence_attended)
        lexical_resource_output = self.lexical_resource_head(lexical_resource_attended)
        grammatical_range_output = self.grammatical_range_head(grammatical_range_attended)
        
        # Concatenate the outputs for each task
        final_output = torch.cat((task_response_output,
                                  coherence_output,
                                  lexical_resource_output,
                                  grammatical_range_output), dim=-1)
        
        return final_output


from torch.nn import LayerNorm
import math
import torch.nn.functional as F
import torch.nn.init as init

class CrossAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.scale = math.sqrt(hidden_size)
        self.norm = LayerNorm(hidden_size)
        self.weight = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.Tensor(hidden_size))
        init.xavier_uniform_(self.weight)
        init.zeros_(self.bias)

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

    def forward(self, essay_input_ids, essay_attention_mask, topic_input_ids, topic_attention_mask):
        essay_outputs = self.electra(input_ids=essay_input_ids, attention_mask=essay_attention_mask).last_hidden_state
        topic_outputs = self.electra(input_ids=topic_input_ids, attention_mask=topic_attention_mask).last_hidden_state

        task_response = self.cross_attention(essay_outputs, topic_outputs, topic_outputs)
        task_response = self.dropout(task_response)
        pooled_task_response = self.pooler(task_response, essay_attention_mask)
        pooled_essay = self.pooler(essay_outputs, essay_attention_mask)
        
        task_response_score = self.task_response_head(pooled_task_response)
        coherence_score = self.coherence_head(pooled_essay)
        lexical_resource_score = self.lexical_resource_head(pooled_essay)
        grammatical_range_score = self.grammatical_range_head(pooled_essay)
        
        return torch.cat([task_response_score, coherence_score, lexical_resource_score, grammatical_range_score], dim=1)