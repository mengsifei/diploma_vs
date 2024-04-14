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


class TraitAttention(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(TraitAttention, self).__init__()
        self.attention_score = nn.Linear(input_dim, attention_dim)
        self.attention_weight = nn.Linear(attention_dim, 1, bias=False)
    
    def forward(self, c, A):
        # Assuming c is [batch_size, feature_size]
        # and A is [batch_size, sequence_length, feature_size]
        # where feature_size corresponds to input_dim
        e = torch.tanh(self.attention_score(c))  # [batch_size, attention_dim]
        scores = self.attention_weight(e)  # [batch_size, 1]
        
        # Repeat scores across the sequence_length dimension
        scores = scores.unsqueeze(1).repeat(1, A.size(1), 1)  # [batch_size, sequence_length, 1]
        
        # Apply softmax to get the attention weights
        attention_weights = F.softmax(scores, dim=1)  # [batch_size, sequence_length, 1]
        
        # Apply attention weights to A
        p = torch.bmm(attention_weights.transpose(1, 2), A)  # [batch_size, 1, feature_size]
        p = p.squeeze(1)  # Remove the singleton dimension
        
        # Concatenate c and p to get the final representation g
        g = torch.cat((c, p), dim=1)  # [batch_size, feature_size * 2]
        return g

    

# class ELECTRA(nn.Module):
#     def __init__(self, hidden_size=256, num_features=4, drop_rate=0.1):
#         super(ELECTRA, self).__init__()
#         self.electra = ElectraModel.from_pretrained('google/electra-small-discriminator')
#         # self.soft_attention = AttentionPooling(hidden_dim=hidden_size)
#         self.meanpooling = MeanPooling()
#         self.drop_rate = drop_rate
#         self.trait_attention = TraitAttention(hidden_size + num_features, hidden_size)

#         # Task-specific heads after trait-attention mechanism
#         self.task_response_head = nn.Linear(hidden_size + num_features, 1)
#         self.coherence_head = nn.Linear(hidden_size + num_features, 1)
#         self.lexical_resource_head = nn.Linear(hidden_size + num_features, 1)
#         self.grammatical_range_head = nn.Linear(hidden_size + num_features, 1)
        
#     def forward(self, input_ids, attention_mask, token_type_ids, features):
#         outputs = self.electra(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
#         sequence_output = outputs.last_hidden_state
#         pooled_output = self.meanpooling(sequence_output, attention_mask)
#         combined_features = torch.cat((pooled_output, features), dim=1)
#         trait_attended_features = self.trait_attention(combined_features, combined_features)
        
#         task_response_output = self.task_response_head((trait_attended_features))
#         coherence_output = self.coherence_head((trait_attended_features))
#         lexical_resource_output = self.lexical_resource_head((trait_attended_features))
#         grammatical_range_output = self.grammatical_range_head((trait_attended_features))
        
#         # Concatenate the outputs for each task
#         final_output = torch.cat((task_response_output,
#                                   coherence_output,
#                                   lexical_resource_output,
#                                   grammatical_range_output), dim=-1)
        
#         return final_output


class ELECTRA(nn.Module):
    def __init__(self, hidden_size=256, num_features=4, drop_rate=0.1):
        super(ELECTRA, self).__init__()
        self.electra = ElectraModel.from_pretrained('google/electra-small-discriminator')
        self.meanpooling = MeanPooling()
        self.drop_rate = drop_rate
        self.trait_attention = AttentionPooling(hidden_size + num_features)  # Attention based on Electra's hidden size

        # Task-specific heads after trait-attention mechanism
        self.task_response_head = nn.Linear(hidden_size + num_features, 1)
        self.coherence_head = nn.Linear(hidden_size + num_features, 1)
        self.lexical_resource_head = nn.Linear(hidden_size + num_features, 1)
        self.grammatical_range_head = nn.Linear(hidden_size + num_features, 1)
        
    def forward(self, input_ids, attention_mask, token_type_ids, features):
        # Get outputs from Electra
        outputs = self.electra(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs.last_hidden_state
        
        # Apply mean pooling to sequence output
        pooled_output = self.meanpooling(sequence_output, attention_mask)
        combined_features = torch.cat([pooled_output, features], dim=1)
        trait_attended_features = self.trait_attention(combined_features)

        
        # Pass through task-specific heads
        task_response_output = self.task_response_head(trait_attended_features)
        coherence_output = self.coherence_head(trait_attended_features)
        lexical_resource_output = self.lexical_resource_head(trait_attended_features)
        grammatical_range_output = self.grammatical_range_head(trait_attended_features)
        
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