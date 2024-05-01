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
        # Ensure queries, keys, values are 3D tensors; [batch, seq_len, features]
        Q = self.query(queries).unsqueeze(1)  # [batch, 1, features]
        K = self.key(keys).unsqueeze(2)       # [batch, features, 1]
        V = self.value(values)                # [batch, features]
        # Batch matrix multiplication, broadcasting over the middle dimension
        attention_scores = torch.bmm(Q, K) / self.scale  # [batch, 1, 1]
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        attention_probs = F.softmax(attention_scores, dim=-1)  # [batch, 1, 1]
        context = torch.bmm(attention_probs, V.unsqueeze(1)).squeeze(1)  # [batch, features]
        context = self.norm(context)
        return context


class ResponsePromptAttention(nn.Module):
    def __init__(self, hidden_size):
        super(ResponsePromptAttention, self).__init__()
        self.scale = math.sqrt(hidden_size)
        self.W_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_v = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, responses, prompts):
        Q = self.W_q(responses)  # [batch_size, hidden_size]
        K = self.W_k(prompts)    # [batch_size, hidden_size]
        V = self.W_v(prompts)    # [batch_size, hidden_size]
        print(Q.shape, K.shape, V.shape)
        # Correct reshaping for batch matrix multiplication
        Q = Q.unsqueeze(1)  # Reshape Q to [batch_size, 1, hidden_size]
        K = K.unsqueeze(2)  # Reshape K to [batch_size, hidden_size, 1]
        print(Q.shape, K.shape)
        # Batch matrix multiplication
        attention_logits = torch.bmm(Q, K) / self.scale
        attention_logits = attention_logits.squeeze(2)  # Reduce to [batch_size, 1]

        attention_weights = F.softmax(attention_logits, dim=1)  # Softmax over the last dimension

        # Batch matrix multiplication for computing the weighted sum of V
        attention_output = torch.bmm(attention_weights, V.unsqueeze(1)).squeeze(1)  # Reduce to [batch_size, hidden_size]

        return attention_output



class CustomELECTRA(nn.Module):
    def __init__(self, hidden_size=256, hidden_dropout_prob=0.2):
        super(CustomELECTRA, self).__init__()
        self.electra = ElectraModel.from_pretrained('google/electra-small-discriminator')
        self.response_prompt_attention = ResponsePromptAttention(hidden_size)
        self.pooler = MeanPooling()  # Assume this can handle a concatenated mask
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.out = nn.Linear(2 * hidden_size, 4)  # Adjusted for concatenated output

    def forward(self, essay_input_ids, essay_attention_mask, essay_token_type_ids, topic_input_ids, topic_attention_mask, topic_token_type_ids):
        # Get outputs from ELECTRA
        essay_outputs = self.electra(input_ids=essay_input_ids, attention_mask=essay_attention_mask, token_type_ids=essay_token_type_ids).last_hidden_state
        topic_outputs = self.electra(input_ids=topic_input_ids, attention_mask=topic_attention_mask, token_type_ids=topic_token_type_ids).last_hidden_state
        
        # Generate attention response
        attention_response = self.response_prompt_attention(essay_outputs, topic_outputs)

        # Concatenate outputs and attention responses
        concatenated_output = torch.cat([essay_outputs, attention_response], dim=-1)

        # Concatenate attention masks to match the concatenated outputs
        # Assuming essay and topic outputs are the same length
        concatenated_mask = torch.cat([essay_attention_mask, topic_attention_mask], dim=1)

        # Pool the concatenated outputs using the concatenated mask
        pooled_output = self.pooler(concatenated_output, concatenated_mask)

        # Apply dropout to the pooled output
        dropout_output = self.dropout(pooled_output)
        
        # Output layer
        return self.out(dropout_output)
