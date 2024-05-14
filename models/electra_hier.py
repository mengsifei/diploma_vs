from transformers import ElectraModel
import torch.nn as nn
import torch
from models.poolings import * 
import torch.nn.functional as F

class BaseModel(nn.Module):
    def __init__(self, model_name='electra', hidden_dropout_prob=0.2, num_labels=4, pooling_name='cls'):
        super(BaseModel, self).__init__()
        self.pooling_name = pooling_name
        self.model = ElectraModel.from_pretrained('google/electra-small-discriminator', output_hidden_states=True)
        self.hidden_size = self.model.config.hidden_size
        self.weighted_pooler = WeightedLayerPooling(12, layer_start=9, layer_weights=nn.Parameter(
            torch.tensor([1, 1, 2, 3], dtype=torch.float)))
        self.pooler = None
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.dropout2 = nn.Dropout(0.1)
        self.out = nn.Linear(self.hidden_size, num_labels)
        self.linear = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.relu = nn.ReLU()
        self.layernorm = nn.LayerNorm(self.hidden_size)
        self.get_pooler()

    def get_pooler(self):
        if self.pooling_name == "mean":
            self.pooler = MeanPooling()
        elif self.pooling_name == "max":
            self.pooler = MaxPooling()
        elif self.pooling_name == "attention":
            self.pooler = AttentionPooling(self.hidden_size)
        elif self.pooling_name == "cls":
            self.pooler = CLSPooling()

    def forward(self, input_ids, attention_mask, token_type_ids):
        batch_size, num_chunks, seq_length = input_ids.size()
        all_pooled_outputs_chunk = []
        all_last_hidden_states = []

        for i in range(num_chunks):
            chunk_input_ids = input_ids[:, i, :]
            chunk_attention_mask = attention_mask[:, i, :]
            chunk_token_type_ids = token_type_ids[:, i, :]

            if chunk_attention_mask.sum() == 0:
                all_pooled_outputs_chunk.append(torch.zeros(batch_size, self.hidden_size, device=input_ids.device))
                all_last_hidden_states.append(torch.zeros(batch_size, self.hidden_size, device=input_ids.device))
            else:
                outputs = self.model(input_ids=chunk_input_ids,
                                     attention_mask=chunk_attention_mask,
                                     token_type_ids=chunk_token_type_ids)
                
                # Collect last hidden state
                last_hidden_state = outputs.last_hidden_state[:, 0]
                all_last_hidden_states.append(last_hidden_state)
                
                # Weighted pooling and pooling
                outputs = torch.stack(outputs.hidden_states)
                weighted_pooling_embeddings = self.weighted_pooler(outputs)
                pooled_embedding = self.pooler(weighted_pooling_embeddings, chunk_attention_mask)
                all_pooled_outputs_chunk.append(pooled_embedding)
        
        # Concatenate pooled outputs and last hidden states
        all_pooled_outputs_chunk = torch.stack(all_pooled_outputs_chunk, dim=1)
        all_last_hidden_states = torch.stack(all_last_hidden_states, dim=1)

        # ResNet-style layer normalization and residual connection
        combined_output = torch.cat([all_pooled_outputs_chunk, all_last_hidden_states], dim=-1)
        linear_output = self.layernorm(self.relu(self.linear(combined_output)))
        pooled_output = linear_output[:, 0]

        # Dropout and final linear layer
        dropped_output = self.dropout(pooled_output)
        prediction = self.out(dropped_output)
        return prediction
