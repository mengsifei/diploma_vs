from models.poolings import *
import torch
from transformers import ElectraModel
from transformers import AutoModel
import torch.nn as nn

class ELECTRA(torch.nn.Module):
    def __init__(self):
        super(ELECTRA, self).__init__()
        self.electra = ElectraModel.from_pretrained('google/electra-small-discriminator')
        self.dropout = nn.Dropout(0.2)
        self.pooler = MeanPooling()
        self.out = nn.Linear(self.electra.config.hidden_size, 4)
    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.electra(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = self.pooler(last_hidden_state, attention_mask)
        dropout_output = self.dropout(pooled_output)
        outputs = self.out(dropout_output)
        return outputs


class BERT(torch.nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.model = AutoModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(0.2)
        self.pooler = MeanPooling()
        self.out = nn.Linear(self.model.config.hidden_size, 4)  # Ensure hidden_size matches
    def resize_token_embeddings(self, new_num_tokens):
        self.model.resize_token_embeddings(new_num_tokens)
    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = self.pooler(last_hidden_state, attention_mask)
        dropout_output = self.dropout(pooled_output)
        outputs = self.out(dropout_output)
        return outputs
 