from models.poolings import *
import torch
from transformers import AutoModel
import torch.nn as nn

class ELECTRA(torch.nn.Module):
    def __init__(self):
        super(ELECTRA, self).__init__()
        self.electra = AutoModel.from_pretrained('google/electra-small-discriminator')
        self.dropout = nn.Dropout(0.2)
        self.pooler = MeanPooling()
        self.out = nn.Linear(self.electra.config.hidden_size, 4)  # Ensure hidden_size matches

    def forward(self, input_ids, attention_mask):
        outputs = self.electra(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = self.pooler(last_hidden_state, attention_mask)
        dropout_output = self.dropout(pooled_output)
        outputs = self.out(dropout_output)
        return outputs