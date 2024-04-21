from transformers import ElectraModel, BertModel, DebertaModel, GPT2Model
import torch.nn as nn
import torch
from models.poolings import *

class BaseModel(nn.Module):
    def __init__(self, model_name='electra', hidden_dropout_prob=0.2, num_labels=4):
        super(BaseModel, self).__init__()
        self.model_name = model_name
        self.model = None
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.out = nn.Linear(self.model.config.hidden_size, num_labels)
        self.pooler = MeanPooling()
    def get_model(self):
        if self.model_name == 'electra':
            self.model = ElectraModel.from_pretrained('google/electra-discriminator-small')
        elif self.model_name == 'bert':
            self.model = BertModel.from_pretrained('bert-base-cased')
        elif self.model_name == 'deberta':
            self.model = DebertaModel.from_pretrained('microsoft/deberta-v3-small')
        elif self.model_name == 'gpt':
            self.model = GPT2Model.from_pretrained('gpt2')
    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = self.pooler(last_hidden_state, attention_mask)
        dropout_output = self.dropout(pooled_output)
        outputs = self.out(dropout_output)
        return dropout_output

