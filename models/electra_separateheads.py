from transformers import ElectraModel
import torch.nn as nn
import torch
from models.poolings import *
from torch import nn

class BaseModel(nn.Module):
    def __init__(self, model_name='electra', hidden_dropout_prob=0.2, num_labels=1):
        super(BaseModel, self).__init__()
        self.model_name = model_name
        self.model = ElectraModel.from_pretrained('google/electra-small-discriminator')
        self.hidden_size = self.model.config.hidden_size
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.pooler = MeanPooling()
        self.TaskResponse = nn.Linear(self.model.config.hidden_size, 1)
        self.CoherenceCohesion = nn.Linear(self.model.config.hidden_size, 1)
        self.LexicalResource = nn.Linear(self.model.config.hidden_size, 1)
        self.Grammar = nn.Linear(self.model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state
        pooled = self.pooler(outputs, attention_mask)
        dropout_output = self.dropout(pooled)
        output1 = self.TaskResponse(dropout_output)
        output2 = self.CoherenceCohesion(dropout_output)
        output3 = self.LexicalResource(dropout_output)
        output4 = self.Grammar(dropout_output)
        final_outputs = torch.cat([output1, output2, output3, output4], dim=1)
        return final_outputs