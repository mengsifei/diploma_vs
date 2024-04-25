from transformers import ElectraModel, BertModel, DebertaModel, GPT2Model, AutoModel
import torch.nn as nn
import torch
from models.poolings import *

class EnhancedOutputHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(EnhancedOutputHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim // 2)  # Reduce dimension
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(input_dim // 2, output_dim)  # Final output dimension

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class BaseModel(nn.Module):
    def __init__(self, model_name='electra', hidden_dropout_prob=0.2, num_labels=4):
        super(BaseModel, self).__init__()
        self.model_name = model_name
        self.model = None
        self.get_model()
        self.pooler = MeanPooling()
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.TaskResponse = EnhancedOutputHead(self.model.config.hidden_size, 1)
        self.CoherenceCohesion = EnhancedOutputHead(self.model.config.hidden_size, 1)
        self.LexicalResource = EnhancedOutputHead(self.model.config.hidden_size, 1)
        self.Grammar = EnhancedOutputHead(self.model.config.hidden_size, 1)
        self.out = nn.Linear(self.model.config.hidden_size, num_labels)
    def get_model(self):
        if self.model_name == 'electra':
            self.model = ElectraModel.from_pretrained('google/electra-small-discriminator')
        elif self.model_name == 'bert':
            self.model = BertModel.from_pretrained('bert-base-cased')
        elif self.model_name == 'gpt':
            self.model = GPT2Model.from_pretrained('gpt2')
        elif self.model_name == 'electra-base':
            self.model = ElectraModel.from_pretrained('google/electra-base-discriminator')
        elif self.model_name == 'simcsc':
            self.model = AutoModel.from_pretrained('princeton-nlp/sup-simcse-bert-base-uncased')
        elif self.model_name == 'xlnet':
            self.model = AutoModel.from_pretrained('xlnet-base-cased')
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = self.pooler(last_hidden_state, attention_mask)
        dropout_output = self.dropout(pooled_output)
        output1 = self.TaskResponse(dropout_output)
        output2 = self.CoherenceCohesion(dropout_output)
        output3 = self.LexicalResource(dropout_output)
        output4 = self.Grammar(dropout_output)
        
        final_outputs = torch.cat([output1, output2, output3, output4], dim=1)
        return final_outputs