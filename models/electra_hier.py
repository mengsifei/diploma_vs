from transformers import ElectraModel, BertModel, DebertaModel, GPT2Model, AutoModel
import torch.nn as nn
import torch
from models.poolings import *

class BaseModel(nn.Module):
    def __init__(self, model_name='electra', hidden_dropout_prob=0.2, num_labels=4):
        super(BaseModel, self).__init__()
        self.model_name = model_name
        self.model = None
        self.get_model()
        self.pooler = MeanPooling()
        # self.pooler = SoftAttention(self.model.config.hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
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
    def forward(self, input_ids, attention_mask, token_type_ids):
        batch_size, num_chunks, seq_length = input_ids.size()
        all_outputs = []
        for i in range(num_chunks):
            chunk_input_ids = input_ids[:, i, :]
            chunk_attention_mask = attention_mask[:, i, :]
            chunk_token_type_ids = token_type_ids[:, i, :]
            outputs = self.model(input_ids=chunk_input_ids, 
                                attention_mask=chunk_attention_mask, 
                                token_type_ids=chunk_token_type_ids)
            all_outputs.append(outputs.last_hidden_state)
        # Stack along a new dimension to keep chunks separate
        all_outputs = torch.stack(all_outputs, dim=1)  # Shape: (batch_size, num_chunks, seq_length, hidden_size)
        pooled_output = self.pooler(all_outputs, attention_mask)  # Make sure attention_mask is correct
        dropped_output = self.dropout(pooled_output)
        prediction = self.out(dropped_output)
        return prediction


from transformers import ElectraModel, BertModel, DebertaModel, GPT2Model, AutoModel
import torch.nn as nn
import torch
from models.poolings import *

class BaseModel(nn.Module):
    def __init__(self, model_name='electra', hidden_dropout_prob=0.2, num_labels=4):
        super(BaseModel, self).__init__()
        self.model_name = model_name
        self.model = ElectraModel.from_pretrained('google/electra-small-discriminator', output_hidden_states=True)
        self.pooler = MeanPooling()
        self.hidden_size=self.model.config.hidden_size
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.out = nn.Linear(self.model.config.hidden_size, num_labels)
    def forward(self, input_ids, attention_mask, token_type_ids):
        batch_size, num_chunks, seq_length = input_ids.size()
        all_outputs = []
        for i in range(num_chunks):
            chunk_input_ids = input_ids[:, i, :]
            chunk_attention_mask = attention_mask[:, i, :]
            chunk_token_type_ids = token_type_ids[:, i, :]
            outputs = self.model(input_ids=chunk_input_ids, 
                                attention_mask=chunk_attention_mask, 
                                token_type_ids=chunk_token_type_ids)
            all_outputs.append(outputs.last_hidden_state)
        # Stack along a new dimension to keep chunks separate
        all_outputs = torch.stack(all_outputs, dim=1)  # Shape: (batch_size, num_chunks, seq_length, hidden_size)
        all_outputs = all_outputs.view(batch_size, -1, self.hidden_size)
        attention_mask = attention_mask.view(batch_size, -1)
        # print(all_outputs.shape, attention_mask.shape)
        pooled_output = self.pooler(all_outputs, attention_mask)  # Make sure attention_mask is correct
        # pooled_output = all_outputs[:, 0, :]
        dropped_output = self.dropout(pooled_output)
        prediction = self.out(dropped_output)
        return prediction
