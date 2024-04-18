from transformers import ElectraModel
import torch.nn as nn
import torch
from models.poolings import *
# Define the base ELECTRA model
class BaseELECTRA(nn.Module):
    def __init__(self, model_name='google/electra-small-discriminator', hidden_dropout_prob=0.2):
        super(BaseELECTRA, self).__init__()
        self.electra = ElectraModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.pooler = MeanPooling()

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.electra(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = self.pooler(last_hidden_state, attention_mask)
        dropout_output = self.dropout(pooled_output)
        return dropout_output

# Define the model for Task Response and Coherence and Cohesion
class ELECTRA_Model1(BaseELECTRA):
    def __init__(self, model_name='google/electra-small-discriminator', hidden_dropout_prob=0.2):
        super(ELECTRA_Model1, self).__init__(model_name, hidden_dropout_prob)
        self.out = nn.Linear(self.electra.config.hidden_size, 2)  # Output for 2 tasks

    def forward(self, input_ids, attention_mask, token_type_ids):
        pooled_output = super().forward(input_ids, attention_mask, token_type_ids)
        outputs = self.out(pooled_output)
        return outputs

# Define the model for Lexical Resource and Grammatical Range and Accuracy
class ELECTRA_Model2(BaseELECTRA):
    def __init__(self, model_name='google/electra-small-discriminator', hidden_dropout_prob=0.2):
        super(ELECTRA_Model2, self).__init__(model_name, hidden_dropout_prob)
        self.out = nn.Linear(self.electra.config.hidden_size, 2)  # Output for 2 tasks

    def forward(self, input_ids, attention_mask, token_type_ids):
        pooled_output = super().forward(input_ids, attention_mask, token_type_ids)
        outputs = self.out(pooled_output)
        return outputs



# class BERT(torch.nn.Module):
#     def __init__(self):
#         super(BERT, self).__init__()
#         self.model = AutoModel.from_pretrained('bert-base-cased')
#         self.dropout = nn.Dropout(0.2)
#         self.pooler = MeanPooling()
#         self.out = nn.Linear(self.model.config.hidden_size, 4)  # Ensure hidden_size matches
#     def resize_token_embeddings(self, new_num_tokens):
#         self.model.resize_token_embeddings(new_num_tokens)
#     def forward(self, input_ids, attention_mask, token_type_ids):
#         outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
#         last_hidden_state = outputs.last_hidden_state
#         pooled_output = self.pooler(last_hidden_state, attention_mask)
#         dropout_output = self.dropout(pooled_output)
#         outputs = self.out(dropout_output)
#         return outputs
 