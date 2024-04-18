import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class mainplm(nn.Module):
    def __init__(self, args):
        super(mainplm, self).__init__()
        self.args = args
        self.plm_batch_size = 1
        self.plm = AutoModel.from_pretrained('bert-base-uncased')

        for param in self.plm.embeddings.parameters():
            param.requires_grad = False
        for i in range(11):
            for param in self.plm.encoder.layer[i].parameters():
                param.requires_grad = False

        self.mlp = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(self.plm.config.hidden_size, 4)
        )
        self.mlp.apply(init_weights)

    def forward(self, document_batch: torch.Tensor, device='cpu'):
        all_plm_output = self.plm(
            document_batch[:, 0:self.plm_batch_size, 0].view(-1, document_batch.size(-1)),  # Flatten the input
            token_type_ids=document_batch[:, 0:self.plm_batch_size, 1].view(-1, document_batch.size(-1)),
            attention_mask=document_batch[:, 0:self.plm_batch_size, 2].view(-1, document_batch.size(-1))
        )
        
        # Directly extract the relevant hidden state instead of using a loop
        plm_output = all_plm_output.last_hidden_state[:, 0, :].view(document_batch.shape[0], -1)
        
        prediction = self.mlp(plm_output)
        return prediction

class chunkplm(nn.Module):
    def __init__(self, args):
        super(chunkplm, self).__init__()
        self.args = args
        self.plm = AutoModel.from_pretrained('bert-base-uncased')

        for param in self.plm.embeddings.parameters():
            param.requires_grad = False
        for i in range(12):
            for param in self.plm.encoder.layer[i].parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(p=0.1)
        self.lstm = nn.LSTM(self.plm.config.hidden_size, self.plm.config.hidden_size)
        self.mlp = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(self.plm.config.hidden_size, 1)
        )
        self.w_omega = nn.Parameter(torch.Tensor(self.plm.config.hidden_size, self.plm.config.hidden_size))
        self.b_omega = nn.Parameter(torch.Tensor(1, self.plm.config.hidden_size))
        self.u_omega = nn.Parameter(torch.Tensor(self.plm.config.hidden_size, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)
        nn.init.uniform_(self.b_omega, -0.1, 0.1)
        self.mlp.apply(init_weights)

    def forward(self, document_batch: torch.Tensor, device='cpu', plm_batch_size=0):
        # Perform PLM operations on all chunks at once
        plm_output = self.plm(
            document_batch.view(-1, document_batch.size(-1)),  # Flatten the input
            attention_mask=(document_batch != self.tokenizer.pad_token_id).view(-1, document_batch.size(-1))
        )[1].view(document_batch.shape[0], plm_batch_size, -1)
        
        # Use built-in LSTM, no need for manual looping
        output, _ = self.lstm(plm_output)
        
        # Attention calculations can be done on batch tensors directly
        attention_w = torch.tanh(torch.matmul(output, self.w_omega) + self.b_omega)
        attention_score = F.softmax(attention_w.matmul(self.u_omega), dim=1)
        attention_hidden = (output * attention_score).sum(dim=1)
        
        prediction = self.mlp(attention_hidden)
        return prediction

import torch
from torch import nn, optim
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

class AESmodel(nn.Module):
    def __init__(self, train_data, val_data, test_data, foldname, args=None):
        super(AESmodel, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.prompt = int(self.args['prompt'])
        self.chunk_sizes = [int(size) for size in self.args['chunk_sizes'].split('_') if size != "0"]
        self.bert_batch_sizes = [int(asap_essay_lengths[self.prompt] / chunk_size) + 1 for chunk_size in self.chunk_sizes]
        
        self.mainplm = mainplm(self.args)
        self.chunkplm = chunkplm(self.args)
        
        # ... Other initializations ...
        
        self.optim = optim.Adam([
            {'params': self.mainplm.parameters(), 'lr': self.args['lr_0']},
            {'params': self.chunkplm.parameters(), 'lr': self.args['lr_1']}
        ])
        
        # ... Data loading and preparation ...
        
        self.train_data = DataLoader(train_data, batch_size=self.args['batch_size'], shuffle=True)
        self.val_data = DataLoader(val_data, batch_size=self.args['batch_size'], shuffle=False)
        self.test_data = DataLoader(test_data, batch_size=self.args['batch_size'], shuffle=False)
        
        # ... Rest of the initialization ...

    def forward(self, inputs):
        # Assuming inputs are preprocessed properly
        mainplm_output = self.mainplm(inputs)
        chunkplm_output = self.chunkplm(inputs)
        # Combine outputs from both models
        output = mainplm_output + chunkplm_output
        return output

    def train_model(self, epochs):
        # Implement training loop using DataLoader
        # ...
