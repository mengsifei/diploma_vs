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
    def __init__(self):
        super(mainplm, self).__init__()
        self.plm_batch_size = 1
        self.plm = AutoModel.from_pretrained('google/electra-small-discriminator')

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

    def forward(self, document_batch: torch.Tensor):
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
    def __init__(self):
        super(chunkplm, self).__init__()
        self.plm = AutoModel.from_pretrained('google/electra-small-discriminator')

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

    def forward(self, document_batch: torch.Tensor, plm_batch_size=0):
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