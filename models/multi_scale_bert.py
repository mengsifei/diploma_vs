import torch
from torch import nn
from transformers import AutoModel


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class mainplm(nn.Module):
    def __init__(self):
        super(mainplm, self).__init__()
        self.plm = AutoModel.from_pretrained('bert-base-cased')

        # Freeze certain layers to reduce computation and stabilize training
        for param in self.plm.embeddings.parameters():
            param.requires_grad = False
        for i in range(11):  # Only the last transformer layer is unfrozen
            for param in self.plm.encoder.layer[i].parameters():
                param.requires_grad = False

        self.mlp = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(self.plm.config.hidden_size, 4)  # Assuming 4 different outputs or rubrics
        )
        self.mlp.apply(init_weights)

    def forward(self, input_ids, attention_mask, token_type_ids):
        batch_size, num_chunks, seq_length = input_ids.size()
        cls_output = torch.zeros((batch_size, self.plm.config.hidden_size), device=input_ids.device)

        # Process each chunk independently
        for i in range(num_chunks):
            chunk_input_ids = input_ids[:, i, :]
            chunk_attention_mask = attention_mask[:, i, :]
            chunk_token_type_ids = token_type_ids[:, i, :]

            outputs = self.plm(input_ids=chunk_input_ids, token_type_ids=chunk_token_type_ids, attention_mask=chunk_attention_mask)
            chunk_cls_output = outputs.last_hidden_state[:, 0, :]  # Take the output of the [CLS] token
            cls_output += chunk_cls_output  # Aggregate outputs by summation

        # Average the CLS outputs over the number of chunks to get the final document representation
        cls_output /= num_chunks

        prediction = self.mlp(cls_output)
        return prediction  # [batch_size, 4]


import torch
from torch import nn
from transformers import AutoModel
import torch.nn.functional as F

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class chunkplm(nn.Module):
    def __init__(self):
        super(chunkplm, self).__init__()
        self.plm = AutoModel.from_pretrained('bert-base-cased')

        # Freezing embeddings and transformer layers
        for param in self.plm.embeddings.parameters():
            param.requires_grad = False
        for i in range(12):  # Freeze all transformer layers
            for param in self.plm.encoder.layer[i].parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(p=0.1)
        self.lstm = nn.LSTM(self.plm.config.hidden_size, self.plm.config.hidden_size, batch_first=True)
        self.fc = nn.Linear(self.plm.config.hidden_size, 4)  # Assuming output dimension matches the task requirement
        self.fc.apply(init_weights)

        # Attention mechanism parameters
        self.w_omega = nn.Parameter(torch.Tensor(self.plm.config.hidden_size, self.plm.config.hidden_size))
        self.b_omega = nn.Parameter(torch.Tensor(1, self.plm.config.hidden_size))
        self.u_omega = nn.Parameter(torch.Tensor(self.plm.config.hidden_size, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)
        nn.init.uniform_(self.b_omega, -0.1, 0.1)

    def forward(self, input_ids, attention_mask, token_type_ids, device='cpu', bert_batch_size=1):
        # Assuming input_ids is [batch_size, num_chunks, seq_length]
        batch_size, num_chunks, seq_length = input_ids.size()
        bert_output = torch.zeros((batch_size, num_chunks, self.plm.config.hidden_size), device=device)

        # Process each chunk independently
        for chunk_idx in range(num_chunks):
            chunk_input_ids = input_ids[:, chunk_idx, :]
            chunk_attention_mask = attention_mask[:, chunk_idx, :]
            chunk_token_type_ids = token_type_ids[:, chunk_idx, :]

            plm_output = self.plm(input_ids=chunk_input_ids, attention_mask=chunk_attention_mask, token_type_ids=chunk_token_type_ids).last_hidden_state
            bert_output[:, chunk_idx, :] = plm_output[:, 0, :]  # Take the [CLS] token's representation

        # Apply LSTM across the chunks dimension
        bert_output = bert_output.view(batch_size, num_chunks, -1)  # Ensure LSTM input shape is correct
        lstm_output, _ = self.lstm(bert_output)

        # Attention calculation over the sequence of chunks
        attention_w = torch.tanh(torch.matmul(lstm_output, self.w_omega) + self.b_omega)
        attention_score = F.softmax(torch.matmul(attention_w, self.u_omega), dim=1)
        attention_output = torch.sum(lstm_output * attention_score, dim=1)

        prediction = self.fc(attention_output)
        return prediction
