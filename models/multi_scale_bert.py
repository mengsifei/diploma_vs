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
        self.plm = AutoModel.from_pretrained('google/electra-small-discriminator')

        # Freeze certain layers to stabilize training and reduce computation
        for param in self.plm.embeddings.parameters():
            param.requires_grad = False
        for i in range(11):  # Only the last transformer layer is unfrozen
            for param in self.plm.encoder.layer[i].parameters():
                param.requires_grad = False

        self.mlp = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(self.plm.config.hidden_size, 4)
        )
        self.mlp.apply(init_weights)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.plm(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_output = outputs.last_hidden_state[:, 0, :]  # Extract the representation of the [CLS] token
        prediction = self.mlp(cls_output)
        return prediction  # [batch_size, 4]

class chunkplm(nn.Module):
    def __init__(self):
        super(chunkplm, self).__init__()
        self.plm = AutoModel.from_pretrained('google/electra-small-discriminator')

        # Freezing embeddings and all transformer layers
        for param in self.plm.embeddings.parameters():
            param.requires_grad = False
        for i in range(12):
            for param in self.plm.encoder.layer[i].parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(p=0.1)
        self.lstm = nn.LSTM(self.plm.config.hidden_size, self.plm.config.hidden_size, batch_first=True)
        self.fc = nn.Linear(self.plm.config.hidden_size, 1)  # Assuming output dimension is 1 per chunk
        self.fc.apply(init_weights)

        # Attention mechanism parameters
        self.w_omega = nn.Parameter(torch.Tensor(self.plm.config.hidden_size, self.plm.config.hidden_size))
        self.b_omega = nn.Parameter(torch.Tensor(1, self.plm.config.hidden_size))
        self.u_omega = nn.Parameter(torch.Tensor(self.plm.config.hidden_size, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)
        nn.init.uniform_(self.b_omega, -0.1, 0.1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        batch_size, seq_count, seq_len = input_ids.shape
        input_ids = input_ids.view(-1, seq_len)
        attention_mask = attention_mask.view(-1, seq_len)
        token_type_ids = token_type_ids.view(-1, seq_len)

        outputs = self.plm(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state
        outputs = outputs[:, 0, :]  # Using the output of the [CLS] token

        outputs = outputs.view(batch_size, seq_count, -1)  # Reshape to [batch_size, num_chunks, feature_dim]
        lstm_output, _ = self.lstm(outputs)  # Apply LSTM across the chunk dimension

        attention_w = torch.tanh(torch.matmul(lstm_output, self.w_omega) + self.b_omega)
        attention_score = torch.softmax(torch.matmul(attention_w, self.u_omega), dim=1)
        attention_output = torch.sum(lstm_output * attention_score, dim=1)  # Sum weighted sequences

        prediction = self.fc(attention_output)
        return prediction  # [batch_size, 1] - Assuming single score per document
