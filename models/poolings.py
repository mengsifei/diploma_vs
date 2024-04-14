import torch.nn as nn
import torch

class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionPooling, self).__init__()
        self.hidden_dim = hidden_dim
        self.w = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.v = nn.Linear(self.hidden_dim, 1)

    def forward(self, h):
        w = torch.tanh(self.w(h))
        weight = self.v(w)
        # weight = weight.squeeze(dim=-1)
        weight = torch.softmax(weight, dim=1)
        # weight = weight.unsqueeze(dim=-1)
        weight_broadcasted = weight.repeat(1, h.size(1))
        # print(weight_broadcasted.shape)
        # print(h.shape)
        out = torch.mul(h, weight_broadcasted)
        print(out.shape)
        out = torch.sum(out, dim=0)
        print(out.shape)
        return out