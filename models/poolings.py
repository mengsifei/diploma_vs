import torch.nn as nn
import torch
import torch.nn.functional as F


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
    def forward(self, hidden_states, mask=None):
        if mask is None:
            return torch.mean(hidden_states, dim=1)
        else:
            mask_expanded = mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = mask_expanded.sum(dim=1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask
            return mean_embeddings

class CLSPooling(nn.Module):
    def __init__(self):
        super(CLSPooling, self).__init__()
    def forward(self, hidden_states, mask=None):
        cls_embeddings = hidden_states[:, 0, :]
        return cls_embeddings

class WeightedLayerPooling(nn.Module):
    def __init__(self, num_hidden_layers, layer_start: int = 4, layer_weights = None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = layer_weights if layer_weights is not None \
            else nn.Parameter(
                torch.tensor([1] * (num_hidden_layers+1 - layer_start), dtype=torch.float)
            )

    def forward(self, all_hidden_states):
        all_layer_embedding = all_hidden_states[self.layer_start:, :, :, :]
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor*all_layer_embedding).sum(dim=0) / self.layer_weights.sum()
        return weighted_average


class MaxPooling(nn.Module):
    def __init__(self):
        super(MaxPooling, self).__init__()

    def forward(self, hidden_states, mask=None):
        if mask is None:
            return torch.max(hidden_states, dim=1).values
        else:
            masked_hidden_states = hidden_states.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
            return torch.max(masked_hidden_states, dim=1).values

class MeanMaxPooling(nn.Module):
    def __init__(self):
        super(MeanMaxPooling, self).__init__()
        self.mean_pooling = MeanPooling()
        self.max_pooling = MaxPooling() 

    def forward(self, last_hidden_state, attention_mask):
        mean_pooling_embeddings = self.mean_pooling(last_hidden_state, attention_mask)
        max_pooling_embeddings = self.max_pooling(last_hidden_state, attention_mask)
        return torch.cat((mean_pooling_embeddings, max_pooling_embeddings), dim=1)


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionPooling, self).__init__()
        self.hidden_dim = hidden_dim
        self.w = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.v = nn.Linear(self.hidden_dim, 1)

    def forward(self, h, attention_mask):
        transformed_h = torch.tanh(self.w(h))
        raw_weights = self.v(transformed_h).squeeze(-1)
        weights = F.softmax(raw_weights, dim=1)
        weighted_h = h * weights.unsqueeze(-1)
        out = torch.sum(weighted_h, dim=1)
        return out
