import torch.nn as nn
import torch
import torch.nn.functional as F
import math

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
class MeanPoolingChunks(nn.Module):
    def __init__(self):
        super(MeanPoolingChunks, self).__init__()

    def forward(self, embeddings, attention_mask):
        # Embeddings shape: (batch_size, num_chunks, seq_length, hidden_size)
        # Attention_mask shape: (batch_size, num_chunks, seq_length)
        batch_size, num_chunks, seq_length, hidden_size = embeddings.size()
        expanded_mask = attention_mask.unsqueeze(-1).expand(-1, -1, -1, hidden_size).float()

        # Summing embeddings across the seq_length dimension
        sum_embeddings = torch.sum(embeddings * expanded_mask, dim=2)  # Shape: (batch_size, num_chunks, hidden_size)
        sum_mask = torch.sum(expanded_mask, dim=2)  # Shape: (batch_size, num_chunks, hidden_size)

        # Avoid division by zero
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        
        # Mean over sequence length
        mean_embeddings = sum_embeddings / sum_mask  # Shape: (batch_size, num_chunks, hidden_size)

        # Average over chunks or other pooling method
        # For simplicity, we take mean across chunks
        final_mean = torch.mean(mean_embeddings, dim=1)  # Shape: (batch_size, hidden_size)

        return final_mean


# class AttentionPooling(nn.Module):
#     def __init__(self, hidden_dim):
#         super(AttentionPooling, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.w = nn.Linear(self.hidden_dim, self.hidden_dim)
#         self.v = nn.Linear(self.hidden_dim, 1)

#     def forward(self, h):
#         w = torch.tanh(self.w(h))
#         weight = self.v(w)
#         # weight = weight.squeeze(dim=-1)
#         weight = torch.softmax(weight, dim=1)
#         # weight = weight.unsqueeze(dim=-1)
#         weight_broadcasted = weight.repeat(1, h.size(1))
#         # print(weight_broadcasted.shape)
#         # print(h.shape)
#         out = torch.mul(h, weight_broadcasted)
#         print(out.shape)
#         out = torch.sum(out, dim=0)
#         print(out.shape)
#         return out

class SelfAttention(nn.Module):
    def __init__(self, feature_dim, attention_heads=1):
        super(SelfAttention, self).__init__()
        self.feature_dim = feature_dim
        self.attention_heads = attention_heads
        self.key = nn.Linear(feature_dim, feature_dim * attention_heads)
        self.query = nn.Linear(feature_dim, feature_dim * attention_heads)
        self.value = nn.Linear(feature_dim, feature_dim * attention_heads)

    def forward(self, x):
        batch_size = x.size(0)

        # Calculate queries, keys, values
        keys = self.key(x).view(batch_size, -1, self.attention_heads, self.feature_dim).transpose(1, 2)
        queries = self.query(x).view(batch_size, -1, self.attention_heads, self.feature_dim).transpose(1, 2)
        values = self.value(x).view(batch_size, -1, self.attention_heads, self.feature_dim).transpose(1, 2)

        # Attention mechanism
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.feature_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply attention weights to values
        weighted_sum = torch.matmul(attention_weights, values)
        weighted_sum = weighted_sum.transpose(1, 2).contiguous().view(batch_size, -1, self.feature_dim * self.attention_heads)

        # Optionally, project back to the original feature dimension (useful if attention_heads > 1)
        if self.attention_heads > 1:
            weighted_sum = weighted_sum.view(batch_size, -1, self.feature_dim * self.attention_heads)
            weighted_sum = weighted_sum.sum(dim=2).view(batch_size, self.feature_dim)
        
        return weighted_sum.sum(dim=1)
    
class SoftAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SoftAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.w = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.v = nn.Linear(self.hidden_dim, 1)

    def forward(self, h):
        w = torch.tanh(self.w(h))

        weight = self.v(w)
        weight = weight.squeeze(dim=-1)

        weight = torch.softmax(weight, dim=1)
        weight = weight.unsqueeze(dim=-1)
        out = torch.mul(h, weight.repeat(1, 1, h.size(2)))

        out = torch.sum(out, dim=1)

        return out