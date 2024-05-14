import torch
import math
import torch.nn as nn
from models.poolings import *
from transformers import ElectraModel

def rotate_every_two(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).reshape_as(x)

def apply_rotary_pos_emb(x, sincos):
    sin, cos = map(lambda t: t.repeat_interleave(2, dim=-1), sincos)
    return (x * cos + rotate_every_two(x) * sin)



class RoPEAttention(nn.Module):
    def init(self, config):
        super().init()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.scale = math.sqrt(self.attention_head_size)
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states, attention_mask=None):
        seq_length = hidden_states.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.float, device=hidden_states.device)
        radians = position_ids[:, None] / (10000 ** (torch.arange(0, self.attention_head_size, 2, dtype=torch.float, device=hidden_states.device) / self.attention_head_size))
        sincos = torch.stack((radians.sin(), radians.cos()), dim=-1).view(1, seq_length, -1).repeat(hidden_states.size(0), 1, 1)

        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)

        q = apply_rotary_pos_emb(q, sincos)
        k = apply_rotary_pos_emb(k, sincos)

        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_probs, v)

        return attention_output

class BaseModel(nn.Module):
    def __init__(self, hidden_dropout_prob=0.2, num_labels=4):
        super(BaseModel, self).__init__()
        self.model = ElectraModel.from_pretrained('google/electra-small-discriminator', output_hidden_states=True)
        self.hidden_size = self.model.config.hidden_size
        self.rope_attention = RoPEAttention(self.model.config)
        self.pooler = MaxPooling()
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.out = nn.Linear(self.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_states = outputs.last_hidden_state
        print(hidden_states.shape, attention_mask.shape)
        attention_output = self.rope_attention(hidden_states, attention_mask)
        pooled_output = self.pooler(attention_output, attention_mask)
        dropout_output = self.dropout(pooled_output)
        out = self.out(dropout_output)

        return out
