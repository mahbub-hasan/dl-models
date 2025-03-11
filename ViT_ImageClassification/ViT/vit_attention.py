import torch
import torch.nn as nn
from project_config import ProjectConfig


class ViTMultiHeadAttention(nn.Module):
    def __init__(self, config: ProjectConfig):
        super(ViTMultiHeadAttention, self).__init__()
        self.config = config
        self.scaler = config.embedding_dim ** -0.5
        self.attention_dim = config.embedding_dim // config.attention_head
        self.query = nn.Linear(in_features=config.embedding_dim, out_features=config.embedding_dim)
        self.key = nn.Linear(in_features=config.embedding_dim, out_features=config.embedding_dim)
        self.value = nn.Linear(in_features=config.embedding_dim, out_features=config.embedding_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape

        # [batch_size, seq_len, embed_dim]
        query_state = self.query(x)
        key_state = self.key(x)
        value_state = self.value(x)

        # [batch_size, seq_len, embed_dim] --> [batch_size, seq_len, head, attention_dim] --> [batch_size, head, seq_len, attention_dim]
        query_state = query_state.view(batch_size, seq_len, self.config.attention_head, self.attention_dim).transpose(1,
                                                                                                                      2)
        key_state = key_state.view(batch_size, seq_len, self.config.attention_head, self.attention_dim).transpose(1, 2)
        value_state = value_state.view(batch_size, seq_len, self.config.attention_head, self.attention_dim).transpose(1, 2)

        # dim of attention score = [batch_size, head, seq_len, seq_len]
        attention_score = torch.matmul(query_state, key_state.transpose(2, 3)) * self.scaler
        attention_score = nn.functional.softmax(attention_score, dim=-1)

        # attention output dim: [batch_size, head, seq_len, attention_dim]
        attention_output = torch.matmul(attention_score, value_state)
        attention_output = attention_output.transpose(1, 2).contiguous()

        # final attention output = [batch_size, seq_len, embed_dim]
        attention_output = attention_output.reshape(batch_size, seq_len, embed_dim)

        return attention_output, attention_score
