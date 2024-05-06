import torch
import numpy as np
from torch import nn as nn

class self_attention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        #q, k, v matrices initialize
        self.query = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.key = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.value = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        
        #scaler factor 
        self.scaler = 1 / (embed_dim // num_heads) ** 0.5
        
        #output transform
        self.output = nn.Linear(in_features=embed_dim, out_features=embed_dim)
    
    def forward(self, x):
        # x has shape of (B, L, H)
        batch_size, length, hidden_state = x.shape

        query = self.query(x)  
        key = self.key(x)
        value = self.value(x)

        #split q, k, v to multiple heads
        query = query.view(batch_size, length, self.num_heads, hidden_state // self.num_heads).transpose(1, 2)
        key = key.view(batch_size, length, self.num_heads, hidden_state // self.num_heads).transpose(1, 2)
        value = value.view(batch_size, length, self.num_heads, hidden_state // self.num_heads).transpose(1, 2) #(B, N, L, H)
        
        #calculate attention scores
        attention_score = torch.matmul(query, key.transpose(-2, -1)) * self.scaler
        attention_probs = torch.softmax(attention_score, dim=-1) # (B, N, L, L)
        
        context = torch.matmul(attention_probs, value) # (B, N, L, H)
        context = context.transpose(1,2).reshape(batch_size, length, -1) # (B, L, N*H)
        
        output = self.output(context)
        return output
        
        
if __name__ == "__main__":
    x = torch.randn(2, 3, 4)
    model = self_attention(4, 2)
    output = model(x)
    print(output.shape)