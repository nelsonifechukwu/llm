GPT_CONFIG_124M = {
"vocab_size": 50257, # Vocabulary size
"context_length": 1024, # Context length
"emb_dim": 768, # Embedding dimension
"n_heads": 12, # Number of attention heads
"n_layers": 12, # Number of layers
"drop_rate": 0.1, # Dropout rate
"qkv_bias": False # Query-Key-Value bias
}

import torch
import torch.nn as nn

class initialGPTModel(nn.Module):
    #LGFS - Layer norm, GELU activation, FFN, Shortcut connection
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["drop_rate"])
        self.transformer_blocks = nn.Sequential(*[initialTransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = initialLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
    
    def forward(self, input):
        B, num_token = input.shape
        token_embs = self.tok_emb(input)
        pos_embs = self.pos_emb(torch.arange(num_token, device=input.device))
        x = token_embs + pos_embs
        x = self.dropout(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        x = self.out_head(x)
        return x
class initialTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
    
    def forward(self, x):
        return x

class initialLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
    def forward(self, x):
        return x