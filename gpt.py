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
from attention import multiHeadAttention

#LG(A/F)DS - Layer norm, GELU activation, FFN, Shortcut connection
#pre-layer norm offers better training dynamics than post-layer norm

class GPTModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["drop_rate"])
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
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
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.ff = FeedForward(cfg)
        self.drop_skip_conn = nn.Dropout(cfg["drop_rate"])
        self.attn = multiHeadAttention(d_in = cfg["emb_dim"], d_out = cfg["emb_dim"], context_length=cfg["context_length"],num_heads=cfg["n_heads"], dropout=cfg["drop_rate"])
    
    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.drop_skip_conn(x)
        x = x + shortcut
        
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_skip_conn(x)
        x = x + shortcut
        return x

class LayerNorm(nn.Module):
    #layer norm normalizes across the feature dim while Batchnorm normalizes across the batch dim
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    def forward(self, x):
        #make the layer activations have mean=0 and variance=1
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased = False) # don't use Bessel correction
        x = (x - mean)/torch.sqrt(var + self.eps) #self.eps to prevent divide by 0
        return x * self.scale + self.shift
    

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0/torch.pi)) * (x + 0.044715 * torch.pow(x,3))
        ))    
    
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4*cfg["emb_dim"]),
            GELU(),
            nn.Linear(4*cfg["emb_dim"], cfg["emb_dim"])    
        )
        
    def forward(self, x):
        return self.layers(x)
        

if __name__ == "__main__":  
    from dataloader import create_dataloader
    with open("verdict.txt", "r") as f:
        verdict = f.read()
    dataloader = create_dataloader(
            verdict, batch_size=3, context_size=5, stride=5, shuffle=False
        )
    data_iter = iter(dataloader)
    input, target = next(data_iter)
    GPT = GPTModel(GPT_CONFIG_124M)
    logits = GPT(input)
    print(logits, logits.shape)