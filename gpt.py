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
    
    
if __name__ == "__main__":  
    from dataloader import create_dataloader
    with open("verdict.txt", "r") as f:
        verdict = f.read()
    dataloader = create_dataloader(
            verdict, batch_size=3, context_size=5, stride=5, shuffle=False
        )
    data_iter = iter(dataloader)
    input, target = next(data_iter)
    GPT = initialGPTModel(GPT_CONFIG_124M)
    logits = GPT(input)
    print(logits, logits.shape)