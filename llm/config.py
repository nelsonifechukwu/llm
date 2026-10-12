GPT_CONFIG_124M = {
"vocab_size": 50257, # Vocabulary size
"context_length": 256, # Context length
"emb_dim": 768, # Embedding dimension
"n_heads": 12, # Number of attention heads
"n_layers": 12, # Number of layers
"drop_rate": 0.1, # Dropout rate
"qkv_bias": False # Query-Key-Value bias
}

GPT_MEDIUM = {
"vocab_size": 50257, # Vocabulary size
"context_length": 1024, # Context length
"emb_dim": 1024, # Embedding dimension
"n_heads": 16, # Number of attention heads
"n_layers": 24, # Number of layers
"drop_rate": 0.1, # Dropout rate
"qkv_bias": False # Query-Key-Value bias
}


GPT_LARGE = {
"vocab_size": 50257, # Vocabulary size
"context_length": 1024, # Context length
"emb_dim": 1280, # Embedding dimension
"n_heads": 20, # Number of attention heads
"n_layers": 36, # Number of layers
"drop_rate": 0.1, # Dropout rate
"qkv_bias": False # Query-Key-Value bias
}

GPT_XLARGE = {
"vocab_size": 50257, # Vocabulary size
"context_length": 1024, # Context length
"emb_dim": 1600, # Embedding dimension
"n_heads": 25, # Number of attention heads
"n_layers": 48, # Number of layers
"drop_rate": 0.1, # Dropout rate
"qkv_bias": False # Query-Key-Value bias
}

#TTLO
#token embed (50257 x 1600)
#pos embed (ctx_length x 1600)          
#transformer  (this whole block × n_layers )
    #layernorm (1600 x 2)                   <--  LayerNorm owns TWO params (scale AND shift), not one
    #attn
        #Wqkv (1600 x 1600 x 3)             <-- correct as weight-only: qkv_bias=False in your cfg
        #out_proj (1600 x 1600) + (1600)    <-- + the bias: nn.Linear defaults to bias=True
    #layernorm (1600 x 2)                   <-- same fix as above
    #ff  (1600*4*1600 + 4*1600) + (4*1600*1600 + 1600)   [first linear weight+bias, second linear weight+bias]
#layernorm (1600 x 2)                       <-- final_norm, same fix
#output head (1600 x 50257)                 <-- correct as weight-only: you explicitly set bias=False here
