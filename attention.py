#implementing attention mechanism for GPT models

#simple attention
import torch
import torch.nn as nn
from embedding import input_embeddings

input = input_embeddings[0] 
#to calculate intermediate attention scores, we perform a dot product between the query token and other tokens in the sequence (as a measure of similarity)

query_1 = input[0]
attn_score_1 = torch.empty(input.shape[0], dtype=torch.float32) 
#use torch.empty when you're going to replace the var anyway, not accumlate
#(never assume torch.empty contains zero. see also tensor.fill_(0))
for i, token in enumerate(input):
    attn_score_1[i] = torch.dot(token, query_1)

#then we normalize the scores to give us the attn weight
    #see avg norm: attn_score_1/attn_score_1.sum()
    #see L2 norm: attn_score_1/torch.linalg.vector_norm()
    #see softmax norm: apply softmax (ensures all  weights are +ve)

#this softmax impl is unstable
def soft_max_norm(input):
    return torch.exp(input) / torch.exp(input).sum(dim=0)

#this softmax is tricky for large-dim embeddings because it
#focuses the entire weight (1) on the largest output 
#which is the dot of the token with itself, and zeros the rest
attn_weight_1 = torch.softmax(attn_score_1, dim=0) 


#then we form the context_vec for the query token
#which is sum(attention score * each token)
context_vec_1 = torch.zeros(query_1.shape)
for i, token in enumerate(input):
    context_vec_1 += attn_weight_1[i]*token

#computing context_vec for all input tokens
def compute_context_vec(input_embeddings, matrix_style = False):
    input = input_embeddings[0]
    all_context_vec = torch.zeros(input.shape)
    if matrix_style:
        all_attn_scores = input @ input.T
        all_attn_weights = torch.softmax(all_attn_scores, dim=1)
        all_context_vec = all_attn_weights @ input
        return all_context_vec
      
    for i, _ in enumerate(input):
        attn_scores_per_tok = input @ input[i]
        attn_weights_per_tok = torch.softmax(attn_scores_per_tok, dim=0)
        #context_vec per input
        #context_vec = (torch.diag(attn_weight) @ input).sum(dim=0)
        scales = attn_weights_per_tok[:, None] #reshape the attn_weight to a col vec
        context_vec = (input * scales).sum(dim=0)
        all_context_vec[i] = context_vec
    return all_context_vec

class selfAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False) -> None:
        super().__init__()
        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
    def forward(self, input):
        b, n_tokens, d_in = input.shape
        keys = self.W_k(input) # => input @ self.W_k = (B, T, d_out)
        values = self.W_v(input) # Linear layers are called, not matmul'd
        dim_k = keys.shape[-1]

        #compute context vector for all input queries
        queries = self.W_q(input)
        all_attn_scores = queries @ keys.transpose(-2, -1)# = (B, T, T)

        #normalize weights
        all_scaled_attn_weights = torch.softmax(all_attn_scores/dim_k**0.5, dim=-1)

        #apply dropout
        dp_all_scaled_attn_weights = self.dropout(all_scaled_attn_weights).reshape(all_attn_scores.shape)

        #compute context vectors for all queries
        all_context_vector = dp_all_scaled_attn_weights @ values
        
        return all_context_vector
    
class causalAttention(nn.Module):
    
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False) -> None:
        super().__init__()
        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))
    def forward(self, input):
        b, n_tokens, d_in = input.shape
        keys = self.W_k(input) # => input @ self.W_k =  (B, T, d_out)
        values = self.W_v(input) # Linear layers are called, not matmul'd
        dim_k = keys.shape[-1]

        #compute context vector for all input queries
        queries = self.W_q(input)
        all_attn_scores = queries @ keys.transpose(-2, -1)#  (B, T, T)

        ##apply masked attention
        masked_all_scaled_attn_scores = all_attn_scores.masked_fill_(self.mask.bool()[:n_tokens, :n_tokens], -torch.inf)

        #normalize masked weights
        norm_masked_all_scaled_attn_weights = torch.softmax(masked_all_scaled_attn_scores/dim_k**0.5, dim=-1)

        #apply dropout
        dp_norm_masked_all_scaled_attn_weights = self.dropout(norm_masked_all_scaled_attn_weights).reshape(all_attn_scores.shape)

        #compute context vectors for all queries
        all_context_vector = dp_norm_masked_all_scaled_attn_weights @ values
        
        return all_context_vector
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self,d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList([causalAttention(d_in, d_out, context_length, dropout, qkv_bias) for _ in range(num_heads)])
    
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
    
class multiHeadAttention(nn.Module):
    
    def __init__(self, d_in, d_out, context_length, num_heads, dropout, qkv_bias=False) -> None:
        super().__init__()
        assert (d_out % num_heads == 0)
        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))
    def forward(self, input):
        """
        say head = 2
        Wqkv = din x [embed_dim/head (64) * head (2) = dout (128)]
        q, k, v = 8 x 4 x 256 @ 256 x 128 = 8 x 4 x 128 
        split q,k,v = 8 x 4 x 2(head) x 64(embed_dim/head)
        transpose q,k,v =. 8 x 2 x 4 x 64
        attn_score = q @ key.T = 8 x 2 x 4 x 64  @ 8 x 2 x 64 x 4
        attn_score = 8 x 2 x 4 x 4
        
        attn_weight = softmax(8 x 2 x 4 x 4, dim = -1)
        c_vec = 8 x 2 x 4 x 4 @ 8 x 2 x 4 x 64 
        c_vec = 8 x 4 x 2 x 64 
        c_vec = 8 x 4 x 128
        
        more efficient that MultiHeadAttentionWrapper as we only need to solve qkv (input @ Wqkv) once
        """
        
        B, n_tokens, _ = input.shape
        queries = self.W_q(input) # (B, T, d_out)
        keys = self.W_k(input) 
        values = self.W_v(input) # Linear layers are called, not matmul'd
       

        #split d_out into num_heads x head_dim 
        queries = queries.view(B, n_tokens, self.num_heads, self.head_dim)
        keys = keys.view(B, n_tokens, self.num_heads, self.head_dim)
        values = values.view(B, n_tokens, self.num_heads, self.head_dim)
        
        #transpose: (B, T, H, H_d) ->  (B, H, T, H_d)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        dim_k = keys.shape[-1]
        
        #calculate attention matrix score, (B, H, T, H_d) @ (B, H, H_d, T) = (B, H, T, T)
        all_attn_scores = queries @ keys.transpose(-2, -1) 

        ##apply masked attention. 
        #[:n_tokens (seq_len of current input), :n_tokens] cause n_tokens may != context_length, the max seq_len for this class.
        masked_all_scaled_attn_scores = all_attn_scores.masked_fill_(self.mask.bool()[:n_tokens, :n_tokens], -torch.inf)

        #normalize masked weights
        norm_masked_all_scaled_attn_weights = torch.softmax(masked_all_scaled_attn_scores/dim_k**0.5, dim=-1)

        #apply dropout
        dp_norm_masked_all_scaled_attn_weights = self.dropout(norm_masked_all_scaled_attn_weights)

        #compute context vectors for all queries
        all_context_vector = dp_norm_masked_all_scaled_attn_weights @ values # (B, H, T, H_d)
        
        #concatenate the embeddings per head per token
        all_context_vector = all_context_vector.transpose(1,2) # (B, T, H, H_d)
        all_context_vector = all_context_vector.contiguous().view(B, n_tokens, self.d_out)
        all_context_vector = self.out_proj(all_context_vector)

        return all_context_vector
    

if __name__ == "__main__":

    all_context_vec = compute_context_vec(input_embeddings, True)
    print(all_context_vec[2])
    print(input[2])
    