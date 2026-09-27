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
    def __init__(self, d_in, d_out, qkv_bias=False) -> None:
        super().__init__()
        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias) #256 x 128
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
    def forward(self, input):
        keys = self.W_k(input) # 8 x 4 x 128
        values = self.W_v(input) # Linear layers are called, not matmul'd
        dim_k = keys.shape[-1]

        #compute attn score for all input queries
        queries = self.W_q(input)
        all_attn_scores = torch.einsum('ijk,abk->ijab', queries, keys) # 8 x 4 x 8 x 4

        #normalize attn score to get attn weights for all input queries
        all_scaled_attn_weights = torch.softmax(
        all_attn_scores.flatten(-2) / dim_k**0.5, dim=-1
            ).unflatten(-1, all_attn_scores.shape[-2:]) #unflatten(-1...) implies split the last dimension into ...

        ##apply masked attention
        #flatten weights 
        flat_all_scaled_attn_weights = all_scaled_attn_weights.reshape(32,32)

        #create mask
        mask = torch.triu(torch.ones(flat_all_scaled_attn_weights.shape), diagonal=1)

        #apply mask
        masked_all_scaled_attn_weights = flat_all_scaled_attn_weights.masked_fill(mask.bool(), -torch.inf)

        #normalize masked weights
        norm_masked_all_scaled_attn_weights = torch.softmax(masked_all_scaled_attn_weights/dim_k**0.5, dim=-1)

        #compute context vectors for all queries
        all_context_vector = torch.einsum('ijkl,kld->ijd',all_scaled_attn_weights, values)
        
        return all_context_vector

if __name__ == "__main__":

    all_context_vec = compute_context_vec(input_embeddings, True)
    print(all_context_vec[2])
    print(input[2])
    