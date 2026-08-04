#implementing attention mechanism for GPT models

#simple attention
import torch
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
input = input_embeddings[0]
all_context_vec = torch.zeros(input.shape)
for i, _ in enumerate(input):
    attn_scores = input @ input[i]
    attn_weight = torch.softmax(attn_scores, dim=0)
    #context_vec per input
    #context_vec = (torch.diag(attn_weight) @ input).sum(dim=0)
    scales = attn_weight[:, None] #reshape the attn_weight to a col vec
    context_vec = (input * scales).sum(dim=0)
    all_context_vec[i] = context_vec

print(all_context_vec[2])
print(input[2])
    