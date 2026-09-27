import torch
from embedding import input_embeddings

x_1 = input_embeddings[0][0] #1 x 256 from 8 x 4 x 256
d_in = int(input_embeddings.shape[2]) #256
d_out = d_in//2 #128

W_q = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False) #256 x 128
W_k = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_v = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

q_1 = x_1 @ W_q #1 x 128
k_1 = x_1 @ W_k
v_1 = x_1 @ W_v

# obtain all KV pairs for all inputs
keys = input_embeddings @ W_k # 8 x 4 x 128
values = input_embeddings @ W_v

#compute attn_scores for all inputs wrt q_1
attn_scores_q1 = keys @ q_1 # 8 x 4
dim_k = keys.shape[-1]
original_shape = attn_scores_q1.shape
attn_scores_q1 = attn_scores_q1.reshape(1, -1)
scaled_attn_weights_q1 = torch.softmax(attn_scores_q1/dim_k**0.5, dim = 1)
scaled_attn_weights_q1 = scaled_attn_weights_q1.reshape(original_shape) # 8 x 4

#compute context vector
ctx_vec_q1 = torch.einsum('ij,ijk->k', scaled_attn_weights_q1, values)

#compute context vector for all input queries
queries = input_embeddings @ W_q
all_attn_scores = queries @ keys.transpose(1, 2)


all_scaled_attn_weights = torch.softmax(
    all_attn_scores.flatten(-2) / dim_k**0.5, dim=-1
).unflatten(-1, all_attn_scores.shape[-2:]) #unflatten(-1...) implies split the last dimension into ... 8 x 4 x 8 x 4

##apply masked attention
#flatten weights (cause of 2 query & 2 key axes)
        
k = all_scaled_attn_weights.dim() // 2 
q_size = all_scaled_attn_weights.shape[:k].numel()
kv_size = all_scaled_attn_weights.shape[k:].numel()
flat_all_scaled_attn_weights = all_scaled_attn_weights.reshape(q_size, kv_size)

#create mask
mask = torch.triu(torch.ones(flat_all_scaled_attn_weights.shape), diagonal=1)

#apply mask
masked_all_scaled_attn_weights = flat_all_scaled_attn_weights.masked_fill(mask.bool(), -torch.inf)

#normalize masked weights
norm_masked_all_scaled_attn_weights = torch.softmax(masked_all_scaled_attn_weights/dim_k**0.5, dim=-1)

 #apply dropout
dropout = torch.nn.Dropout(0.5)
dp_norm_masked_all_scaled_attn_weights = dropout(norm_masked_all_scaled_attn_weights).reshape(all_attn_scores.shape)

#compute context vectors for all queries
all_context_vector = torch.einsum('ijkl,kld->ijd',dp_norm_masked_all_scaled_attn_weights, values)

print(all_context_vector.shape)