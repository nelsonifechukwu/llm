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
attn_scores_q1 = keys @ q_1
dim_k = keys.shape[-1]
original_shape = attn_scores_q1.shape
attn_scores_q1 = attn_scores_q1.reshape(1, -1)
scaled_attn_weights_q1 = torch.softmax(attn_scores_q1/dim_k**0.5, dim = 1)
scaled_attn_weights_q1 = scaled_attn_weights_q1.reshape(original_shape)


print(scaled_attn_weights_q1.shape)