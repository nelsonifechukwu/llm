import torch
from embedding import input_embeddings

x_1 = input_embeddings[0][0] #1 x 256 from 8 x 4 x 256
d_in = int(input_embeddings.shape[2]) #256
d_out = d_in//2 #128

W_q = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False) #256 x 128
W_k = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_v = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

q_1 = x_1 @ W_q
k_1 = x_1 @ W_k
v_1 = x_1 @ W_v

# obtain all KV pairs for all inputs
keys = input_embeddings @ W_k
values = input_embeddings @ W_v


print(keys.shape)