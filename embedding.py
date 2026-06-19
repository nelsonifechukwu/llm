import torch
from dataloader import create_dataloader

with open("verdict.txt", "r", encoding="utf-8") as f:
    verdict = f.read()
    

#load tokens
data_loader = create_dataloader(verdict, 5, 5, 1, False, True)
data_loader = iter(data_loader)
input, target = next(data_loader)

#embedding layer
torch.manual_seed(123)
vocab_size = 50257 #vocab size for bpe
output_dim = 256
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

#tokens -> embeddings (similar to one-hot encoding)
token_embeddings = embedding_layer(input)

"""
[one-hot vector][embedding matrix]    
                [embedding matrix]
                [embedding matrix]
"""
one_hot = torch.nn.functional.one_hot(input.squeeze(dim=0), vocab_size)
linear = torch.nn.Linear(vocab_size, output_dim, bias=False)
linear.weight = torch.nn.Parameter(embedding_layer.weight)
output = one_hot.float() @ linear.weight #similar to linear(one_hot.float())
print(output)
print(token_embeddings)

