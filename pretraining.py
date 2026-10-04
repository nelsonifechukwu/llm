#calculate generation loss

import torch
from gpt import GPTModel
from config import GPT_CONFIG_124M

inputs = torch.tensor([[16833, 3626, 6100], [40, 1107, 588]])
model = GPTModel(GPT_CONFIG_124M)
with torch.no_grad():
        logits = model(inputs) # get output (B x T x V)
prob = torch.softmax(logits, dim=-1) # convert to prob distribution
idx_next = torch.argmax(prob, dim=-1, keepdim=True) #get the likely next word

#show max prob value
max_prob_value = idx_next[1].flatten().tolist()
print(prob[0, [0,1,2], max_prob_value])