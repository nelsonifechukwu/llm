#calculate generation loss
#in model training, the goal is to max the value corresponding to the index of the token in the target vector

import torch
import torch.nn as nn
from gpt import GPTModel
from config import GPT_CONFIG_124M

inputs = torch.tensor([[16833, 3626, 6100], [40, 1107, 588]])
targets = torch.tensor([[3626, 6100, 44], [1107, 588, 109]])
model = GPTModel(GPT_CONFIG_124M)
with torch.no_grad():
        logits = model(inputs) # get output (B x T x V)
prob = torch.softmax(logits, dim=-1) # convert to prob distribution
idx_next = torch.argmax(prob, dim=-1, keepdim=True) #get the likely next word

target_1 = prob[0, [0,1,2], targets[0]].unsqueeze(0)
target_2 = prob[1, [0,1,2], targets[1]].unsqueeze(0)
t_output_probs = torch.cat((target_1, target_2), dim=0)


#cross entropy loss
loss = nn.functional.cross_entropy(logits.flatten(0,1) , targets.flatten())
print(loss)

