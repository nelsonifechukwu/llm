#calculate generation loss
#in model training, the goal is to max the value corresponding to the index of the token in the target vector

import torch
import torch.nn as nn
torch.manual_seed(123)
from gpt import GPTModel
from config import GPT_CONFIG_124M, GPT_XLARGE
from dataloader import create_dataloader

GPT_XLARGE["context_length"] = 256
cfg = GPT_XLARGE
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

#Data prep
with open("verdict.txt", "r") as f:
    verdict = f.read()
train_ratio = 0.90
split_idx = int(train_ratio * len(verdict))
train_data = verdict[:split_idx]
val_data = verdict[split_idx:]

train_loader = create_dataloader(train_data, cfg["context_length"], cfg["context_length"], 2)
val_loader = create_dataloader(val_data, cfg["context_length"], cfg["context_length"], 2)

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = nn.functional.cross_entropy(logits.flatten(0,1), target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (in_batch, out_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(in_batch, out_batch, model,device)
            total_loss += loss.item()
        else: 
            break
        return total_loss/num_batches
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)
print("Training loss:", train_loss)
print("Validation loss:", val_loss)