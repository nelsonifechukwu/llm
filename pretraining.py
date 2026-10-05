#calculate generation loss
#in model training, the goal is to max the value corresponding to the index of the token in the target vector

import torch
import torch.nn as nn
torch.manual_seed(123)
from gpt import GPTModel
from config import GPT_CONFIG_124M, GPT_XLARGE
from dataloader import create_dataloader
from tokenizer import Tokenizer

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

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): " f"Train loss {train_loss:.3f}, "f"Val loss {val_loss:.3f}")
        generate_and_print_sample(model, tokenizer, device, start_context)
    return train_losses, val_losses, track_tokens_seen

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()    
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:] #don't go beyond context_length
        with torch.no_grad():
            logits = model(idx_cond) # get output (B x T x V)
        logits = logits[:, -1, :] # get last embedding from every batch (B x V)
        prob = torch.softmax(logits, dim=-1) # convert to prob distribution
        idx_next = torch.argmax(prob, dim=-1, keepdim=True) #get the likely next word
        idx = torch.cat((idx, idx_next), dim=1) #append to input for next generation
    return idx

def generate_and_print_sample(model, tokenizer, device, start_context):
    tt = Tokenizer()
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = torch.tensor(tt.encode(start_context)).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(model=model, idx=encoded,max_new_tokens=50, context_size=context_size)
    decoded_text = tt.decode(token_ids.flatten().tolist())
    print(decoded_text.replace("\n", " "))
    model.train()
    
    