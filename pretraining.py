# calculate generation loss
# in model training, the goal is to max the value corresponding to the index of the token in the target vector

import torch
import torch.nn as nn

torch.manual_seed(123)
from gpt import GPTModel
from config import GPT_CONFIG_124M, GPT_XLARGE 
from dataloader import create_dataloader
from tokenizer import Tokenizer

cfg = GPT_CONFIG_124M

model = GPTModel(GPT_CONFIG_124M)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

###Testing
inputs = torch.tensor([[16833, 3626, 6100], [40, 1107, 588]])
targets = torch.tensor([[3626, 6100, 44], [1107, 588, 109]])
with torch.no_grad():
    logits = model(inputs)  # get output (B x T x V)
prob = torch.softmax(logits, dim=-1)  # convert to prob distribution
idx_next = torch.argmax(prob, dim=-1, keepdim=True)  # get the likely next word
target_1 = prob[0, [0, 1, 2], targets[0]].unsqueeze(0)
target_2 = prob[1, [0, 1, 2], targets[1]].unsqueeze(0)
t_output_probs = torch.cat((target_1, target_2), dim=0)

# Data prep
with open("verdict.txt", "r") as f:
    verdict = f.read()
train_ratio = 0.90
split_idx = int(train_ratio * len(verdict))
train_data = verdict[:split_idx]
val_data = verdict[split_idx:]

train_loader = create_dataloader(
    train_data, cfg["context_length"], cfg["context_length"], 2
)
val_loader = create_dataloader(
    val_data, cfg["context_length"], cfg["context_length"], 2
)


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (in_batch, out_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(in_batch, out_batch, model, device)
            total_loss += loss.item()
        else:
            break
        return total_loss / num_batches


with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)
print("Training loss:", train_loss)
print("Validation loss:", val_loss)


def train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
):
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
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, "
                    f"Val loss {val_loss:.3f}"
                )
        generate_and_print_sample(model, device, start_context, 2.0, 25)
    return train_losses, val_losses, track_tokens_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def generate_text_simple(
    model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]  # don't go beyond context_length
        with torch.no_grad():
            logits = model(idx_cond)  # get output (B x T x V)
        logits = logits[:, -1, :]  # get last embedding from every batch (B x V)
        # convert to prob distribution and get the likely next word #see also
        
        # in top-k sampling, we restrict the sampler to only sample from top-k probabilities from the softmax(logits), and masking others (-inf -> 0 after softmax)
        if top_k:
            top_logits, top_pos = torch.topk(logits, top_k)
            min_val = top_logits[:, -1] #top_logits is in descending order
            logits = torch.where(
                condition=logits < min_val,  # the minimum in the top logits
                input=torch.tensor(float('-inf')).to(logits.device),
                other=logits,
            )
        # reducing temperature -> tends towards arg_max like certainty, increasing it adds more variety to the possible token to be generated -> a more uniformly distributed next-token probabilities
        if temperature > 0.0:  # apply temperature scaling & probabilistic sampling.
            scaled_logits = logits / temperature
            prob = torch.softmax(scaled_logits, dim=-1)
            idx_next = torch.multinomial(prob, num_samples=1)
        else:
            #no need to apply softmax here since it's a monotonic op
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        if idx_next == eos_id: #stop generating if end of sequence is encountered
            break
        idx = torch.cat((idx, idx_next), dim=1)  # append to input for next generation
    return idx


def generate_and_print_sample(model, device, start_context, temperature, top_k):
    tt = Tokenizer()
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = torch.tensor([tt.encode(start_context)]).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded, max_new_tokens=50, context_size=context_size, 
            temperature = temperature, top_k=top_k,
        )
    decoded_text = tt.decode(token_ids.flatten().tolist())
    print(decoded_text.replace("\n", " "))
    model.train()
    
def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
    num_epochs = 10
    train_losses, val_losses, tokens_seen = train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs=num_epochs,
        eval_freq=5,
        eval_iter=5,
        start_context="I turned to Mrs. Gisburn",
    )

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
