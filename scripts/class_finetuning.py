from llm.pretraining import generate_and_print_sample
from llm.gpt import GPTModel
from llm.loading import load_weights_into_gpt
import torch
import torch.nn as nn

from llm.gpt_download import download_and_load_gpt2
from llm.spam_dataset import train_loader, val_loader
from llm.config import model_configs

device = "cuda" if torch.cuda.is_available() else "cpu"

CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "Every effort moves"

BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True,
}

BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

model = GPTModel(BASE_CONFIG)
model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

settings, params = download_and_load_gpt2(model_size=model_size, models_dir="data/gpt2")
load_weights_into_gpt(model, params)
model.eval()

# Classification Finetuning

## freeze all layers
for param in model.parameters():
    param.requires_grad = False

## modify output head
num_class = 2
model.out_head = nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_class)

## let's allow more layers (close to the output head) to be trainable
for param in model.transformer_blocks[-1].parameters():
    param.requires_grad = True
for param in model.final_norm.parameters():
    param.requires_grad = True


## only optimise the last token (the last token is the one with the complete info of all tokens from causal masking)

input = torch.tensor([[22, 23, 1092]])
output = model(input)
prob = torch.softmax(output[:, -1, :], dim=-1)
label = torch.argmax(prob)


## calculate classification acc
def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]
            predicted_labels = torch.argmax(logits, dim=-1)
            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
        return correct_predictions / num_examples


def train_classifier_simple(
    model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter
):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            examples_seen += input_batch.shape[0]
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(
                    f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, "
                    f"Val loss {val_loss:.3f}"
                )

        train_accuracy = calc_accuracy_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_accuracy = calc_accuracy_loader(
            val_loader, model, device, num_batches=eval_iter
        )

        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen


def calc_loss_batch(input_batch, target_batch, model, device):
    # focus on optimizing only the last token, rather than all tokens in input_batch
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:, -1, :]
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
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


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def classify_review(text, model, device, max_length=None,pad_token_id=50256):
    from llm.tokenizer import Tokenizer
    tt = Tokenizer()
    model.eval()
    input_ids = tt.encode(text)
    supported_context_length = model.pos_emb.weight.shape[1]
    eff_max_length = min(max_length, supported_context_length)
    input_ids = input_ids[:eff_max_length]
    input_ids += [pad_token_id] * (eff_max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]
    predicted_label = torch.argmax(logits, dim=-1).item()
    return "spam" if predicted_label == 1 else "not spam"

if __name__ == "__main__":
    import time

    start_time = time.time()
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    num_epochs = 5

    train_losses, val_losses, train_accs, val_accs, examples_seen = (
        train_classifier_simple(
            model,
            train_loader,
            val_loader,
            optimizer,
            device,
            num_epochs=num_epochs,
            eval_freq=50,
            eval_iter=5,
        )
    )
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    import matplotlib.pyplot as plt

    def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
        fig, ax1 = plt.subplots(figsize=(5, 3))
        ax1.plot(epochs_seen, train_values, label=f"Training {label}")
        ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel(label.capitalize())
        ax1.legend()
        ax2 = ax1.twiny()
        ax2.plot(examples_seen, train_values, alpha=0)
        ax2.set_xlabel("Examples seen")
        fig.tight_layout()
        plt.savefig(f"assets/{label}-plot.pdf")
        plt.show()
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))
    plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)
    epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))
    plot_values(
    epochs_tensor, examples_seen_tensor, train_accs, val_accs,
    label="accuracy"
    )
        
    torch.save(model.state_dict(), "checkpoints/classifier.pth")
    
    #load and test model
    model.load_state_dict(torch.load("checkpoints/classifier.pth", weights_only=True))
    model.eval()
    
    input = "Win 100M now. Click this link"
    print(classify_review(input, model, device, 20))
