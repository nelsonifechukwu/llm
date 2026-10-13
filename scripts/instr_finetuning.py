import torch
from llm.loading import load_weights_into_gpt
from llm.gpt_download import download_and_load_gpt2
from llm.instr_dataset import train_loader, val_loader
from llm.config import model_configs
from llm.gpt import GPTModel
from llm.pretraining import calc_loss_loader, train_model_simple
from llm.instr_dataset import format, val_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_CONFIG = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,  # Dropout rate
    "qkv_bias": True,  # Query-key-value bias
}

CHOOSE_MODEL = "gpt2-medium (355M)"
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2(model_size=model_size, models_dir="data/gpt2")
model = GPTModel(BASE_CONFIG)
model.eval()
model.to(device)


if __name__ == "__main__":
    import time

    start_time = time.time()
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)
    num_epochs = 2
    train_losses, val_losses, tokens_seen = train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs=num_epochs,
        eval_freq=5,
        eval_iter=5,
        start_context=format(val_data[0]),
    )
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")
    torch.save(model.state_dict(), "checkpoints/instruction.pth")