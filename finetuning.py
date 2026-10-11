from pretraining import generate_and_print_sample
from gpt import GPTModel
from loading import load_weights_into_gpt
import torch.nn as nn

from gpt_download import download_and_load_gpt2

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

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

settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")
load_weights_into_gpt(model, params)
model.eval()

generate_and_print_sample(model, "cpu", "Hello, who are you?", 50, 1.0, 0.0)

# Classification Finetuning

## freeze all layers
for param in model.parameters():
    param.requires_grad = False
    
## modify output head
num_class = 2
model.out_head = nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_class)