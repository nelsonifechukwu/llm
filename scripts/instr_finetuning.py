import torch
from llm.loading import load_weights_into_gpt
from llm.gpt_download import download_and_load_gpt2
from llm.instr_dataset import train_loader, val_loader
from llm.config import model_configs
from llm.gpt import GPTModel

BASE_CONFIG = {
"vocab_size": 50257, # Vocabulary size
"context_length": 1024, # Context length
"drop_rate": 0.0, # Dropout rate
"qkv_bias": True # Query-key-value bias
}

CHOOSE_MODEL = "gpt2-medium (355M)"
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2(
model_size=model_size,
models_dir="data/gpt2"
)
model = GPTModel(BASE_CONFIG)
model.eval()
