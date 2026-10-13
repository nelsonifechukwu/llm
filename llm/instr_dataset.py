import json
import os
import urllib.request
import torch
from torch.utils.data import DataLoader, Dataset
from llm.tokenizer import Tokenizer


file_path = "data/instruction-data.json"
url = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
    "/main/ch07/01_main-chapter-code/instruction-data.json"
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EOF_TOKEN = 50256
def download_and_load_file(file_path, url):
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    with open(file_path, "r") as file:
        return json.load(file)


data = download_and_load_file(file_path, url)


def format(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text


# partition the dataset
train_portion = int(len(data) * 0.85)
test_portion = int(len(data) * 0.1)
val_portion = len(data) - train_portion - test_portion
train_data = data[:train_portion]
test_data = data[train_portion : train_portion + test_portion]
val_data = data[train_portion + test_portion :]


# Batch the dataset (in this batching, we want automatic padding. So, the collate func has to change)
class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(tokenizer.encode(full_text))

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


def custom_collate_fn(batch, pad_token_id=EOF_TOKEN, ignore_index = -100, max_length= None,device=None):
    batch_max_length = max(
        len(item) + 1 for item in batch
    )  # +1 to keep the appended last token when shifting the input to make the target
    inputs_lst, targets_lst = [], []
    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])
        
        #replace all the extra pad_token_id except the first one with ignore_index and default ignore_index for cross_entropy loss in pytorch is -100 -> the token id should be ignored in computing the cross_entropy loss
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze(0)
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index
            
        if max_length is not None:
            inputs = inputs[:max_length]
            targets = targets[:max_length]
            
        inputs_lst.append(inputs)
        targets_lst.append(targets)
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor

#create new custom_collate_fn with modified args
from functools import partial 
new_custom_collate_fn = partial(custom_collate_fn, max_length=1024, device=device)
