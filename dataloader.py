from dataprep import Tokenizer
import torch
from torch.utils.data import DataLoader, Dataset


with open("verdict.txt", "r", encoding="utf-8") as f:
    verdict = f.read()

# enc_text = tokenizer.encode(verdict)
# enc_sample = enc_text[100:]

# context_size = 4 #aka max_length, is similar to Claude's context window--the number of tokens the model considers when making predictions, allowing it to understand and generate coherent text based on that context.
# x = enc_sample[:context_size]
# y = enc_sample[1:context_size+1]

# # pair of input and target sequences
# for i in range(context_size):
#     print(tokenizer.decode(x[:i+1]), "----->", tokenizer.decode([y[i]]))


class CustomDataset(Dataset):
    def __init__(self, input, tokenizer, context_size, stride):
        self.token_ids = tokenizer.encode(input)
        self.input_ids = []
        self.target_ids = []
        for i in range(0, len(self.token_ids) - context_size, stride):
            input_chunk = self.token_ids[i : i + context_size]  # [a,b,c,d,e]
            target_chunk = self.token_ids[i + 1 : i + context_size + 1]  # [b,c,d,e,f]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(
    input,
    context_size=1,
    stride=1,
    batch_size=1,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):  
    tokenizer = Tokenizer()
    dataset = CustomDataset(input, tokenizer, context_size, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return dataloader


if __name__ == "__main__":
    dataloader = create_dataloader(
        verdict, batch_size=3, context_size=5, stride=5, shuffle=False
    )
    data_iter = iter(dataloader)
    input, target = next(data_iter)
    print(input)
    print(target)
