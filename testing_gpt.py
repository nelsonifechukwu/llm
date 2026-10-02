from dataloader import create_dataloader
from gpt import GPTModel, GPT_CONFIG_124M

with open("verdict.txt", "r") as f:
        verdict = f.read()
        dataloader = create_dataloader(
            verdict, batch_size=3, context_size=5, stride=5, shuffle=False
        )
data_iter = iter(dataloader)
input, target = next(data_iter)
GPT = GPTModel(GPT_CONFIG_124M)
logits = GPT(input)
print(logits, logits.shape)