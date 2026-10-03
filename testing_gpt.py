from dataloader import create_dataloader
from gpt import GPTModel
from config import GPT_CONFIG_124M, GPT_XLARGE
import torch
from dataprep import Tokenizer

with open("verdict.txt", "r") as f:
    verdict = f.read()
    dataloader = create_dataloader(
        verdict, batch_size=3, context_size=5, stride=5, shuffle=False
    )
data_iter = iter(dataloader)
input, target = next(data_iter)
model = GPTModel(GPT_CONFIG_124M)
logits = model(input)
print(logits, logits.shape)


total_params = sum(p.numel() for p in model.parameters())
print(f"Total Params = {total_params}")

# the above no_of_weights is not quite the 124m. This is cause of weight tying. GPT2 reused the token_emb weights in the output head

total_params_gpt2 = total_params - sum(p.numel() for p in model.out_head.parameters())
print(
    f"Number of trainable parameters "
    f"considering weight tying: {total_params_gpt2:,}"
)


def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:] #don't go beyond context_length
        with torch.no_grad():
            logits = model(idx_cond) # get output
        logits = logits[:, -1, :] # get last embedding from every batch
        prob = torch.softmax(logits, dim=-1) # convert to prob distribution
        idx_next = torch.argmax(prob, dim=-1, keepdim=True) #get the likely next word
        idx = torch.cat((idx, idx_next), dim=1) #append to input for next generation
    return idx

tt = Tokenizer()
test_input = "Hello, how are"
input_tensor = torch.tensor([tt.encode(test_input)])
model.eval()
output = generate_text_simple(model, input_tensor, 100, 10)
text = tt.decode(output.flatten().tolist()) 
print(text)