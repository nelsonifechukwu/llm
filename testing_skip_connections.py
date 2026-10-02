#it's simple adding the output of one layer to the next (if both outputs are compatible) to avoid vanishing gradients at the earlier layers during backprop

import torch
import torch.nn as nn
from gpt import GELU

class skipConnection(nn.Module):  
    def __init__(self, layer_sizes, skip=False):
        super().__init__()
        self.skip = skip
        self.layer_sizes = layer_sizes
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()), 
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()), 
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU())]
            
        )
    
    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)
            if self.skip and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x

def print_gradients(model, x):
    output = model(x)
    target = torch.tensor([[0.]])
    loss = nn.MSELoss()
    loss = loss(output, target)
    loss.backward()
    for name, param in model.named_parameters():
        if 'weight' in name:
            print (f"{name}'s avg grad is {param.grad.abs().mean().item()}")

if __name__ == "__main__":  
    layer_sizes = [3,3,3,2,1]
    model = skipConnection(layer_sizes)
    input = torch.tensor([1., 0, 2.,])
    output = model(input)
    print(output)
    print_gradients(model, input)
    model = skipConnection(layer_sizes, skip=True)
    print("after applying skip connections")
    print_gradients(model, input)