import torch
from dataloader import create_dataloader

with open("verdict.txt", "r", encoding="utf-8") as f:
    verdict = f.read()
    
create_dataloader(verdict)