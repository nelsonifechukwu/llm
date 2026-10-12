# llm

Learning transformers by building a GPT-2 (124M) from scratch, following *Build a Large Language Model (From Scratch)*.

## Layout

```
llm/          model + data code (tokenizer, dataloader, embedding, attention, gpt, config,
              pretraining, gpt_download, loading, dataset)
scripts/      entry points: class_finetuning.py, instr_finetuning.py (WIP)
tests/        sanity checks: testing_gpt.py, testing_self_attention.py, testing_skip_connections.py
data/         verdict.txt (tracked); gpt2 weights, SMS spam data and CSVs (git-ignored)
checkpoints/  saved model weights (git-ignored)
assets/       figures and plots
```

Run everything from the repo root as a module, e.g.:

```
python -m scripts.class_finetuning
python -m tests.testing_gpt
python -m llm.pretraining
```
