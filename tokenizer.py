import re
import tiktoken
from typing import Optional, Union

# from importlib.metadata import version
# print("tiktoken version:", version("tiktoken"))

with open("verdict.txt", "r") as f:
    verdict = f.read()
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', verdict)
preprocessed = [token for token in preprocessed if token.strip()]

# build vocab, token -> id mapping
vocab = sorted(list(set(preprocessed)))
vocab = vocab + ["<|endoftext|>", "<|unk|>"]
vocab = {token: i for i, token in enumerate(vocab)}


class Tokenizer:
    def __init__(self, use_bpe = True, vocab: Optional[dict] = None):
        self.use_bpe = use_bpe
        if self.use_bpe:
            return
        if vocab is None:
            raise ValueError("vocabulary input missing")
        self.str_to_int = vocab
        self.int_to_str = {i: token for token, i in vocab.items()}


    @staticmethod
    def _tokenize(input):
        tokens = re.split(r'([,.:;?_!"()\']|--|\s)', input)
        tokens = [token for token in tokens if token.strip()]
        return tokens

    def encode(self, input):
        if self.use_bpe:
            enc = tiktoken.get_encoding("gpt2")
            return enc.encode(input)
        tokens = Tokenizer._tokenize(input)
        encoded_ids = [
            (
                self.str_to_int[token]
                if token in self.str_to_int.keys()
                else self.str_to_int["<|unk|>"]
            )
            for token in tokens
        ]
        return encoded_ids

    def decode(self, ids):
        if self.use_bpe:
            enc = tiktoken.get_encoding("gpt2")
            return enc.decode(ids)
        return " ".join(
            [
                (
                    self.int_to_str[id]
                    if id in self.int_to_str
                    else self.int_to_str[self.str_to_int["<|unk|>"]]
                )
                for id in ids
            ]
        )

if __name__ == "__main__":
    tk = Tokenizer(False)
    stxt = "you are a good person"
    print(tk.encode(stxt))
    val = [5832, 389, 257, 922, 1048]
    print(tk.decode(val))
