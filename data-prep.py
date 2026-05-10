import re
import tiktoken

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
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: token for token, i in vocab.items()}

    @staticmethod
    def _tokenize(text):
        tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        tokens = [token for token in tokens if token.strip()]
        return tokens

    def encode(self, text, bpe=False):
        if bpe:
            enc = tiktoken.get_encoding("gpt2")
            return enc.encode(text)
        tokens = Tokenizer._tokenize(text)
        encoded_ids = [
            (
                self.str_to_int[token]
                if token in self.str_to_int.keys()
                else self.str_to_int["<|unk|>"]
            )
            for token in tokens
        ]
        return encoded_ids

    def decode(self, ids, bpe=False):
        if bpe:
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


tk = Tokenizer(vocab)
stxt = "you are a good person"
print(tk.encode(stxt, bpe=True))
val = [5832, 389, 257, 922, 1048]
print(tk.decode(val, bpe=True))
