import re

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

    def encode(self, text):
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

    def decode(self, ids):
        return " ".join([self.int_to_str[id] for id in ids])


tk = Tokenizer(vocab)
stxt = "you are a good person"
print(tk.encode(stxt))
