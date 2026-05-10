
import re
with open("verdict.txt", "r") as f:
    verdict = f.read()
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', verdict)
preprocessed = [token for token in preprocessed if token.strip()]

#build vocab, token -> id mapping
vocab = sorted(list(set(preprocessed)))
vocab = vocab + ["<|endoftext|>", "<|unk|>"]
vocab = {token: i for i, token in enumerate(vocab)}

class Tokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: token for token, i in vocab.items()}
        
    def tokenize(self, text):
        tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        tokens = [token for token in tokens if token.strip()]
        return tokens

    def encode(self, text):
        encoded_ids =[]
        tokens = self.tokenize(text)
        for token in tokens:
            if token not in self.str_to_int.keys():
                token = "<|unk|>"
            encoded_ids.append(self.str_to_int[token])
        return encoded_ids


    def decode(self, ids):
        return ' '.join([self.int_to_str[id] for id in ids])

tk = Tokenizer(vocab)
stxt = [3,6,10,22,5,8,12,15,50]
print(tk.decode(stxt))