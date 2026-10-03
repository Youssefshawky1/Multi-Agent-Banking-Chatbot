import pandas as pd
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

banking_data = pd.read_csv("data/banking_train.csv")
label_map = {
    "billing": 0,
    "technical": 1,
    "general": 2
}
banking_data["label_id"] = banking_data["label"].map(
    label_map)  # AI GENERATED LINE


def tokenize(text):
    tokens = text.lower().split(" ")
    return tokens


def build_vocab(texts):
    # AI GENERATED CODE BELOW
    vocab = {
        g: i for i, g in enumerate(["<pad>", "<unk>"])
    }
    # AI GENERATED CODE ABOVE
    for text in texts:
        tokens = tokenize(text)
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


MAX_LENGTH = 30  # here we put this number because about 4.5% onlyy of the sentences are longer than 30 words


def encode(text, vocab):
    tokens = tokenize(text)
    encoded = []
    for token in tokens:
        if token in vocab:
            encoded.append(vocab[token])
        else:
            encoded.append(vocab["<unk>"])
    encoded = encoded[0:MAX_LENGTH]
    encoded += [vocab["<pad>"]] * \
        (MAX_LENGTH - len(encoded))  # AI GENERATED LINE
    return encoded


vocab = build_vocab(banking_data["text"])
encoded_texts = banking_data["text"].apply(lambda x: encode(x, vocab))
# AI GENERATED CODE BELOW
X = torch.tensor(encoded_texts.tolist(), dtype=torch.long)
y = torch.tensor(banking_data["label_id"].tolist(), dtype=torch.long)
# AI GENERATED CODE ABOVE
dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
