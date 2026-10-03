from sklearn.metrics import f1_score
import pandas as pd
from model import IntentClassifier
from preprocess import vocab, train_loader, encode, label_map
import torch
import torch.nn as nn


model = IntentClassifier(vocab_size=len(vocab))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 5
best_f1 = 0.0
validation_data = pd.read_csv("data/banking_val.csv")

validation_data["label_id"] = validation_data["label"].map(label_map)

X_val = torch.tensor(
    validation_data["text"].apply(
        lambda x: encode(x, vocab)
    ).tolist(),
    dtype=torch.long
)

y_val = torch.tensor(
    validation_data["label_id"].tolist(),
    dtype=torch.long
)
for epoch in range(epochs):
    total_loss = 0
    # AI GENERATED CODE BELOW
    for batch_X, batch_y in train_loader:
        lengths = (batch_X != 0).sum(dim=1)
        output = model(batch_X, lengths)
        loss = criterion(output, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    average_loss = total_loss / len(train_loader)
    print("Epoch:", epoch + 1, "Average Loss:", average_loss)

    model.eval()
    with torch.no_grad():

        lengths = (X_val != 0).sum(dim=1)

        output = model(X_val, lengths)

        predictions = torch.argmax(output, dim=1)

    validation_accuracy = (
        (predictions == y_val).sum().item()
        / len(y_val)
    )

    validation_f1 = f1_score(
        y_val.numpy(),
        predictions.numpy(),
        average="macro"
    )

    print("Validation Accuracy:", validation_accuracy)
    print("Validation Macro-F1:", validation_f1)

    if validation_f1 > best_f1:

        best_f1 = validation_f1

        torch.save(
            {
                "model_state": model.state_dict(),
                "vocab": vocab,
                "label_map": label_map
            },
            "training/classifier.pt"
        )

    model.train()

# AI GENERATED CODE ABOVE
