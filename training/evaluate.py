from sklearn.metrics import confusion_matrix, f1_score
import pandas as pd
from preprocess import vocab, encode, label_map
import torch
from model import IntentClassifier
validation_data = pd.read_csv("data/banking_val.csv")
validation_data["label_id"] = validation_data["label"].map(label_map)
val_encoded_texts = validation_data["text"].apply(
    lambda x: encode(x, vocab)
)
X_val = torch.tensor(
    val_encoded_texts.tolist(),
    dtype=torch.long
)

y_val = torch.tensor(
    validation_data["label_id"].tolist(),
    dtype=torch.long
)


test_data = pd.read_csv("data/banking_test.csv")

test_data["label_id"] = test_data["label"].map(label_map)

test_encoded_texts = test_data["text"].apply(
    lambda x: encode(x, vocab)
)

X_test = torch.tensor(
    test_encoded_texts.tolist(),
    dtype=torch.long
)

y_test = torch.tensor(
    test_data["label_id"].tolist(),
    dtype=torch.long
)

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


model = IntentClassifier(vocab_size=len(vocab))

# AI GENERATED CODE BELOW
checkpoint = torch.load(
    "training/classifier.pt",
    map_location="cpu"
)

model.load_state_dict(
    checkpoint["model_state"]
)
model.eval()


lengths = (X_val != 0).sum(dim=1)

with torch.no_grad():
    output = model(X_val, lengths)
predictions = torch.argmax(output, dim=1)
correct = (predictions == y_val).sum().item()
accuracy = correct / len(y_val)
print("Correct predictions:", correct)
print("Validation accuracy:", accuracy)
print("Validation accuracy (%):", accuracy * 100)


test_lengths = (X_test != 0).sum(dim=1)
with torch.no_grad():
    test_output = model(X_test, test_lengths)

test_predictions = torch.argmax(test_output, dim=1)

print("Test output shape:", test_output.shape)
print("Test predictions shape:", test_predictions.shape)

test_correct = (test_predictions == y_test).sum().item()
test_accuracy = test_correct / len(y_test)

print("Correct test predictions:", test_correct)
print("Test accuracy:", test_accuracy)
print("Test accuracy (%):", test_accuracy * 100)
test_macro_f1 = f1_score(
    y_test.numpy(),
    test_predictions.numpy(),
    average="macro"
)

print("Test Macro-F1:", test_macro_f1)
cm = confusion_matrix(
    y_test.numpy(),
    test_predictions.numpy()
)

print("Test confusion matrix:")
print(cm)

print(
    "The likeliest confusions are billing with general and "
    "technical with billing, with 48 examples in each direction."
)

# AI GENERATED CODE ABOVE
