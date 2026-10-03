from collections import Counter
from datasets import load_dataset
import pandas as pd


mapping = {'B': "billing", 'T': "technical", 'G': "general"}


def load_label_map(path="data/labeled.md"):
    labeled_mapping = {}
    with open(path, "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                print("BAD LINE:", line)
                continue
            label_name = parts[1].replace("\\_", "_")
            label_letter = parts[2].upper()
            labeled_mapping[label_name] = mapping[label_letter]
    return labeled_mapping


LABEL_MAP = load_label_map()


def prepare_split(dataset, names):
    prepared_data = []
    for example in dataset:
        text = example["text"]
        label_id = example["label"]
        original_label = names[label_id]
        our_label = LABEL_MAP[original_label]
        prepared_data.append({
            "text": text,
            "label": our_label
        })
    return prepared_data


def load_banking77():
    ds = load_dataset("PolyAI/banking77", trust_remote_code=True)
    names = ds["train"].features["label"].names
    split = ds["train"].train_test_split(test_size=0.1, seed=42)
    train = prepare_split(split["train"], names)
    val = prepare_split(split["test"], names)
    test = prepare_split(ds["test"], names)
    return train, val, test, names


# AI GENERTAED CODE BELOW
if __name__ == "__main__":
    assert len(LABEL_MAP) == 77, f"Expected 77 labels, got {len(LABEL_MAP)}"
    print("Names per intent:", dict(Counter(LABEL_MAP.values())))

    train, val, test, names = load_banking77()

    print("Missing from file:", set(names) - set(LABEL_MAP))
    print("Typos in file:", set(LABEL_MAP) - set(names))
    print("Sizes:", len(train), len(val), len(test))
    print("First example:", train[0])
    print("Rows per intent:", Counter(r["label"] for r in train))
# END OF AI GENERATED CODE
    pd.DataFrame(train).to_csv("data/banking_train.csv", index=False)
    pd.DataFrame(val).to_csv("data/banking_val.csv", index=False)
    pd.DataFrame(test).to_csv("data/banking_test.csv", index=False)
