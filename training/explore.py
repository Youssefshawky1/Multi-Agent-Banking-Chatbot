import prepare_data
from datasets import load_dataset

ds = load_dataset("PolyAI/banking77", trust_remote_code=True)

names = ds["train"].features["label"].names

for i, example in enumerate(ds["train"]):
    text = example["text"]
    label_id = example["label"]
    original_label = names[label_id]
    our_label = prepare_data.LABEL_MAP[original_label]
    print("Text:", text)
    print("Label ID:", label_id)
    print("Original label:", original_label)
    print("Our label:", our_label)
    if i == 5:
        break
