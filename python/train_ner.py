import torch
import numpy as np
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from datasets import Dataset
import onnx
import os
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForTokenClassification
import json

# Setup mac acceleration
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Training on: {device}")

# Create dataset
# In a real scenario, you would load this from a JSON file.
# Note: Inputs are LISTS of words, not raw strings.
# data = [
#     {"tokens": ["Look", "at", "the", "door"], "tags": ["B-LOOK", "O", "O", "O"]},
#     {"tokens": ["Move", "forward", "quickly"], "tags": ["B-MOVE", "I-MOVE", "O"]},
#     {"tokens": ["Turn", "left"], "tags": ["B-TURN", "I-TURN"]},
#     {"tokens": ["Check", "the", "inventory"], "tags": ["B-LOOK", "O", "O"]},
#     {"tokens": ["Rotate", "180", "degrees"], "tags": ["B-TURN", "O", "O"]},
#     {"tokens": ["Run", "to", "the", "exit"], "tags": ["B-MOVE", "O", "O", "O"]},
#     {"tokens": ["Do", "you", "see", "that"], "tags": ["O", "O", "B-LOOK", "O"]},
# ]

# Create ID mappings
# label_list = ["O", "B-LOOK", "I-LOOK", "B-MOVE", "I-MOVE", "B-TURN", "I-TURN"]

# Load from file
with open("synthetic_data.json", "r") as f:
    data = json.load(f)

# # Ensure your label_list matches the tags used in the generator!
label_list = ["O", "B-LOOK", "I-LOOK", "B-MOVE", "I-MOVE", "B-TURN", "I-TURN"]

label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}

# Convert string tags to IDs in the data
for item in data:
    item["ner_tags"] = [label2id[tag] for tag in item["tags"]]

# Create HuggingFace dataset
raw_dataset = Dataset.from_list(data)
train_test_split = raw_dataset.train_test_split(test_size=0.2)
dataset = train_test_split

# Tokenization and alignment
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their original word
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                # Special ID: PyTorch ignores this during loss calculation
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # First token of the word gets the label
                label_ids.append(label[word_idx])
            else:
                # Sub-tokens get ignored (or you can set to I-TAG)
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# Load model
model = DistilBertForTokenClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
).to(device)

# Trainer setup
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

args = TrainingArguments(
    output_dir="./model/ner_results",
    # evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8, # Keep low for Mac
    per_device_eval_batch_size=8,
    num_train_epochs=15, # NER needs more epochs usually
    weight_decay=0.01,
    # use_mps_device=True # Force Mac GPU
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

print("Starting training...")
trainer.train()

print("Saving model...")
model.save_pretrained("./model/my_ner_model")
tokenizer.save_pretrained("./model/my_ner_model")
print("Saved to ./model/my_ner_model")

###

# model_path = "./model/my_ner_model"
# output_path = "./model/onnx_export"

# model = ORTModelForTokenClassification.from_pretrained(model_path)
# tokenizer = AutoTokenizer.from_pretrained(model_path)

# # model.save_pretrained(output_path)
# model.save_pretrained(
#     output_path,
#     save_as_external_path=True,
#     all_tensors_to_one_file=True,
#     external_data_format="weights.onnx.data",
# )
# tokenizer.save_pretrained(output_path)

# onnx_model = onnx.load(output_path + "/model.onnx")
# final_onnx_path = os.path.join(output_path, "model.onnx")
# data_file_name = "model.onnx.data"

# onnx.save_model(
#     onnx_model,
#     final_onnx_path,
#     save_as_external_data=True,
#     all_tensors_to_one_file=True,
#     location=data_file_name,
#     size_threshold=1024,
#     convert_attribute=False,
# )

# print("Saved to ", final_onnx_path)
