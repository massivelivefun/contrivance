import torch
import pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. SETUP MAC ACCELERATION (MPS)
# This checks if Metal Performance Shaders are available
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Training on: {device}")

# 2. PREPARE DATA
# Load your CSV
df = pd.read_csv('data.csv')

# Create numerical labels from text labels (LOOK -> 0, MOVE -> 1, etc.)
le = LabelEncoder()
df['label_id'] = le.fit_transform(df['label'])
label_map = dict(zip(le.classes_, le.transform(le.classes_)))
print(f"Label Mappings: {label_map}")

# Split into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(), df['label_id'].tolist(), test_size=0.2
)

# 3. TOKENIZATION
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

# Tokenize the data
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)

# Create a Torch Dataset
class CommandDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = CommandDataset(train_encodings, train_labels)
val_dataset = CommandDataset(val_encodings, val_labels)

# 4. LOAD MODEL
# num_labels must match the number of categories you have
model = DistilBertForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=len(label_map) 
).to(device)

# 5. TRAINING ARGUMENTS
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,              # Increase this if loss is still high
    per_device_train_batch_size=8,    # Smaller batch size for Mac memory efficiency
    per_device_eval_batch_size=8,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    use_mps_device=True               # Explicitly tell HuggingFace to use Mac MPS
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# 6. TRAIN AND SAVE
print("Starting training...")
trainer.train()

print("Saving model...")
model.save_pretrained("./my_command_model")
tokenizer.save_pretrained("./my_command_model")
print("Done! Model saved to ./my_command_model")
