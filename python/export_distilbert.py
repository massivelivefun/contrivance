from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# load model
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = DistilBertForSequenceClassification.from_pretrained(model_name)
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

# create dummy input for tracing
dummy_input = tokenizer("This is a sample", return_tensors="pt")

# export
torch.onnx.export(
    model,
    (dummy_input['input_ids'], dummy_input['attention_mask']),
    "distilbert.onnx",
    input_names=['input_ids', 'attention_mask'],
    output_names=['logits'],
    dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence'},
                  'attention_mask': {0: 'batch_size', 1: 'sequence'},
                  'logits': {0: 'batch_size'}}
)
