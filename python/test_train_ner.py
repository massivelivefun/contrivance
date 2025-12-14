from transformers import pipeline

# Load your custom NER model
ner_pipeline = pipeline("ner", model="./my_ner_model", aggregation_strategy="simple", device="mps")

text = "Please turn right and look at the window"
results = ner_pipeline(text)

print(results)
