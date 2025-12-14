from transformers import pipeline

# Load your locally trained model
classifier = pipeline("text-classification", model="./my_command_model", device="mps")

# Test it
commands = ["turn right quickly", "look at the sky", "I am happy today"]

results = classifier(commands)
print(results)
# Expected output: 
# [{'label': 'LABEL_3', 'score': 0.99}, ...] (You match LABEL_X back to your map)
