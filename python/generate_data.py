import random
import json

# 1. DEFINE YOUR VOCABULARY
# These are the words the model will learn to recognize.
vocab = {
    "MOVE_VERB": ["run", "walk", "go", "sprint", "crawl", "jog", "move", "travel"],
    "TURN_VERB": ["turn", "rotate", "spin", "face", "pivot", "swivel"],
    "LOOK_VERB": ["look", "stare", "gaze", "watch", "check", "examine", "scan"],
    
    "DIRECTION": ["left", "right", "forward", "backward", "up", "down", "around", "north", "south"],
    "OBJECT":    ["door", "window", "enemy", "box", "wall", "light", "player", "sky", "floor"],
    "DEGREE":    ["90 degrees", "180 degrees", "a little bit", "all the way"],
    
    # Filler words (Noise) - tagged as 'O'
    "FILLER_START": ["please", "could you", "hey", "can you", "just", "kindly"],
    "PREP":         ["at", "to", "towards", "over", "the"]
}

# 2. DEFINE TEMPLATES
# We use placeholders like {KEY|TAG}
# KEY = looks up word in vocab list above
# TAG = the label this word gets (B-TAG, I-TAG)
templates = [
    # Simple Commands
    "{MOVE_VERB|MOVE} {DIRECTION|O}",                       # "Run left"
    "{TURN_VERB|TURN} {DIRECTION|O}",                       # "Turn right"
    "{LOOK_VERB|LOOK} {PREP|O} {OBJECT|O}",                 # "Look at box"
    
    # Polite/Complex Commands
    "{FILLER_START|O} {MOVE_VERB|MOVE} {DIRECTION|O}",      # "Please go forward"
    "{FILLER_START|O} {TURN_VERB|TURN} {DEGREE|O}",         # "Hey rotate 90 degrees"
    "{TURN_VERB|TURN} {DIRECTION|O} and {LOOK_VERB|LOOK}",  # "Turn left and look"
    "{MOVE_VERB|MOVE} {PREP|O} {OBJECT|O}",                 # "Walk to door"
]

def generate_sentence(template):
    tokens = []
    tags = []
    
    # Split template by spaces to process word by word
    # (Note: simpler splitting for this demo; real NLP might need smarter tokenization)
    raw_parts = template.split()
    
    for part in raw_parts:
        # Check if this part is a placeholder like {KEY|TAG}
        if "{" in part and "}" in part:
            content = part.strip("{}")
            key, tag_type = content.split("|")
            
            # Pick a random word from the vocabulary list
            word_choice = random.choice(vocab[key])
            
            # Handle multi-word choices (e.g., "90 degrees")
            sub_words = word_choice.split()
            
            for i, sub_word in enumerate(sub_words):
                tokens.append(sub_word)
                
                # Assign Tags
                if tag_type == "O":
                    tags.append("O")
                else:
                    # If it's the first word, it's B-TAG (Beginning)
                    # If it's subsequent words, it's I-TAG (Inside)
                    if i == 0:
                        tags.append(f"B-{tag_type}")
                    else:
                        tags.append(f"I-{tag_type}")
                        
        # Handle static words like "and" in templates
        else:
            tokens.append(part)
            tags.append("O")
            
    return {"tokens": tokens, "tags": tags}

# 3. GENERATE MASSIVE DATASET
dataset_size = 2000 # How many sentences do you want?
generated_data = []

for _ in range(dataset_size):
    # Pick a random template
    temp = random.choice(templates)
    # Generate data
    entry = generate_sentence(temp)
    generated_data.append(entry)

# 4. SAVE TO FILE
output_file = "synthetic_data.json"
with open(output_file, "w") as f:
    json.dump(generated_data, f, indent=2)

print(f"Successfully generated {dataset_size} labeled sentences to {output_file}")
print("Sample output:")
print(json.dumps(generated_data[0], indent=2))
