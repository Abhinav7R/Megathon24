import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Function to extract concern using hard prompts
def extract_concern(statement):
    prompt = f"Extract the specific concern related to mental health from the following statement: '{statement}'. Concern: "
    
    # Encode the input prompt
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    # Generate the output
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=40, num_return_sequences=1)

    # Decode the generated output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Find the part after 'Concern:'
    if "Concern:" in generated_text:
        # Split the text at 'Concern:' and take the second part
        concern_part = generated_text.split("Concern:")[1]
#         print(concern_part)
        # Further split by sentence and take the first sentence
        first_sentence = concern_part.split('.')[0].strip()  # Get the first sentence before the period
        print(first_sentence)
        return first_sentence
    
    return ""  # Return empty if 'Concern:' is not found

# Example statements
statements = [
    "Sometimes, I think I'm feeling very low.",
    "I've been doubting if I made the right career choice.",
    "I am feeling much better these days",
    "It's been hard, I am feeling hopeful.",
    "Sometimes, I think I'm happy and excited."
]

# Extract concerns
extracted_concerns = [extract_concern(statement) for statement in statements]

# Print extracted concerns
# for statement, concern in zip(statements, extracted_concerns):
#     print(f"Statement: {statement}\nExtracted Concern: {concern}\n")

def convert_to_phrase(sentence):
    # Words and phrases to remove
    remove_words = ["I'm", "I am", "am", "I think", "I", "I've", "sometimes", "very", "if", "the"]
    # Split the sentence into words
    words = sentence.split()
    
#     print(words)
    
    # Remove unwanted words and anything after 'if'
    phrase_words = []
    for word in words:
        if word in remove_words:
            continue
        # Stop adding words if we encounter 'if'
        if word.lower() == "if":
            break
        phrase_words.append(word)

    # Join the remaining words to form the phrase
    phrase = ' '.join(phrase_words).strip()
    return phrase

# Print extracted concerns
for statement, concern in zip(statements, extracted_concerns):
    print(f"Statement: {statement}")
#     print(concern)
    print(convert_to_phrase(concern))
    print("\n")