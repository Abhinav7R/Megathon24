import google.generativeai as genai
import pandas as pd

# Configure the Google Generative AI API with your API key
genai.configure(api_key="AIzaSyBFJpYOFWXRUqIfmb_Q5uRA9BBEIYFjIgc")

def extract_emotional_phrases(text):
    # Create a GenerativeModel instance for the Gemini model
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Generate content to extract emotional phrases
    prompt = f"Extract emotional phrases from the following text: \"{text}\"."
    
    # Call the model to generate content
    response = model.generate_content(prompt)
    
    # Return the generated text
    return response.text.strip()

# Read sentences from CSV file
df = pd.read_csv('./dataset.csv')

i = 0
# Iterate over the sentences in the 'text' column
for sentence in df['text']:
    print(f"Sentence: '{sentence}'")
    emotional_phrases = extract_emotional_phrases(sentence)
    print(f"Emotional Phrases: {emotional_phrases}\n")
    if i==10:
        break