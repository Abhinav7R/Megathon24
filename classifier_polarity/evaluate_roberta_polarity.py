import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from deep_translator import GoogleTranslator
from langdetect import detect, DetectorFactory

# Fix random seed for consistency in language detection
DetectorFactory.seed = 0

def translate_to_english(text, source_lang):
    """Translate given text to English from the specified source language."""
    translated = GoogleTranslator(source=source_lang, target='en').translate(text)
    return translated

def load_model(model_path):
    """Load the model directly using torch.load."""
    model = torch.load(model_path)
    model.eval()  # Set the model to evaluation mode
    return model

def predict(model, tokenizer, text, max_length=128):
    """Predict the polarity of the given text."""
    # Tokenize and encode the input text
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    # Perform prediction
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).item()  # Get the predicted class

    return preds

def main():
    model_path = "polarity_classifier_new.pt"  # Path to your saved model
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')  # Load the tokenizer
    model = load_model(model_path)

    while True:
        text = input("\nEnter text for polarity prediction (or type 'exit' to quit):\n")
        if text.lower() == 'exit':
            break
        
        # Detect language
        detected_language = detect(text)
        print(f"\nDetected language: {detected_language}")

        # Translate the text to English if it's not already in English
        if detected_language != 'en':
            text = translate_to_english(text, detected_language)
            print(f"Translated text: {text}")

        prediction = predict(model, tokenizer, text)
        
        # Map the predicted index back to polarity values
        if prediction == 2:
            polarity = -1
        elif prediction == 0:
            polarity = 0
        elif prediction == 1:
            polarity = 1
        
        print(f"\nPredicted polarity: {polarity}\n")

if __name__ == "__main__":
    main()
