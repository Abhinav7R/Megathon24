import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from deep_translator import GoogleTranslator
from langdetect import detect, DetectorFactory

# Fix random seed for consistency in language detection
DetectorFactory.seed = 0

category_mapping = {
    "Anxiety": 0,
    'Career Confusion': 1,
    'Depression': 2,
    'Eating Disorder': 3,
    'Health Anxiety': 4,
    'Insomnia': 5,
    'Mixed Emotions': 6,
    'Positive Outlook': 7,
    'Stress': 8
}

def translate_to_english(text, source_lang):
    """Translate given text to English from the specified source language."""
    translated = GoogleTranslator(source=source_lang, target='en').translate(text)
    return translated

def load_model(model_path):
    """Load the model directly using torch.load."""
    model = torch.load(model_path)
    model.eval()  # Set the model to evaluation mode
    return model

lambda_threshold = 0.4

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
        logits_reshaped = logits.view(-1, 9, 10)
        category_scores = logits_reshaped.sum(dim=-1)
        preds = torch.argmax(category_scores, dim=-1)
        top3_scores, top3_indices = torch.topk(category_scores, k=3, dim=-1)  # Shape: [batch_size, 3]

        top3_argmax_values = []
        naveen_list = [] #lol
        
        for index in top3_indices[0]:
            chunk = logits_reshaped[0, index] 
            chunk = torch.nn.functional.softmax(chunk, dim=-1)
            argmax_value = torch.argmax(chunk).item()
            # print(argmax_value)
            if argmax_value > lambda_threshold:
                top3_argmax_values.append(argmax_value)
                naveen_list.append(index)

    return naveen_list,top3_argmax_values

def main():
    model_path = "cat_int_classifier_new.pt"  # Path to your saved model
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')  # Load the tokenizer
    model = load_model(model_path)

    while True:
        text = input("\nEnter text for Concern and Intensity prediction (or type 'exit' to quit):\n")
        if text.lower() == 'exit':
            break
        
        # Detect language
        detected_language = detect(text)
        print(f"\nDetected language: {detected_language}")

        # Translate the text to English if it's not already in English
        if detected_language != 'en':
            text = translate_to_english(text, detected_language)
            print(f"Translated text: {text}")

        prediction_top_cat,prediction_top_intensity = predict(model, tokenizer, text)

        # prediction_value=prediction.item()
        count=0
        for cat in prediction_top_cat:
            flag=0
            for key, value in category_mapping.items():
                if value == cat:
                    print(f"The predicted concern is",key)
                    print(f"The predicted intensity of the concern is",prediction_top_intensity[count])
                    count=count+1
                    flag=1
                    break
            if(flag==0):
                print(f"Could not predict the concern")
                        
if __name__ == "__main__":
    main()
