'''
The code loads a pre-trained T5 model and tokenizer for emotion detection and defines a function to detect emotion from 
a given sentence. It then reads up sentences from dataset.csv, uses the function to classify each sentence's emotion and 
prints the sentence with its detected emotion.
'''

import csv
import warnings
from transformers import T5Tokenizer, T5ForConditionalGeneration

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

tokenizer = T5Tokenizer.from_pretrained("mrm8488/t5-base-finetuned-emotion")
model = T5ForConditionalGeneration.from_pretrained("mrm8488/t5-base-finetuned-emotion")

# Function to get the emotion of a text
def get_emotion(text):
    input_ids = tokenizer.encode(text + '</s>', return_tensors='pt')
    output = model.generate(input_ids=input_ids, max_length=2)
    dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output]
    label = dec[0]
    return label

input_file = 'dataset.csv'

with open(input_file, mode='r', newline='') as infile:
    reader = csv.reader(infile)
    
    for i, row in enumerate(reader):
        if i >= 2:  
            break
        
        sentence = row[0]
        emotion_label = get_emotion(sentence)
        
        print(f"Sentence: {sentence}")
        print(f"Emotion detected: {emotion_label}")
        print()
