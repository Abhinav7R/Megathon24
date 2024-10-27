import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score,recall_score,precision_score,accuracy_score
import csv
import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel
import csv
import random
# Training loop
from tqdm import tqdm
import torch.nn.functional as F
import torch

class MultiLabelDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        label_tensor = torch.tensor(label, dtype=torch.long) 

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label_tensor # Convert to float for multi-label
        }
        

emotion_mapping = {
    "joy": 0,
    "sadness": 1,
    "anger": 2,
    "surprise": 3,
    "love": 4
}

mapping_intensity = {
    0: "No",
    1: "Barely noticeable.",
    2: "Mild",
    3: "Moderate",
    4: "Significant",
    5: "Noticeable",
    6: "High",
    7: "Very high",
    8: "Severe",
    9: "Extreme",
    10: "Crisis level"
}

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

inputs = []
targets = []

unique_emotions = set()
unique_categories = set()

model_name = 'roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=90, problem_type="multi_label_classification")



train_inputs, train_targets = [], []
test_inputs, test_targets = [], []

# Load the dataset
with open('dataset.csv', newline='', encoding='utf-8') as csvfile:
    reader = list(csv.DictReader(csvfile))
    
    # Shuffle the rows for random splitting
    random.shuffle(reader)
    
    # Calculate split index for 80-20 split
    split_index = int(0.8 * len(reader))

    # Loop through the rows with an index to differentiate train and test sets
    for idx, row in enumerate(reader):
        emotion_mapped = emotion_mapping.get(row['emotion'], -1)  # Use -1 if emotion not in mapping
        category_mapped = category_mapping.get(row['category'], -1)  # Use -1 if category not in mapping
        texts = row['text']
        emotions = row['emotion']
        input_data = texts + emotions

        # Calculate the one-hot vector index
        one_hot_length = 90
        index = int(row['intensity']) + 10 * category_mapped

        # Initialize one-hot vector and set target index to 1 if within bounds
        if 0 <= index < one_hot_length:
            one_hot_vector = [0] * one_hot_length
            one_hot_vector[index] = 1
        else:
            print(f"Index {index} is out of bounds for a one-hot vector of length {one_hot_length}.")
            continue

        # Append to training or testing set based on index
        if idx < split_index:
            inputs.append(input_data)
            targets.append(one_hot_vector)
        else:
            test_inputs.append(input_data)
            test_targets.append(one_hot_vector)

print("Train size:", len(train_inputs), "Test size:", len(test_inputs))

model_name = 'roberta-base'

dataset = MultiLabelDataset(inputs,targets, tokenizer)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
val_dataset = MultiLabelDataset(inputs,targets, tokenizer)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)
intensity_loss_fn = torch.nn.MSELoss()  # For intensity prediction


def train_model(model, dataloader, num_epochs=5):
    model = model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        epoch_loss = 0
        batch_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}", leave=False)
        
        for batch in batch_bar:
            optimizer.zero_grad()

            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            loss = intensity_loss_fn(logits, labels.float())
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            
            # Update batch progress with the current batch loss
            batch_bar.set_postfix(loss=loss.item())
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(dataloader):.4f}")
        torch.save(model, "cat_int_classifier_new.pt")
        evaluate_model(model, val_dataloader)


from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_model(model, dataloader):
    model.eval()
    predictions_cat, true_labels_cat = [], []
    intensity_preds, intensity_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Reshape logits and labels to [batch_size, 9, 10] for 9 categories and 10 intensity levels
            logits_reshaped = logits.view(-1, 9, 10)
            labels_reshaped = labels.view(-1, 9, 10)

            # Calculate category predictions
            category_scores = logits_reshaped.sum(dim=-1)
            preds_cat = torch.argmax(category_scores, dim=-1)
            true_cat = torch.argmax(labels_reshaped.sum(dim=-1), dim=-1)

            # Collect category predictions and labels
            predictions_cat.extend(preds_cat.cpu().numpy())
            true_labels_cat.extend(true_cat.cpu().numpy())

            # Calculate intensity predictions for each category
            intensity_preds_batch = logits_reshaped[torch.arange(logits_reshaped.size(0)), preds_cat]
            intensity_labels_batch = labels_reshaped[torch.arange(labels_reshaped.size(0)), true_cat]

            # Collect intensity predictions and true labels for evaluation
            intensity_preds.extend(intensity_preds_batch.cpu().numpy())
            intensity_labels.extend(intensity_labels_batch.cpu().numpy())

    # Calculate category evaluation metrics
    f1 = f1_score(true_labels_cat, predictions_cat, average='macro')
    accuracy = accuracy_score(true_labels_cat, predictions_cat)
    precision = precision_score(true_labels_cat, predictions_cat, average='macro', zero_division=0)
    recall = recall_score(true_labels_cat, predictions_cat, average='macro', zero_division=0)

    print("Category Evaluation:")
    print(f"Validation F1 Score: {f1:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation Precision: {precision:.4f}")
    print(f"Validation Recall: {recall:.4f}")

    # Calculate intensity evaluation metrics (MSE and MAE)
    mse_intensity = mean_squared_error(intensity_labels, intensity_preds)
    mae_intensity = mean_absolute_error(intensity_labels, intensity_preds)

    print("\nIntensity Evaluation:")
    print(f"Mean Squared Error (MSE): {mse_intensity:.4f}")
    print(f"Mean Absolute Error (MAE): {mae_intensity:.4f}")


train_model(model, dataloader)

# model = torch.load('cat_int_classifier.pt')
# evaluate_model(model, val_dataloader)

