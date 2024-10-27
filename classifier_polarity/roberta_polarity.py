import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import csv

# Define the custom dataset class
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
            'labels': label_tensor
        }

# Initialize lists to store extracted values
text_list = []
polarity_list = []

# Read the CSV file without limiting the number of entries
with open('dataset.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        text_list.append(row['text'])
        
        # Map the polarity values: 0 -> 0, 1 -> 1, -1 -> 2
        polarity = float(row['polarity']) if row['polarity'] is not None else None
        if polarity == -1:
            polarity_list.append(2)  # Map -1 to index 2
        elif polarity == 0:
            polarity_list.append(0)  # Keep 0 as is
        elif polarity == 1:
            polarity_list.append(1)  # Keep 1 as is

# Load the tokenizer and model
model_name = 'roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=3, problem_type="multi_label_classification")

# Prepare the dataset
dataset = MultiLabelDataset(text_list, polarity_list, tokenizer)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Prepare dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Training loop with validation
def train_model(model, train_dataloader, val_dataloader, num_epochs=3):
    model = model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        for batch in train_dataloader:
            optimizer.zero_grad()

            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

        # Validation after each epoch
        evaluate_model(model, val_dataloader)

    torch.save(model, "polarity_classifier_new.pt")

# Evaluation function
def evaluate_model(model, dataloader):
    model = model.eval()  # Set model to evaluation mode
    predictions, true_labels = [], []

    with torch.no_grad():  # Disable gradient calculations for evaluation
        for batch in dataloader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Use argmax to get class predictions
            preds = torch.argmax(logits, dim=1)

            predictions.append(preds)
            true_labels.append(labels)

    # Concatenate predictions and true labels for scoring
    predictions = torch.cat(predictions).cpu().numpy()
    true_labels = torch.cat(true_labels).cpu().numpy()

    # Calculate F1 score
    f1 = f1_score(true_labels, predictions, average='macro')
    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predictions)
    # Calculate precision
    precision = precision_score(true_labels, predictions, average='macro', zero_division=0)
    # Calculate recall
    recall = recall_score(true_labels, predictions, average='macro', zero_division=0)

    print(f"Validation F1 Score: {f1:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation Precision: {precision:.4f}")
    print(f"Validation Recall: {recall:.4f}")

# Train the model with validation
train_model(model, train_dataloader, val_dataloader)
