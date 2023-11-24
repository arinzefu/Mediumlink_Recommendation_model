import pandas as pd
import numpy as np

data = pd.read_csv('medium-data-science-articles-2020.csv')

data.head()

data.shape

df = data.drop('author_page', axis=1)

data = df.drop('reading_time', axis=1)

data.shape

tag_counts = data['tag'].value_counts()

tag_counts

missing_values = data.isnull().sum()

print("Missing Values Count per Column:")
print(missing_values)

data = data.fillna(0)

missing_values = data.isnull().sum()

print("Missing Values Count per Column:")
print(missing_values)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

# Load a pre-trained tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

data['text'] = (
    data['title'].astype(str) + " " +
    data['subtitle'].astype(str) + " " +
    data['tag'].astype(str) + " " +
    data['author'].astype(str)
)

from sklearn.preprocessing import StandardScaler

# Normalize numeric features
numeric_features = ['claps', 'responses']
scaler = StandardScaler()
data[numeric_features] = scaler.fit_transform(data[numeric_features])

text_embeddings = []

bert_model = BertModel.from_pretrained('bert-base-uncased')

for index, row in data.iterrows():
    # Tokenize the text
    tokenized = tokenizer(row['text'], padding="max_length", truncation=True, max_length=32, return_tensors='pt')

    input_ids = tokenized['input_ids']
    attention_mask = tokenized['attention_mask']

    # Forward pass through the BertModel
    with torch.no_grad():
        outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)

    # Extract the embeddings
    embeddings = outputs.last_hidden_state.mean(dim=1)

    # numeric features
    numeric_features_array = row[numeric_features].values.astype(float)
    numeric_features_tensor = torch.tensor(numeric_features_array, dtype=torch.float32)

    # Concatenate
    combined_embedding = torch.cat((embeddings.squeeze(), numeric_features_tensor))

    # Convert to numpy and append to the list
    text_embeddings.append(combined_embedding.numpy())

# Convert the list of embeddings to a single NumPy array
text_embeddings = np.array(text_embeddings)

# Convert the NumPy array to a torch tensor
text_embeddings = torch.tensor(text_embeddings)

text_embeddings

num_urls = len(data['url'].unique())

# Create a TensorDataset
dataset = TensorDataset(text_embeddings)

# Create a DataLoader
batch_size = 16
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# The model
class RecommendationModel(nn.Module):
    def __init__(self, input_size, num_urls):
        super(RecommendationModel, self).__init__()
        self.fc = nn.Linear(input_size, num_urls)

    def forward(self, embeddings):
        scores = self.fc(embeddings.squeeze(dim=1))
        return scores

device = torch.device("cuda")

# Create an instance of the model
input_size = text_embeddings.size(1)
recommendation_model = RecommendationModel(input_size, num_urls)

print(recommendation_model)

optimizer = torch.optim.AdamW(recommendation_model.parameters(), lr=0.001)

for batch_idx, batch in enumerate(dataloader):
    embeddings = batch

    print(f"Batch {batch_idx + 1} - Number of Embeddings: {len(embeddings)}")

    print("Embeddings (first few):", embeddings[:5])

    break

from tqdm import tqdm


# Training loop
num_epochs = 3

for epoch in range(num_epochs):
    recommendation_model.train()
    total_loss = 0.0


    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch')):
        # Extract embeddings
        embeddings = batch[0]

        # Forward pass
        scores = recommendation_model(embeddings)

        # Generate random negative samples
        negative_samples = torch.randint(high=num_urls, size=scores.shape, dtype=torch.long)

        # Calculate pairwise ranking loss
        loss = nn.MarginRankingLoss(margin=1.0)(scores, negative_samples, torch.ones_like(scores))

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate total loss
        total_loss += loss.item()


    average_loss = total_loss / (batch_idx + 1)
    print(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss:.4f}')

# save the model
torch.save(recommendation_model.state_dict(), 'recommendation_model.pth')

input_text = "learning python"

# Tokenize input text
tokenized = tokenizer(input_text, padding="max_length", truncation=True, max_length=32, return_tensors='pt')

# Extract input_ids and attention_mask
input_ids = tokenized['input_ids']
attention_mask = tokenized['attention_mask']

# Forward pass through the BertModel
with torch.no_grad():
    outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)

# Extract the embeddings
embeddings = outputs.last_hidden_state.mean(dim=1)

# Include numeric features
combined_embedding = embeddings.squeeze()

# Convert to NumPy and append to the list
text_embeddings_list = []
text_embeddings_list.append(combined_embedding.numpy())

# Convert the list of embeddings to a single NumPy array
text_embeddings = torch.tensor(text_embeddings_list)

text_embeddings

# Set the model to evaluation mode
recommendation_model.eval()

# Initialize variables for evaluation
total_loss = 0.0

recommended_urls = []
recommended_scores = []

with torch.no_grad():
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch')):
        # Extract embeddings
        embeddings = batch[0]

        # Forward pass
        scores = recommendation_model(embeddings)

        recommended_url_index = scores.argmax().item()

        recommended_url = data['url'].unique()[recommended_url_index]

        # Append recommended URL and its score to the lists
        recommended_urls.append(recommended_url)
        recommended_scores.append(scores[0].item())

        # Generate random negative samples
        negative_samples = torch.randint(high=num_urls, size=scores.shape, dtype=torch.long)

        # Calculate pairwise ranking loss
        loss = nn.MarginRankingLoss(margin=1.0)(scores, negative_samples, torch.ones_like(scores))

        total_loss += loss.item()

average_loss = total_loss / (batch_idx + 1)
print(f'Evaluation Loss: {average_loss:.4f}')

for url, score in zip(recommended_urls, recommended_scores):
    print(f'Recommended URL: {url}, Score: {score:.4f}')
