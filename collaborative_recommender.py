from sklearn.model_selection import train_test_split

import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import schedule
import time

import logging

class InteractionDataset(Dataset):
    def __init__(self, interactions):
        self.interactions = interactions
        self.user_ids = {id: idx for idx, id in enumerate(interactions['visitorid'].unique())}
        self.item_ids = {id: idx for idx, id in enumerate(interactions['itemid'].unique())}

        with open('user_ids.set', 'wb') as f:
            pickle.dump(self.user_ids, f)
        with open('item_ids.set', 'wb') as f:
            pickle.dump(self.item_ids, f)

        self.interactions['visitorid'] = self.interactions['visitorid'].map(self.user_ids)
        self.interactions['itemid'] = self.interactions['itemid'].map(self.item_ids)

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        user = self.interactions.iloc[idx, 1]
        item = self.interactions.iloc[idx, 3]
        rating = self.interactions.iloc[idx, 2]
        return torch.tensor(user, dtype=torch.long), torch.tensor(item, dtype=torch.long), torch.tensor(rating, dtype=torch.float)

class RecommenderNet(nn.Module):
    def __init__(self, num_users, num_items, embedding_size=50):
        super(RecommenderNet, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.item_embedding = nn.Embedding(num_items, embedding_size)
        self.fc1 = nn.Linear(embedding_size*2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, user, item):
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        x = torch.cat([user_emb, item_emb], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze()

logging.basicConfig(filename='recommender.log', level=logging.INFO)

def log_evaluation(loss, epoch):
    s = f'Epoch {epoch}: Loss = {loss}'
    print(s)
    logging.info(s)

def train(model, train_loader, criterion, optimizer, epochs=4):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for user, item, rating in train_loader:
            optimizer.zero_grad()
            outputs = model(user, item)
            loss = criterion(outputs, rating)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        log_evaluation(total_loss/len(train_loader), epoch)

# After adding new user_ids and new item_ids
def retrain_model(interactions, user_ids, item_ids, epochs=5):
    # Prepare the new dataset
    train_data, test_data = train_test_split(interactions, test_size=0.2, random_state=42)
    train_dataset = InteractionDataset(train_data)
    test_dataset = InteractionDataset(test_data)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    num_users = len(train_dataset.user_ids)
    num_items = len(train_dataset.item_ids)

    # Reinitialize the model
    model = RecommenderNet(num_users, num_items)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train(model, train_loader, criterion, optimizer, epochs)

    return model

if __name__ == '__main__':
    # Data cleaned in the "Data Exploration and Cleaning.ipynb"
    with open('events_df_cleaned.df', 'rb') as f:
        events = pickle.load(f)
    interaction_matrix = events.pivot_table(index='visitorid', columns='itemid', values='event', aggfunc='count').fillna(0)

    event_type_mapping = {'view': 1, 'addtocart': 5, 'transaction': 10} # Give proportial weights to interactivity
    events['event'] = events['event'].map(event_type_mapping)

    user_history = events.groupby('visitorid')['itemid'].agg(list).reset_index()

    train_data, test_data = train_test_split(events, test_size=0.2, random_state=42)

    train_dataset = InteractionDataset(train_data)
    test_dataset = InteractionDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    num_users = events['visitorid'].nunique()
    num_items = events['itemid'].nunique()

    model = RecommenderNet(num_users, num_items)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    train(model, train_loader, criterion, optimizer)

    def evaluate(model, test_loader, criterion):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for user, item, rating in test_loader:
                outputs = model(user, item)
                loss = criterion(outputs, rating)
                total_loss += loss.item()
        print(f'Test Loss: {total_loss/len(test_loader)}')
    evaluate(model, test_loader, criterion)

    def save_model(model, path='recommender_model.pth'):
        torch.save(model.state_dict(), path)

    save_model(model)

    print("Model saved to disk. Scheduling retrain at midnight everyday...")

    # Schedule the retraining every day at midnight
    schedule.every().day.at("00:00").do(retrain_model)

    while True:
        schedule.run_pending()
        time.sleep(100)
