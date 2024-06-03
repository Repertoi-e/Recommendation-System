import pandas as pd

from flask import Flask, request, jsonify
import pickle
import torch

import time
import random

from collaborative_recommender import RecommenderNet
from content_based_cosine_sim import get_content_based_recommendations

def load_cosine_sim(path='cosine_sim.pkl'):
    with open(path, 'rb') as f:
        cosine_sim = pickle.load(f)
    return cosine_sim

def load_model(path, num_users, num_items):
    model = RecommenderNet(num_users, num_items)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

app = Flask(__name__)

#
# Load the model
#
with open('user_ids.set', 'rb') as f:
    user_ids = pickle.load(f)

with open('events_df_cleaned.df', 'rb') as f:
    events = pickle.load(f)
user_history = events.groupby('visitorid')['itemid'].agg(list).reset_index()

num_users = events['visitorid'].nunique()
num_items = events['itemid'].nunique()

model = load_model('recommender_model.pth', num_users, num_items)
cosine_sim = load_cosine_sim()

category_tree = pd.read_csv('retail_rocket/category_tree.csv')

# Create a mapping from item IDs to indices
with open('item_id_to_idx.dict', 'rb') as f:
    item_id_to_idx = pickle.load(f)
idx_to_item_id = {idx: item_id for item_id, idx in item_id_to_idx.items()}

def predict_interactivity(model, user_id, item_id, user_ids, item_ids):
    if user_id not in user_ids or item_id not in item_ids:
        return "User or item not found in training data."
    
    user_idx = torch.tensor([user_ids[user_id]], dtype=torch.long)
    item_idx = torch.tensor([item_ids[item_id]], dtype=torch.long)
    
    model.eval()
    with torch.no_grad():
        prediction = model(user_idx, item_idx).item()
    
    return prediction

def recommend_future_items(user_id, user_history, user_ids, item_ids, num_recommendations=10):
    user_items = user_history[user_history.visitorid == user_id].itemid.to_list()
    print(user_items)
    if not user_items or len(user_items) == 0:
        user_items = None
    else:
        user_items = user_items[0]

    # Use collaborative filtering for users with history
    collaborative_recommendations = []
    if user_items:
        for item_id in item_ids:
            if item_id not in user_items:
                interactivity = predict_interactivity(model, user_id, item_id, user_ids, item_ids)
                collaborative_recommendations.append((item_id, interactivity))
        collaborative_recommendations = sorted(collaborative_recommendations, key=lambda x: x[1], reverse=True)
        collaborative_recommendations = [item[0] for item in collaborative_recommendations[:num_recommendations]]

    # Get content-based recommendations
    content_based_recommendations = []
    if user_items:
        # Pick a random item from the user's history to base content recommendations on
        random_item_id = random.choice(user_items)
        content_based_recommendations = get_content_based_recommendations(random_item_id, cosine_sim, item_id_to_idx, idx_to_item_id)
    else:
        # Use a random item id if the user has no history
        random_item_id = random.choice(list(item_ids.keys()))
        content_based_recommendations = get_content_based_recommendations(random_item_id, cosine_sim, item_id_to_idx, idx_to_item_id)
    
    # Get items from the same parent category
    parent_category_recommendations = []
    if user_items:
        for item_id in user_items:
            parent_category = category_tree[category_tree['categoryid'] == item_id]['parentid'].values
            if parent_category.size > 0:
                parent_category_id = parent_category[0]
                sibling_items = category_tree[category_tree['parentid'] == parent_category_id]['categoryid'].values
                sibling_items = [item for item in sibling_items if item not in user_items and item in item_ids]
                parent_category_recommendations.extend(sibling_items)

    # Prioritize collaborative recommendations, sprinkle in content-based and parent category recommendations
    combined_recommendations = collaborative_recommendations
    combined_recommendations.extend([item for item in content_based_recommendations if item not in combined_recommendations])
    combined_recommendations.extend([item for item in parent_category_recommendations if item not in combined_recommendations])

    # Return the top N recommendations
    return combined_recommendations[:num_recommendations]

@app.route('/recommend', methods=['POST'])
def recommend():
    start_time = time.time()
    
    data = request.json
    user_id = data['user_id']
    
    # Generate recommendations
    recommendations = recommend_future_items(user_id, user_history, user_ids, item_id_to_idx)
    
    end_time = time.time()
    response_time = end_time - start_time
    
    return jsonify({
        'recommendations': [int(rec) for rec in recommendations],
        'response_time': response_time
    })

if __name__ == '__main__':
    app.run(debug=True)