import pandas as pd
import numpy as np

import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def get_content_based_recommendations(item_id, cosine_sim, item_id_to_idx, idx_to_item_id):
    if item_id not in item_id_to_idx:
        print("Item ID not found in item properties.")
        return []
    
    idx = item_id_to_idx[item_id]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Get top 10 similar items, excluding itself
    item_indices = [i[0] for i in sim_scores]
    recommended_items = [idx_to_item_id[i] for i in item_indices if i in idx_to_item_id]
    return recommended_items

if __name__ == '__main__':
    item_properties1 = pd.read_csv('retail_rocket/item_properties_part1.csv')
    item_properties2 = pd.read_csv('retail_rocket/item_properties_part2.csv')
    item_properties = pd.concat([item_properties1, item_properties2], ignore_index=True)

    del item_properties1, item_properties2

    # Take just a few item properties to speed up training
    item_properties = item_properties[-10_000:]

    # Merge item properties into a single string per item
    item_properties['properties'] = item_properties.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

    # Use TF-IDF to vectorize the item properties
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(item_properties['properties'])

    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    item_id_to_idx = {item_id: idx for idx, item_id in enumerate(item_properties['itemid'].unique())}
    idx_to_item_id = {idx: item_id for item_id, idx in item_id_to_idx.items()}
    with open('item_id_to_idx.dict', 'wb') as f:
        pickle.dump(item_id_to_idx, f)

    item_id = 59
    print("Items similar to {item_id}: ", get_content_based_recommendations(item_id, cosine_sim, item_id_to_idx, idx_to_item_id))

    with open('cosine_sim.pkl', 'wb') as f:
        pickle.dump(cosine_sim, f)


def add_new_user(user_id, user_ids, interactions):
    if user_id not in user_ids:
        user_ids[user_id] = len(user_ids)
        # Add a new row with the new user and a dummy item to maintain consistency
        interactions = interactions.append({'visitorid': user_id, 'itemid': 0, 'event': 0, 'timestamp': pd.Timestamp.now()}, ignore_index=True)
    return user_ids, interactions

def add_new_item(item_id, item_ids, item_id_to_idx, idx_to_item_id, interactions, cosine_sim):
    if item_id not in item_ids:
        new_idx = len(item_ids)
        item_ids[item_id] = new_idx
        item_id_to_idx[item_id] = new_idx
        idx_to_item_id[new_idx] = item_id
        
        # Expand the cosine_sim matrix to accommodate the new item
        new_row = np.zeros((1, cosine_sim.shape[1]))
        new_col = np.zeros((cosine_sim.shape[0] + 1, 1))
        cosine_sim = np.vstack((cosine_sim, new_row))
        cosine_sim = np.hstack((cosine_sim, new_col))
        
        # Add a new row with the new item and a dummy user to maintain consistency
        interactions = interactions.append({'visitorid': 0, 'itemid': item_id, 'event': 0, 'timestamp': pd.Timestamp.now()}, ignore_index=True)

    return item_ids, item_id_to_idx, idx_to_item_id, interactions, cosine_sim