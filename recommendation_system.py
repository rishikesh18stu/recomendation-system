
import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from surprise import SVD # type: ignore
from surprise import Dataset # type: ignore
from surprise.model_selection import cross_validate # type: ignore
from surprise import Reader # type: ignore
from flask import Flask, request, jsonify # type: ignore

# Load dataset
data = pd.read_csv('ratings_Beauty.csv')

# 1. Content-Based Filtering
def content_based_filtering(user_recent_product_id):
    # Create a product catalog with 'ProductId'
    products = data[['ProductId']].drop_duplicates()

    # Add a dummy 'combined_features' column (since we lack product names and categories)
    products['combined_features'] = products['ProductId']

    # Convert text data into TF-IDF features
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(products['combined_features'])

    # Compute cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Find the index of the recent product
    product_idx = products.index[products['ProductId'] == user_recent_product_id].tolist()[0]

    # Get similarity scores for the recent product
    sim_scores = list(enumerate(cosine_sim[product_idx]))

    # Sort products by similarity score
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get top 10 similar products
    top_products = [products.iloc[i[0]].ProductId for i in sim_scores[1:11]]

    return top_products

# 2. Collaborative Filtering (Using Surprise Library)
def collaborative_filtering(user_id):
    # Create a subset of the dataset for collaborative filtering
    ratings_data = data[['UserId', 'ProductId', 'Rating']]

    # Prepare the data for Surprise
    reader = Reader(rating_scale=(1, 5))
    surprise_data = Dataset.load_from_df(ratings_data, reader)

    # Train-test split
    trainset = surprise_data.build_full_trainset()

    # Build and train the SVD model
    model = SVD()
    cross_validate(model, surprise_data, cv=5, verbose=True)
    model.fit(trainset)

    # Predict top products for the user
    all_products = data['ProductId'].unique()
    predictions = []
    for product_id in all_products:
        predictions.append((product_id, model.predict(user_id, product_id).est))

    # Sort by predicted rating
    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)

    # Get top 10 recommendations
    top_recommendations = [pred[0] for pred in predictions[:10]]

    return top_recommendations

app = Flask(__name__)

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    user_id = data['user_id']
    recent_product_id = data['recent_product_id']
    
    # Call your recommendation functions
    content_based = content_based_filtering(recent_product_id)
    collaborative = collaborative_filtering(user_id)
    
    return jsonify({
        'content_based': content_based,
        'collaborative': collaborative
    })

if __name__ == '__main__':
    app.run(debug=True)