import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from flask import Flask, jsonify, request
from PIL import Image
import io
from torchvision import models
import logging

# Setup logging
logging.basicConfig(level=logging.ERROR)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Flask app initialization
app = Flask(__name__)

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
args = parser.parse_args()

# Load the trained model
model = models.resnet50(pretrained=False)
num_classes = 10  # Adjust based on your dataset
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(args.model_path, map_location=device))
model = model.to(device)
model.eval()

# Genre labels
genre_labels = [
    "action", "animation", "comedy", "documentary", "drama",
    "fantasy", "horror", "romance", "science fiction", "thriller"
]

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Standard for ResNet50
])


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Application Flask pour la recommandation de films

"""
# Importations des bibliothèques nécessaires
from flask import Flask, request, jsonify
from annoy import AnnoyIndex
import pandas as pd
from PIL import Image
import io
import torchvision.transforms as transforms
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import SnowballStemmer
from nltk import word_tokenize
import re
nltk.download('punkt')
nltk.download('stopwords')



# Chargement des données et des modèles
print("Chargement des bases de données Annoy...")
annoy_db = AnnoyIndex(576, metric='angular')  # Correspond à la dimension des vecteurs
annoy_db.load('TabIndex.ann')  # Chemin vers la base Annoy pour les affiches

annoy_bert = AnnoyIndex(768, metric='angular')  # Dimensions pour Bert
annoy_bert.load('annoy_bert.ann')

annoy_tfidf = AnnoyIndex(200, metric='angular')  # Dimensions pour TF-IDF
annoy_tfidf.load('annoy_tfidf.ann')

movies_metadata = pd.read_csv('movies_metadata.csv', low_memory=False)  # Métadonnées des films

# Fonctions auxiliaires
def get_movie_titles_from_indices(indices):
    """Récupère les titres de films associés aux indices."""
    return movies_metadata.loc[indices, 'title'].tolist()

# Root endpoint
@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Movie Genre Prediction API!"})

# API endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read and preprocess image
        img_binary = request.data
        img = Image.open(io.BytesIO(img_binary)).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)

        # Perform prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

        # Format probabilities
        probabilities = probabilities.squeeze(0).tolist()
        rounded_probabilities = [round(p, 4) for p in probabilities]

        return jsonify({
            "predicted_class": int(predicted.item()),
            "class_name": genre_labels[predicted.item()],
            "probabilities": rounded_probabilities
        })
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500
    
@app.route('/reco_poster', methods=['POST'])
def reco_poster():
    """Recommandation basée sur une affiche de film."""
    try:
        vector = request.json['vector']
        print("Vector reçu pour la recommandation : ", vector)
        
        closest_indices = annoy_db.get_nns_by_vector(vector[0], 3)
        print("Indices recommandés : ", closest_indices)
        
        return jsonify(closest_indices)
    except Exception as e:
        print("Erreur dans reco_poster : ", e)
        return jsonify({'error': str(e)}), 500

@app.route('/reco_texte', methods=['POST'])
def reco_texte():
    #Recommandation basée sur une description textuelle.
    try:
        vector = request.json['vector']
        method_choice = request.json.get('method_choice', 'Bert') # Méthode par défaut : Bert
        print(f"Méthode sélectionnée : {method_choice}")

        if method_choice == 'Bert':
            closest_indices = annoy_bert.get_nns_by_vector(vector, 5)
        elif method_choice == 'TFIDF':
            closest_indices = annoy_tfidf.get_nns_by_vector(vector, 5)
        else:
            return jsonify({'error': 'Méthode non supportée'}), 400
        reco = get_movie_titles_from_indices(closest_indices)
        print("Titres recommandés : ", reco)
        return jsonify(reco)
    except Exception as e:
            print("Erreur dans reco_texte : ", e)
    return jsonify({'error': str(e)}), 500
        
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
