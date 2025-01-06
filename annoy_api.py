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

# Initialisation de l'application Flask
app = Flask(__name__)

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

# Routes API
@app.route('/')
def home():
    """Page d'accueil simple pour vérifier le fonctionnement du serveur."""
    return 'Bienvenue sur le système de recommandation de films!'

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
        

# Lancement de l'application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
