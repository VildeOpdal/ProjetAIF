"""
Application Flask pour la recommandation de films

"""
# Importations nécessaires
import gradio as gr
import requests
#import random
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import torch
#from torchvision import datasets
#from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from model_3 import model_3
from model_2 import model_2
import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from transformers import  DistilBertTokenizerFast #DistilBertModel,
import numpy as np
#from annoy import AnnoyIndex
#from flask import Flask, jsonify, request, send_from_directory
#import io
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.stem import SnowballStemmer
import nltk
from nltk.corpus import stopwords
import re


# Chargement des données 
df = pd.read_csv('DF_path.csv')

# Normalisation des images via des transformations prédéfinies
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean, std)
inv_normalize = transforms.Normalize(
    mean=[-m / s for m, s in zip(mean, std)],
    std=[1 / s for s in std]
)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

	
# Prétraitement d'une image pour obtenir un vecteur représentatif
def process_image(image):
	    
    image_transfo = transform(image)
    vector = model_3(image_transfo.unsqueeze(0)).cpu().detach().numpy().tolist()

    response = requests.post('http://annoy-db:5001/reco_poster', json={'vector': vector})
    if response.status_code == 200:
        indices = response.json()
        paths = [str(df.path[i]) for i in indices]

        fig, axs = plt.subplots(1, len(paths), figsize=(5 * len(paths), 5))
        for i, path in enumerate(paths):
            img = Image.open(path)
            axs[i].imshow(img)
            axs[i].axis('off')
        return fig
    else:
        return plt.figure()

# Chargement des métadonnées de films
movies_metadata = pd.read_csv('movies_metadata.csv', low_memory=False)

### PARTIE 2 ###

# Tokenizer utilisant un stemmer
class StemTokenizer:
	ignore_tokens = [',', '.', ';', ':', '"', '``', "''", '`']

 
	def __init__(self):
		self.stemmer = SnowballStemmer('english')

 
	def __call__(self, doc):
		doc = doc.lower()
		return [self.stemmer.stem(t) for t in word_tokenize(re.sub("[^a-z' ]", "", doc)) if t not in self.ignore_tokens]

 
tokenizer = StemTokenizer()
tokenizer2 = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

 
# Extraction des embeddings DistilBERT
def get_distilbert_embeddings(text):
	if pd.isna(text):
		return np.zeros(model_2.config.dim).tolist()

 
	inputs = tokenizer2(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
	inputs = {key: value.cpu() for key, value in inputs.items()}
	with torch.no_grad():
		outputs = model_2(**inputs)

 
	return outputs.last_hidden_state.mean(dim=1).squeeze().detach().cpu().numpy().tolist()



# Embeddings TF-IDF


 
# Télécharger les ressources nécessaires
nltk.download('stopwords')
nltk.download('punkt')

 
# Initialisation des mots d'arrêt
stop_words = list(stopwords.words('english'))

 
# Embeddings TF-IDF
def get_tfidf_embeddings(text):
	"""
	Génère des embeddings TF-IDF pour un texte donné.
	"""
	# Initialisation du vectoriseur TF-IDF
	tfidf = TfidfVectorizer(stop_words=stop_words, max_features=200)
	 
	# Vérification et conversion du champ 'overview' en texte
	movies_metadata['overview'] = movies_metadata['overview'].fillna('').astype('U')
	 
	# Apprentissage du vocabulaire TF-IDF à partir des descriptions des films
	tfidf.fit(movies_metadata['overview'])
	 
	# Transformation du texte donné en vecteur
	vector = tfidf.transform([text])
	 
	# Retourner le vecteur en tant que liste plate
	return vector.toarray().flatten().tolist()



# Traitement des descriptions pour générer des recommandations
def process_description(description, method_choice):
	try:
		if method_choice == "Bert":
			vector = get_distilbert_embeddings(description)
		elif method_choice == "TFIDF":
			vector = get_tfidf_embeddings(description)
		else:
			return 'Méthode non supportée'

 
		response = requests.post('http://annoy-db:5001/reco_texte', json={'vector': vector, 'method_choice': method_choice})
		if response.status_code == 200:
			titles = response.json()
			return ", ".join(titles)
		else:
			return 'Erreur lors de la récupération des recommandations.'
	except Exception as e:
		return f'Erreur : {e}'

# Interfaces utilisateur via Gradio
iface_poster = gr.Interface(fn=process_image, inputs="image", outputs="plot",
                            title="Système de recommandation de films",
                            description="Faites glisser une affiche pour découvrir des films similaires.")

#iface_texte = gr.Interface(
#    fn=process_description,
#    inputs=[
#        gr.Textbox(placeholder="Saisissez une description..."),
#        gr.Dropdown(["Bert", "TFIDF"], label="Méthode")
#    ],
#    outputs="text"
#)

#demo = gr.TabbedInterface([iface_poster, iface_texte], ["Poster", "Description"])
#demo.launch(server_name="0.0.0.0")
