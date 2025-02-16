import gradio as gr
from PIL import Image
import requests
import io
import os
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision.transforms as transforms
#from sklearn.feature_extraction.text import TfidfVectorizer
#from nltk import word_tokenize
#from nltk.stem import SnowballStemmer
#from nltk.corpus import stopwords
#import re
#import nltk
#import numpy as np
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

from gradio_webapp import get_distilbert_embeddings, get_tfidf_embeddings
from model_3 import model_3 


# Genre labels
genre_labels = [
    "action", "animation", "comedy", "documentary", "drama",
    "fantasy", "horror", "romance", "science fiction", "thriller"
]

# API URL
API_URL = os.getenv("MOVIE_GENRE_API_URL", "http://localhost:5001")

# Load data
df = pd.read_csv('DF_path.csv')
#movies_metadata = pd.read_csv('movies_metadata.csv', low_memory=False)

# Normalization and transformations
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean, std)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

def process_image(image):
    image_transfo = transform(image)
    vector = model_3(image_transfo.unsqueeze(0)).cpu().detach().numpy().tolist()

    response = requests.post('http://localhost:5001/reco_poster', json={'vector': vector})
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

# Modèle d'anomalie : détecter si une image est un poster
transform2 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])
anomaly_model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
anomaly_model.eval() 
def is_movie_poster(image):
    image_tensor = transform2(image).unsqueeze(0)
    with torch.no_grad():
        output = anomaly_model(image_tensor)
    _, predicted = torch.max(output, 1)
    print(predicted.item())
    return predicted.item() == 917  # Hypothèse : classe 1 = poster


# Predict genre
def predict_genre(image):
    try:
        if not is_movie_poster(image):
            return "L'image fournie n'est pas un poster de film."
        img_binary = io.BytesIO()
        image.save(img_binary, format="PNG")

        response = requests.post(f"{API_URL}/predict", data=img_binary.getvalue())
        response.raise_for_status()

        result = response.json()
        class_name = result["class_name"]
        probabilities = result["probabilities"]

        formatted_probabilities = "\n".join(
            [f"{genre_labels[i]}: {prob:.2%}" for i, prob in enumerate(probabilities)]
        )
        return f"Predicted Genre: {class_name}\n\nProbabilities:\n{formatted_probabilities}"
    except requests.exceptions.RequestException as e:
        return f"Error connecting to the API: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

# Process description for recommendations
def process_description(description, method_choice):
    try:
        if method_choice == "Bert":
            vector = get_distilbert_embeddings(description)
        elif method_choice == "TFIDF":
            vector = get_tfidf_embeddings(description)
        else:
            return 'Méthode non supportée'

        response = requests.post(f"{API_URL}/reco_texte", json={'vector': vector, 'method_choice': method_choice})
        if response.status_code == 200:
            titles = response.json()
            return ", ".join(titles)
        else:
            return 'Erreur lors de la récupération des recommandations.'
    except Exception as e:
        return f'Erreur : {e}'

# Interface Gradio
iface_poster = gr.Interface(
    fn=process_image,
    inputs="image",
    outputs="plot",
    title="Recommandation par Affiche",
    description="Faites glisser une affiche pour découvrir des films similaires."
)

iface_genre = gr.Interface(
    fn=predict_genre,
    inputs=gr.Image(type="pil"),
    outputs="text",
    live=True,
    title="Prédiction de Genre",
    description="Upload a movie poster to predict its genre."
)

iface_description = gr.Interface(
    fn=process_description,
    inputs=[
        gr.Textbox(placeholder="Saisissez une description..."),
        gr.Dropdown(["Bert", "TFIDF"], label="Méthode")
    ],
    outputs="text",
    title="Recommandation par Description",
    description="Entrez une description textuelle pour obtenir des recommandations."
)

# Gradio Tabbed Interface
demo = gr.TabbedInterface(
    [iface_poster, iface_genre, iface_description],
    ["Poster", "Genre", "Description"]
)

# Lancement
if __name__ == '__main__':
    demo.launch(server_name="0.0.0.0", server_port=7860, debug=True)
