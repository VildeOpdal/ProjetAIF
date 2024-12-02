### Project ###

API_URL = "http://movies_genre_api:5000"

import gradio as gr
from PIL import Image
import requests
import io

genre_labels = [
    "action", "animation", "comedy","documentary", "drama", "fantasy", "horror", "romance", 
    "science Fiction", "thriller"
]

def recognize_digit(image):
    print(image)
    image = Image.fromarray(image.astype('uint8'))
    img_binary = io.BytesIO()
    image.save(img_binary, format="PNG")
    img_binary.seek(0)
    # Send request to the API
    response = requests.post(API_URL, files={"file": img_binary})
    # response = requests.post("http://127.0.0.1:5000/predict", data=img_binary.getvalue())
    print(response)
    return genre_labels[response.json()["prediction"][0]]

if __name__=='__main__':

    gr.Interface(fn=recognize_digit, 
                inputs="image", 
                outputs='label',
                live=True,
                description="Upload a movie poster image to predict its genre.",
                ).launch(debug=True, share=True)
