### Project ###

import gradio as gr
from PIL import Image
import requests
import io


def recognize_digit(image):
    print(image)
    image = Image.fromarray(image['composite'].astype('uint8'))
    img_binary = io.BytesIO()
    image.save(img_binary, format="PNG")
    # Send request to the API
    response = requests.post("http://127.0.0.1:5000/predict", data=img_binary.getvalue())
    return response.json()["prediction"]

if __name__=='__main__':

    gr.Interface(fn=recognize_digit, 
                inputs="image", 
                outputs='label',
                live=True,
                description="Upload a movie poster image to predict its genre.",
                ).launch(debug=True, share=True)
