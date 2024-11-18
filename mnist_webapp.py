"""import gradio as gr
from PIL import Image
import requests
import io
#import numpy as np


def recognize_digit(image):
    # Convert to PIL Image necessary if using the API method
    image_array = image['composite']
    image = Image.fromarray(image_array.astype('uint8'))
    # Convert image to bytes
    img_binary = io.BytesIO()
    image.save(img_binary, format="PNG")
    # Send request to the API
    response = requests.post("http://127.0.0.1:5000/predict", data=img_binary.getvalue())
    #print("Predicted :", response.json()["prediction"])
    print(response)
    if response.status_code == 200:
        print("Predicted:", response.json().get("prediction"))
    else:
        print(f"Error {response.status_code}: {response.text}")
        # Parse and return the prediction from the API response
    
    if response.status_code == 200:
        prediction = response.json().get("prediction", "Error")
        print("Predicted:", prediction)
        return prediction
    else:
        print("Error in API call:", response.status_code)
        return "Error in prediction"

if __name__=='__main__':

    gr.Interface(fn=recognize_digit, 
                inputs="sketchpad", 
                outputs='label',
                live=True,
                description="Draw a number on the sketchpad to see the model's prediction.",
                ).launch(debug=True, share=True)"""

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
                inputs="sketchpad", 
                outputs='label',
                live=True,
                description="Draw a number on the sketchpad to see the model's prediction.",
                ).launch(debug=True, share=True);
