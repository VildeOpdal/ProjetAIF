import gradio as gr
from PIL import Image
import requests
import io

# Genre labels
genre_labels = [
    "action", "animation", "comedy", "documentary", "drama",
    "fantasy", "horror", "romance", "science fiction", "thriller"
]

API_URL = "http://movies_genre_api:5000/predict"  # Docker
# API_URL = "http://127.0.0.1:5000/predict"       # local

# API request function
def predict_genre(image):
    try:
        # Convert image to binary for API request
        img_binary = io.BytesIO()
        image.save(img_binary, format="PNG")

        # Send request to Flask API
        response = requests.post(API_URL, data=img_binary.getvalue())
        response.raise_for_status()

        # Parse API response
        result = response.json()
        class_name = result["class_name"]
        probabilities = result["probabilities"]

        # Format result for display
        formatted_probabilities = "\n".join(
            [f"{genre_labels[i]}: {prob:.2%}" for i, prob in enumerate(probabilities)]
        )
        return f"Predicted Genre: {class_name}\n\nProbabilities:\n{formatted_probabilities}"

    except Exception as e:
        return f"Error: {e}"

# Gradio interface
if __name__ == '__main__':
    gr.Interface(
        fn=predict_genre,
        inputs=gr.Image(type="pil"),
        outputs="text",
        live=True,
        description="Upload a movie poster to predict its genre."
    ).launch(debug=True, share=True)
import gradio as gr
from PIL import Image
import requests
import io

# Genre labels
genre_labels = [
    "action", "animation", "comedy", "documentary", "drama",
    "fantasy", "horror", "romance", "science fiction", "thriller"
]

# API request function
def predict_genre(image):
    try:
        # Convert image to binary for API request
        img_binary = io.BytesIO()
        image.save(img_binary, format="PNG")

        # Send request to Flask API
        response = requests.post("http://127.0.0.1:5000/predict", data=img_binary.getvalue())
        response.raise_for_status()

        # Parse API response
        result = response.json()
        class_name = result["class_name"]
        probabilities = result["probabilities"]

        # Format result for display
        formatted_probabilities = "\n".join(
            [f"{genre_labels[i]}: {prob:.2%}" for i, prob in enumerate(probabilities)]
        )
        return f"Predicted Genre: {class_name}\n\nProbabilities:\n{formatted_probabilities}"

    except Exception as e:
        return f"Error: {e}"

# Gradio interface
if __name__ == '__main__':
    gr.Interface(
        fn=predict_genre,
        inputs=gr.Image(type="pil"),
        outputs="text",
        live=True,
        description="Upload a movie poster to predict its genre."
    ).launch(debug=True, share=True)
