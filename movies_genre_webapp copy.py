import gradio as gr
from PIL import Image
import requests
import io
import os

# Genre labels
genre_labels = [
    "action", "animation", "comedy", "documentary", "drama",
    "fantasy", "horror", "romance", "science fiction", "thriller"
]

# API URL
API_URL = os.getenv("MOVIE_GENRE_API_URL", "http://movies_genre_api:5000/predict")

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

    except requests.exceptions.RequestException as e:
        return f"Error connecting to the API: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"


# Gradio interface
iface_poster = gr.Interface(
    fn=predict_genre,
    inputs=gr.Image(type="pil"),
    outputs="text",
    live=True,
    description="Upload a movie poster to predict its genre."
)

# Launch Gradio interface
if __name__ == '__main__':
    iface_poster.launch(server_name="0.0.0.0", server_port=7860, debug=True)
