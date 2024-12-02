import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from flask import Flask, jsonify, request
from PIL import Image
import io
from torchvision import models

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
    transforms.Normalize([0.5], [0.5])
])

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

        return jsonify({
            "predicted_class": int(predicted.item()),
            "class_name": genre_labels[predicted.item()],
            "probabilities": probabilities.squeeze(0).tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
