### Project ###

import argparse
import torch
import torchvision.transforms as transforms
from flask import Flask, jsonify, request
from PIL import Image
import io
from model import MovieNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

app = Flask(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default = 'models/Movies.pth', help='model path')
args = parser.parse_args()
model_path = args.model_path

model = MovieNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Define the genre labels
'''
genre_labels = [
    "action", "animation", "comedy","documentary", "drama", "fantasy", "horror", "romance", 
    "science Fiction", "thriller"
]
'''

transform = transforms.Compose([
    transforms.Resize((224, 224)), ### Taille à vérifier
    transforms.Lambda(lambda img: img.convert('L') if img.mode == 'RGB' else img), ####
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@app.route('/predict', methods=['POST'])
def predict():
    img_binary = request.data
    img_pil = Image.open(io.BytesIO(img_binary))

    # Transform the PIL image
    tensor = transform(img_pil).to(device)
    tensor = tensor.unsqueeze(0)  
    
    # Make prediction
    with torch.no_grad():
        outputs = model(tensor)
        #_, predicted = outputs.max(1)
        _, predicted = outputs

    # Map the prediction to a genre
    #genre = genre_labels[int(predicted[0])]         ###new line


    #return jsonify({"prediction": (predicted[0])})
    #return jsonify({"prediction": genre})
    return jsonify({"prediction": predicted})

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    # Get the image data from the request
    images_binary = request.files.getlist("images[]")

    tensors = []

    for img_binary in images_binary:
        img_pil = Image.open(img_binary.stream)
        tensor = transform(img_pil)
        tensors.append(tensor)

    # Stack tensors to form a batch tensor
    batch_tensor = torch.stack(tensors, dim=0)

    # Make prediction
    with torch.no_grad():
        outputs = model(batch_tensor)
        #_, predictions = outputs.max(1)
        _, predicted = outputs

        # Map the predictions to genres
        #genres = [genre_labels[int(pred)] for pred in predictions]

    #return jsonify({"predictions": predictions.tolist()})
    #return jsonify({"prediction": genres})
    return jsonify({"prediction": predicted})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

