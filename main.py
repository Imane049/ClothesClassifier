from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import requests
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Load the model
model_path = os.path.join(os.path.dirname(__file__), 'keras_model.h5')
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Class labels
class_labels = ['Chaussures', 'Pantalon', 'Veste', 'T-shirt']

@app.route('/predict', methods=['POST'])
def predict():
    # Ensure model is loaded
    if model is None:
        return jsonify({'error': 'Model not loaded properly'}), 500

    # Get image URL or file
    if 'url' in request.form:
        image_url = request.form['url']
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content))
        except requests.exceptions.RequestException as e:
            return jsonify({'error': 'Failed to download image from URL', 'details': str(e)}), 400
    elif 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        try:
            img = Image.open(file)
        except Exception as e:
            return jsonify({'error': 'Failed to read image file', 'details': str(e)}), 500
    else:
        return jsonify({'error': 'No image found. Provide a file or a URL.'}), 400

    # Preprocess and predict
    try:
        img = img.resize((224, 224))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)
        predicted_class = class_labels[np.argmax(prediction)]

        return jsonify({'prediction': predicted_class})
    except Exception as e:
        return jsonify({'error': 'Prediction failed', 'details': str(e)}), 500

if __name__ == '__main__':
    # Set to listen on the specified port, for compatibility with deployment
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
