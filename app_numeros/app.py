from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.models import load_model
import base64
import re
import os

app = Flask(__name__)

# Cargar el modelo solo cuando se use (evita errores al iniciar)
def get_model():
    model_path = os.path.join(os.path.dirname(__file__), 'model.h5')
    return load_model(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener y procesar imagen
        image_data = request.form['image']
        image_data = re.sub('^data:image/.+;base64,', '', image_data)
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        
        # Preprocesamiento
        image = image.resize((28, 28)).convert('L')
        image_array = 255 - np.array(image)
        image_array = image_array.reshape(1, 28, 28, 1).astype('float32') / 255
        
        # Predecir
        model = get_model()
        predicted_digit = np.argmax(model.predict(image_array))
        
        return jsonify({'digit': int(predicted_digit)})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)))