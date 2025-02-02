import io 
import base64
from PIL import Image
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS

model = tf.keras.models.load_model('mnist_model.keras')

app = Flask(__name__)

CORS(app, resources={r"/predict": {"origins": "https://127.0.0.1:5500"}})

@app.route('/predict', methods=['POST'])

def predict():
    data = request.get_json()
    image_data = data['image']

    img_data = base64.b64decode(image_data.split(',')[1])
    img = Image.open(io.BytesIO(img_data))

    #debug
    img.show()
    img = img.convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img)

    img_array = img_array.astype('float32') / 255.0

    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)

    predicted_class = np.argmax(prediction, axis=1)[0]

    #debug
    #img.show()
    confidence = np.max(prediction, axis=1)[0]  # Get the confidence score (probability of the predicted class)

    return jsonify({'prediction': str(predicted_class)})

if __name__ == '__main__':
    app.run(debug=True)