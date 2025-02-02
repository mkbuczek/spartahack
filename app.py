import io
import base64
from PIL import Image
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS

# Load your trained letter recognition model (for A-Z letters)
model = tf.keras.models.load_model('emnist_letter_model.keras')

app = Flask(__name__)

CORS(app)
print(model.output_shape)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_data = data['image']

    # Decode the base64 image data
    img_data = base64.b64decode(image_data.split(',')[1])
    img = Image.open(io.BytesIO(img_data))

    # Preprocess the image
    img = img.convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28 (standard size for letter recognition)
    img_array = np.array(img)

    # Invert the image (if needed)
    img_array = 255 - img_array  # Invert the colors (black to white and vice versa)

    # Normalize the image
    img_array = img_array.astype('float32') / 255.0  # Normalize to [0, 1]

    # Add channel dimension and batch dimension
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension (grayscale)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Get prediction (probabilities for each class)
    prediction = model.predict(img_array)

    # Get the predicted class (0-25 for A-Z)
    predicted_class = np.argmax(prediction, axis=1)[0]

    # Map predicted class to a letter (0 -> A, 1 -> B, ..., 25 -> Z)
    predicted_letter = chr(predicted_class + ord('A'))
    

    # Get the confidence of the prediction (probability of the predicted class)
    confidence = prediction[0][predicted_class]  # Confidence of the predicted class

    # Return the prediction and its confidence
    return jsonify({
        'prediction': predicted_letter,  # Predict letter (A-Z)
        'confidence': float(confidence)  # Convert to float for easy JSON response
    })

if __name__ == '__main__':
    app.run(debug=True)
