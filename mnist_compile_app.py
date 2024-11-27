from flask import Flask, render_template, jsonify, send_file, request
import os
from app import acquire_image, threshold_image, evaluate_digit
import tensorflow as tf
import numpy as np
from PIL import Image
import logging
import io
import base64
import re

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# Load the model at startup
model = tf.keras.models.load_model("mnist_cnn_model.keras")

@app.route('/')
def index():
    return render_template('mnist_compile_index.html')

@app.route('/capture', methods=['POST'])
def capture():
    try:
        # Acquire image from Arduino
        success, image = acquire_image(force_compile=False)
        
        if not success or image is None:
            return jsonify({
                'success': False,
                'error': 'Failed to acquire image from Arduino'
            })

        # Process the image
        thresholded_image = threshold_image(image, threshold=100)
        
        # Prepare images for display
        # Convert original image to base64
        original_buffer = io.BytesIO()
        image.save(original_buffer, format='PNG')
        original_base64 = base64.b64encode(original_buffer.getvalue()).decode()

        # Convert thresholded image to base64
        threshold_buffer = io.BytesIO()
        thresholded_image.save(threshold_buffer, format='PNG')
        threshold_base64 = base64.b64encode(threshold_buffer.getvalue()).decode()

        # Evaluate digit
        # Prepare image for model
        img = thresholded_image.convert("RGBA")
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        background.paste(img, mask=img)
        img = background.convert("L")
        img = img.resize((28, 28), Image.Resampling.BICUBIC)

        # Process image for model
        img_array = np.array(img, dtype=np.float32)
        img_array = 255 - img_array
        img_array = img_array / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        debug_img = Image.fromarray((img_array[0, :, :, 0] * 255).astype('uint8'))
        debug_img.save('debug_img.png')
        img_array = np.array(debug_img, dtype=np.float32)
        img_array = img_array / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        # Get predictions
        probabilities = model.predict(img_array)
        predicted_digit = int(probabilities.argmax(axis=1)[0])
        confidence = float(probabilities[0][predicted_digit])

        return jsonify({
            'success': True,
            'original_image': f'data:image/png;base64,{original_base64}',
            'threshold_image': f'data:image/png;base64,{threshold_base64}',
            'predicted_digit': predicted_digit,
            'confidence': confidence
        })

    except Exception as e:
        logging.error(f"Error during image capture: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image data from the request
        data = request.get_json()
        if not data or 'image' not in data:
            raise ValueError("No image data received")

        # Extract the base64 image data
        image_data = data['image']
        # Remove the data URL prefix and get clean base64
        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]
        
        # Add padding if needed
        padding = len(image_data) % 4
        if padding:
            image_data += '=' * (4 - padding)
        
        # Convert base64 to PIL Image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to grayscale
        image = image.convert('L')
        
        # Resize to 28x28
        image = image.resize((28, 28), Image.Resampling.BICUBIC)
        
        # Convert to numpy array and normalize
        img_array = np.array(image, dtype=np.float32)
        img_array = 255 - img_array  # Invert colors
        img_array = img_array / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        # Make prediction
        probabilities = model.predict(img_array)
        predicted_digit = int(probabilities.argmax(axis=1)[0])
        confidence = float(probabilities[0][predicted_digit])
        
        # Get all probabilities as a list and ensure they're JSON serializable
        all_probabilities = [float(p) for p in probabilities[0]]
        
        # Debug log
        logging.debug(f"Probabilities: {all_probabilities}")
        
        # Create preview images
        # Original grayscale
        original_buffer = io.BytesIO()
        image.save(original_buffer, format='PNG')
        original_base64 = base64.b64encode(original_buffer.getvalue()).decode()

        # Processed image
        processed_image = Image.fromarray((img_array[0, :, :, 0] * 255).astype('uint8'))
        processed_buffer = io.BytesIO()
        processed_image.save(processed_buffer, format='PNG')
        processed_base64 = base64.b64encode(processed_buffer.getvalue()).decode()

        response_data = {
            'digit': predicted_digit,
            'confidence': confidence,
            'preview': f'data:image/png;base64,{processed_base64}',
            'original_preview': f'data:image/png;base64,{original_base64}',
            'probabilities': all_probabilities
        }
        
        # Debug log
        logging.debug(f"Sending response: {response_data}")
        
        return jsonify(response_data)

    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == '__main__':
    # Create artifacts directory if it doesn't exist
    os.makedirs('artifacts', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5100)