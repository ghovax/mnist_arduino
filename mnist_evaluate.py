import logging
import numpy as np
from PIL import Image
import tensorflow as tf

# Setup basic logging configuration
logging.basicConfig(
    level=logging.INFO,
)

def evaluate_digit(image_path):
    """Evaluate the processed image using MNIST model to predict the digit."""
    try:
        # Load and compile the model with metrics
        model = tf.keras.models.load_model("mnist_model.keras")
        
        # Load and preprocess the image
        img = Image.open(image_path).convert('RGBA')  # Convert to RGBA first
        # Create a white background
        background = Image.new('RGBA', img.size, (255, 255, 255, 255))
        # Paste the image using itself as mask (this handles transparency)
        background.paste(img, mask=img)
        # Now convert to grayscale
        img = background.convert('L')
        img = img.resize((28, 28), Image.Resampling.BICUBIC)  # Resize to MNIST format
        
        # Convert to numpy array and normalize
        img_array = np.array(img, dtype=np.float32)
        # Invert the colors to get white digit on black background
        img_array = 255 - img_array
        img_array = img_array / 255.0  # Normalize to [0,1]
        
        # Reshape for model input (add batch and channel dimensions)
        img_array = img_array.reshape(1, 28, 28, 1)

        # Save the preprocessed image for debugging
        debug_img = (img_array[0, :, :, 0] * 255).astype(np.uint8)
        Image.fromarray(debug_img).save('artifacts/debug_mnist_input.png')
        
        # Get prediction using debug image
        predictions = model.predict(img_array, verbose=0)
        
        # Apply softmax to convert logits to probabilities
        probabilities = tf.nn.softmax(predictions).numpy()

        # Get the three most likely predictions
        top_3_indices = np.argsort(probabilities[0])[-3:][::-1]
        top_3_str = ", ".join([f"{d} ({probabilities[0][d]:.2%})" for d in top_3_indices])
        
        return top_3_str
        
    except Exception as e:
        logging.error(f"Failed to evaluate image: {str(e)}")
        return None

def main():
    # Evaluate the processed image
    top_3_str = evaluate_digit('artifacts/processed_image.png')
    if top_3_str is not None:
        logging.info(f"The most likely digits are: {top_3_str}")

if __name__ == "__main__":
    main()