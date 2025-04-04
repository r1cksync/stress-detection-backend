from flask import Flask, request, jsonify
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.metrics import MeanSquaredError
from flask_cors import CORS
import logging
import tempfile
import requests
import re

# Set up logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# Configure CORS using the FRONTEND_URL environment variable
CORS(app, resources={r"/api/*": {"origins": os.getenv("FRONTEND_URL", "http://localhost:3000")}})

# Use a temporary directory for uploads on Render
UPLOAD_FOLDER = tempfile.gettempdir()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the stress detection model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'stress_model.h5')

# Download the model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    logging.info("Downloading stress detection model...")
    model_url = "https://www.dropbox.com/scl/fi/pxrrn1g441z58q61dmz4n/stress_model.h5?rlkey=1p0hmvtlduavs4nlbohqga6ni&st=i1rivr61&dl=1"
    session = requests.Session()  # Use a session to handle cookies and redirects
    
    # Initial request to the Dropbox URL
    response = session.get(model_url, stream=True, allow_redirects=True)
    
    # Check if the response is an HTML page (indicating a preview page)
    content_type = response.headers.get('Content-Type', '')
    if 'text/html' in content_type:
        logging.info("Received Dropbox preview page. Attempting to handle redirect...")
        html_content = response.text
        
        # Look for a meta refresh tag or a download link in the HTML
        refresh_match = re.search(r'<meta http-equiv="refresh" content="0;url=([^"]+)">', html_content)
        if refresh_match:
            direct_url = refresh_match.group(1)
            logging.info(f"Found redirect URL: {direct_url}")
            # Follow the redirect to download the file
            response = session.get(direct_url, stream=True, allow_redirects=True)
        else:
            # Alternatively, look for a download button link (less reliable)
            download_match = re.search(r'href="([^"]+)"[^>]*>Download<', html_content)
            if download_match:
                direct_url = download_match.group(1)
                logging.info(f"Found download URL: {direct_url}")
                response = session.get(direct_url, stream=True, allow_redirects=True)
            else:
                logging.error("Could not find a direct download URL in the HTML.")
                response = None
    else:
        logging.info("Direct download link worked without redirect.")

    # Proceed with the download if we have a valid response
    if response and response.status_code == 200:
        content_length = response.headers.get('Content-Length')
        expected_size = 109420000  # Expected size of stress_model.h5 in bytes (104.31 MB)
        if content_length:
            content_length = int(content_length)
            logging.info(f"Downloading file of size: {content_length} bytes")
            if content_length < 10000:  # If the file is too small, it's likely not the model
                logging.error("Downloaded file is too small to be the model. Aborting.")
            else:
                with open(MODEL_PATH, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                logging.info("Model downloaded successfully.")
        else:
            logging.error("No Content-Length header in response. Aborting download.")
    else:
        logging.error(f"Failed to download model: HTTP {response.status_code if response else 'N/A'}")

# Log before loading the model
logging.info("Starting to load the stress detection model...")
try:
    # Explicitly provide the mse metric as a custom object
    model = load_model(MODEL_PATH, custom_objects={'mse': MeanSquaredError()})
    logging.info("Stress detection model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load stress detection model: {str(e)}")
    model = None

# Log after model loading attempt
logging.info("Model loading attempt completed. Model is None: %s", model is None)

def load_and_preprocess_image(image_path):
    # Load the image in grayscale (FER2013 images are grayscale)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Resize to 224x224 (model input size)
    img = cv2.resize(img, (224, 224))
    
    # Convert grayscale to RGB by duplicating channels
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    # Convert to numpy array and add batch dimension
    img = np.expand_dims(img, axis=0)
    
    # Preprocess the image for ResNet50
    img = preprocess_input(img)
    return img

@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify the backend is running.
    """
    return jsonify({"status": "healthy", "message": "Backend is running"}), 200

@app.route('/api/predict-stress', methods=['POST'])
def predict_stress():
    if model is None:
        logging.error("Stress detection model not loaded")
        return jsonify({'error': 'Stress detection model not loaded'}), 500

    if 'image' not in request.files:
        logging.error("No image uploaded in request")
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        logging.error("No image selected")
        return jsonify({'error': 'No image selected'}), 400
    
    # Save the uploaded image to the temporary directory
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(image_path)
    logging.debug(f"Image saved to {image_path}")
    
    try:
        # Load and preprocess the image
        img = load_and_preprocess_image(image_path)
        
        # Make prediction
        stress_level = model.predict(img)[0][0]
        
        # Ensure the prediction is within the valid range (0 to 100)
        stress_level = float(np.clip(stress_level, 0, 100))
        
        logging.info(f"Predicted stress level: {stress_level}")
        return jsonify({'stress_level': stress_level})
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up: delete the uploaded image
        if os.path.exists(image_path):
            os.remove(image_path)
            logging.debug(f"Image deleted from {image_path}")

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5001))
    logging.info(f"Starting Flask app on port {port}...")
    app.run(host='0.0.0.0', port=port)