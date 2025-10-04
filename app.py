from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from werkzeug.utils import secure_filename
import base64
import io

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Disease classes mapping - Updated with more common skin conditions
DISEASE_CLASSES = {
    0: 'Eczema',
    1: 'Melanoma', 
    2: 'Basal Cell Carcinoma',
    3: 'Benign Keratosis',
    4: 'Normal Skin',
    5: 'Psoriasis',
    6: 'Seborrheic Keratosis',
    7: 'Vitiligo',
    8: 'Rash',
    9: 'Acne',
    10: 'Wart',
    11: 'Mole',
    12: 'Dermatitis',
    13: 'Rosacea',
    14: 'Fungal Infection'
}

# Global variable to store the loaded model
model = None

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    """Load the pre-trained CNN model."""
    global model
    try:
        model_path = 'model/skin_model.h5'
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            print("Model loaded successfully!")
        else:
            print(f"Model file not found at {model_path}")
            print("Please ensure skin_model.h5 exists in the model/ directory")
    except Exception as e:
        print(f"Error loading model: {str(e)}")

def preprocess_image(image_path):
    """
    Preprocess the uploaded image for model prediction.
    Resize to 128x128 and normalize pixel values.
    """
    try:
        # Load image
        img = Image.open(image_path)
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to 224x224 for better feature extraction
        img = img.resize((224, 224))
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Normalize pixel values to [0, 1]
        img_array = img_array.astype('float32') / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        return None

def predict_disease(image_array):
    """
    Make prediction using the loaded model.
    """
    global model
    try:
        if model is None:
            return None, "Model not loaded"
        
        # Make prediction
        predictions = model.predict(image_array)
        predicted_class = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]) * 100)
        
        disease_name = DISEASE_CLASSES.get(predicted_class, "Unknown")
        
        return disease_name, confidence
    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        return None, str(e)

@app.route('/')
def home():
    """Render the home page with detailed information."""
    return render_template('home.html')

@app.route('/predict')
def predict_page():
    """Render the prediction interface page."""
    return render_template('index.html')

@app.route('/ml-details')
def ml_details():
    """Render the machine learning details page."""
    return render_template('ml_details.html')

@app.route('/tech-specs')
def tech_specs():
    """Render the technical specifications page."""
    return render_template('tech_specs.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction."""
    try:
        # Check if file was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image file uploaded'}), 400
        
        file = request.files['image']
        
        # Check if file was selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check if file type is allowed
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload JPG, PNG, or JPEG files only.'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        # Add timestamp to avoid filename conflicts
        import time
        timestamp = str(int(time.time()))
        filename = f"{timestamp}_{filename}"
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Preprocess image
        img_array = preprocess_image(file_path)
        if img_array is None:
            return jsonify({'error': 'Error processing image'}), 500
        
        # Make prediction
        disease, confidence = predict_disease(img_array)
        if disease is None:
            return jsonify({'error': f'Prediction failed: {confidence}'}), 500
        
        # Prepare response
        result = {
            'success': True,
            'disease': disease,
            'confidence': round(confidence, 2),
            'image_url': f"/static/uploads/{filename}"
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in predict route: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return render_template('index.html'), 404

if __name__ == '__main__':
    # Ensure upload directory exists
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs('model', exist_ok=True)
    
    # Load the model when the app starts
    print("Loading skin disease prediction model...")
    load_model()
    
    print("Starting Flask application...")
    print("Access the application at: http://127.0.0.1:3000/")
    app.run(debug=True, host='127.0.0.1', port=3000)
