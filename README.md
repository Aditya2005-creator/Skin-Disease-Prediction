<<<<<<< HEAD
# Skin Disease Prediction System

A Flask-based web application that uses machine learning to predict skin diseases from uploaded images. The system employs a Convolutional Neural Network (CNN) to analyze skin images and provide predictions with confidence scores.

## Features

- 🖼️ **Image Upload**: Support for JPG, PNG, and JPEG image formats
- 🤖 **AI Prediction**: CNN-based skin disease classification
- 📊 **Confidence Scores**: Detailed prediction confidence levels
- 🎨 **Modern UI**: Responsive design with Bootstrap and custom CSS
- ⚡ **Real-time Processing**: Fast image analysis with loading indicators
- 🛡️ **Error Handling**: Graceful error handling and user feedback

## Supported Skin Conditions

The system can predict the following skin diseases:
- Eczema
- Melanoma
- Basal Cell Carcinoma
- Benign Keratosis
- Normal Skin
- Psoriasis
- Seborrheic Keratosis

## Project Structure

```
ml project/
├── app.py                 # Flask backend application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── templates/
│   └── index.html        # Main web interface
├── static/
│   └── uploads/          # Directory for uploaded images
└── model/
    ├── skin_model.h5     # Trained CNN model (you need to provide this)
    └── README.md         # Model information
```

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Step 1: Clone or Download
```bash
# If using git
git clone <repository-url>
cd "ml project"

# Or simply navigate to your project directory
cd "/Users/Aditya/Desktop/CURSOR PROJECTS/ml project"
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Add Your Model
1. Place your trained CNN model file as `skin_model.h5` in the `model/` directory
2. Ensure your model:
   - Accepts input shape (128, 128, 3)
   - Outputs probability distribution over 7 disease classes
   - Is saved in Keras/TensorFlow format (.h5)

### Step 5: Run the Application
```bash
python app.py
```

The application will start on `http://127.0.0.1:5000/`

## Usage

1. **Access the Application**: Open your web browser and navigate to `http://127.0.0.1:5000/`

2. **Upload an Image**: 
   - Click "Choose Image" or drag and drop an image file
   - Supported formats: JPG, PNG, JPEG
   - Maximum file size: 16MB

3. **View Results**: 
   - The system will automatically analyze your image
   - Results show predicted disease and confidence level
   - Image preview and detailed analysis are displayed

## Technical Details

### Backend (Flask)
- **Framework**: Flask 2.3.3
- **ML Library**: TensorFlow 2.13.0
- **Image Processing**: Pillow 10.0.1
- **File Handling**: Werkzeug 2.3.7

### Frontend
- **Framework**: HTML5, CSS3, JavaScript
- **UI Library**: Bootstrap 5.3.0
- **Icons**: Font Awesome 6.0.0
- **Features**: Drag & drop, responsive design, loading animations

### Model Requirements
- **Input**: RGB images resized to 128x128 pixels
- **Preprocessing**: Normalization (pixel values 0-1)
- **Architecture**: CNN-based classification model
- **Output**: 7-class probability distribution

## API Endpoints

### GET /
- **Description**: Renders the main upload page
- **Response**: HTML page with upload interface

### POST /predict
- **Description**: Processes uploaded image and returns prediction
- **Input**: Multipart form data with image file
- **Response**: JSON with prediction results

Example Response:
```json
{
    "success": true,
    "disease": "Eczema",
    "confidence": 87.5,
    "image_url": "/static/uploads/timestamp_filename.jpg"
}
```

## Customization

### Adding New Disease Classes
1. Update `DISEASE_CLASSES` dictionary in `app.py`
2. Retrain your model with additional classes
3. Update the frontend display as needed

### Modifying Model Input Size
1. Change image resize dimensions in `preprocess_image()` function
2. Update model architecture accordingly
3. Ensure consistent input shape throughout

### Styling Changes
- Modify CSS in the `<style>` section of `index.html`
- Bootstrap classes can be customized for different themes
- Color scheme can be changed via CSS variables

## Troubleshooting

### Common Issues

**Model Not Found Error**
- Ensure `skin_model.h5` exists in the `model/` directory
- Check file permissions and path

**Import Errors**
- Verify all dependencies are installed: `pip install -r requirements.txt`
- Check Python version compatibility

**Upload Issues**
- Ensure `static/uploads/` directory exists
- Check file size limits (16MB max)
- Verify supported image formats

**Prediction Errors**
- Check model input/output shape compatibility
- Verify image preprocessing pipeline
- Ensure model is properly loaded

### Debug Mode
The application runs with `debug=True` for development. For production:
- Set `debug=False` in `app.py`
- Use a production WSGI server like Gunicorn
- Configure proper error logging

## Development

### Running in Development
```bash
python app.py
```

### Running in Production
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## Security Notes

- Change the secret key in production
- Implement proper file validation
- Add rate limiting for uploads
- Use HTTPS in production
- Sanitize uploaded filenames

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is for educational and research purposes. Please ensure compliance with medical software regulations if used in clinical settings.

## Disclaimer

⚠️ **Important Medical Disclaimer**: This application is for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis, treatment, or advice. Always consult with qualified healthcare professionals for medical concerns.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review error messages in the console
3. Ensure all dependencies are properly installed
4. Verify model compatibility

---

**Built with ❤️ using Flask, TensorFlow, and modern web technologies**
=======
# Skin-Disease-Prediction
Skin-Disease-Prediction
>>>>>>> 84493f15fe3820269916f3b63b29bbde8ec77419
