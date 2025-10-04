#!/bin/bash

# Skin Disease Prediction System - Startup Script
# This script activates the virtual environment and starts the Flask application

echo "=============================================="
echo "Skin Disease Prediction System"
echo "=============================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run the following commands first:"
    echo "python3 -m venv venv"
    echo "source venv/bin/activate"
    echo "pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if model exists
if [ ! -f "model/skin_model.h5" ]; then
    echo "⚠️  Model file not found. Creating demo model..."
    python create_demo_model.py
fi

# Start the Flask application
echo "🚀 Starting Flask application..."
echo ""
echo "The application will be available at:"
echo "👉 http://127.0.0.1:5000/"
echo ""
echo "Press Ctrl+C to stop the application"
echo "=============================================="
echo ""

python app.py
