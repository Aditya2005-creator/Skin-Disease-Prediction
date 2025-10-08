#!/usr/bin/env python3
"""
Quick Training Script for Skin Disease Prediction
Simplified version for beginners - trains a basic CNN model
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import os

# Simple configuration
IMG_SIZE = 224
BATCH_SIZE = 16  # Smaller for beginners
EPOCHS = 20      # Fewer epochs for quick training

# Disease classes
DISEASE_CLASSES = [
    'Eczema', 'Melanoma', 'Basal Cell Carcinoma', 'Benign Keratosis', 'Normal Skin',
    'Psoriasis', 'Seborrheic Keratosis', 'Vitiligo', 'Rash', 'Acne',
    'Wart', 'Mole', 'Dermatitis', 'Rosacea', 'Fungal Infection'
]

def create_simple_model():
    """Create a simple CNN model for beginners."""
    
    print("🏗️  Creating Simple CNN Model...")
    
    model = models.Sequential([
        # Input layer
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        
        # Simple convolutional layers
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(DISEASE_CLASSES), activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def generate_simple_data():
    """Generate simple training data."""
    
    print("📊 Generating Simple Training Data...")
    
    # Create directories
    base_dir = 'simple_dataset'
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')
    
    for split in ['train', 'test']:
        for disease in DISEASE_CLASSES:
            disease_dir = os.path.join(base_dir, split, disease)
            os.makedirs(disease_dir, exist_ok=True)
    
    # Generate simple images
    np.random.seed(42)
    
    for split in ['train', 'test']:
        samples_per_class = 50 if split == 'train' else 10  # Fewer samples for quick training
        
        for i, disease in enumerate(DISEASE_CLASSES):
            disease_dir = os.path.join(base_dir, split, disease)
            
            print(f"  📁 Generating {samples_per_class} samples for {disease}...")
            
            for j in range(samples_per_class):
                # Create simple colored images
                if disease == 'Vitiligo':
                    img = np.full((IMG_SIZE, IMG_SIZE, 3), [255, 255, 255], dtype=np.uint8)  # White
                elif disease == 'Rash':
                    img = np.full((IMG_SIZE, IMG_SIZE, 3), [255, 100, 100], dtype=np.uint8)  # Red
                elif disease == 'Acne':
                    img = np.full((IMG_SIZE, IMG_SIZE, 3), [255, 150, 150], dtype=np.uint8)  # Light red
                elif disease == 'Normal Skin':
                    img = np.full((IMG_SIZE, IMG_SIZE, 3), [240, 220, 180], dtype=np.uint8)  # Skin tone
                else:
                    # Random colors for other diseases
                    color = np.random.randint(100, 255, 3)
                    img = np.full((IMG_SIZE, IMG_SIZE, 3), color, dtype=np.uint8)
                
                # Add some noise for variation
                noise = np.random.normal(0, 10, img.shape)
                img = np.clip(img + noise, 0, 255).astype(np.uint8)
                
                # Save image
                from PIL import Image
                pil_img = Image.fromarray(img)
                filename = f"{disease}_{j+1}.jpg"
                pil_img.save(os.path.join(disease_dir, filename))
    
    print(f"✅ Created simple dataset with {len(DISEASE_CLASSES)} classes")
    return base_dir

def quick_train():
    """Quick training function for beginners."""
    
    print("=" * 60)
    print("🚀 QUICK SKIN DISEASE MODEL TRAINING")
    print("=" * 60)
    
    # Create dataset
    base_dir = generate_simple_data()
    
    # Setup data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        os.path.join(base_dir, 'train'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Validation generator
    val_generator = val_datagen.flow_from_directory(
        os.path.join(base_dir, 'train'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    # Create model
    model = create_simple_model()
    
    print("\n📋 MODEL SUMMARY")
    print("=" * 30)
    model.summary()
    
    # Setup callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            'model/skin_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train the model
    print(f"\n🏋️  Starting quick training for {EPOCHS} epochs...")
    print("=" * 50)
    
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Final results
    print(f"\n🎯 TRAINING COMPLETED!")
    print("=" * 50)
    print(f"Final Validation Accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"Model saved as: model/skin_model.h5")
    
    # Clean up
    import shutil
    shutil.rmtree(base_dir)
    print("🧹 Cleaned up temporary dataset")
    
    return model, history

if __name__ == "__main__":
    # Ensure model directory exists
    os.makedirs('model', exist_ok=True)
    
    print("🎓 QUICK TRAINING FOR BEGINNERS")
    print("This script will train a simple model in about 5-10 minutes")
    print("Perfect for learning and experimentation!")
    print()
    
    # Train the model
    model, history = quick_train()
    
    print("\n" + "=" * 60)
    print("🎉 QUICK TRAINING COMPLETED!")
    print("=" * 60)
    print("\n📁 Model saved as: model/skin_model.h5")
    print("🚀 Your Flask app will now use the new model!")
    print("\nTo restart your Flask app:")
    print("python app.py")
    print("\n💡 For more advanced training, use: python train_model.py")
