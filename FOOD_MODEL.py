import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import pandas as pd
import cv2
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import requests
from io import BytesIO

# Food-101 class names
FOOD_CLASSES = [
    'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare',
    'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito',
    'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake',
    'ceviche', 'cheesecake', 'cheese_plate', 'chicken_curry', 'chicken_quesadilla',
    'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder',
    'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes',
    'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict',
    'escargots', 'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras',
    'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice',
    'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich',
    'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup',
    'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna',
    'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup',
    'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters',
    'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck',
    'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib',
    'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake', 'risotto',
    'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits',
    'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake',
    'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare',
    'waffles'
]

# Calorie database (calories per 100g)
CALORIE_DATABASE = {
    'apple_pie': 237, 'baby_back_ribs': 297, 'baklava': 429, 'beef_carpaccio': 150,
    'beef_tartare': 180, 'beet_salad': 75, 'beignets': 350, 'bibimbap': 200,
    'bread_pudding': 234, 'breakfast_burrito': 185, 'bruschetta': 195, 'caesar_salad': 190,
    'cannoli': 290, 'caprese_salad': 150, 'carrot_cake': 415, 'ceviche': 120,
    'cheesecake': 321, 'cheese_plate': 340, 'chicken_curry': 175, 'chicken_quesadilla': 220,
    'chicken_wings': 290, 'chocolate_cake': 371, 'chocolate_mousse': 265, 'churros': 380,
    'clam_chowder': 95, 'club_sandwich': 240, 'crab_cakes': 180, 'creme_brulee': 297,
    'croque_madame': 325, 'cup_cakes': 305, 'deviled_eggs': 185, 'donuts': 452,
    'dumplings': 180, 'edamame': 120, 'eggs_benedict': 230, 'escargots': 190,
    'falafel': 333, 'filet_mignon': 250, 'fish_and_chips': 220, 'foie_gras': 462,
    'french_fries': 365, 'french_onion_soup': 85, 'french_toast': 220, 'fried_calamari': 175,
    'fried_rice': 163, 'frozen_yogurt': 159, 'garlic_bread': 350, 'gnocchi': 130,
    'greek_salad': 110, 'grilled_cheese_sandwich': 290, 'grilled_salmon': 206, 'guacamole': 160,
    'gyoza': 200, 'hamburger': 295, 'hot_and_sour_soup': 45, 'hot_dog': 290,
    'huevos_rancheros': 210, 'hummus': 166, 'ice_cream': 207, 'lasagna': 135,
    'lobster_bisque': 150, 'lobster_roll_sandwich': 280, 'macaroni_and_cheese': 164, 'macarons': 407,
    'miso_soup': 40, 'mussels': 172, 'nachos': 346, 'omelette': 154,
    'onion_rings': 280, 'oysters': 68, 'pad_thai': 190, 'paella': 180,
    'pancakes': 227, 'panna_cotta': 185, 'peking_duck': 337, 'pho': 90,
    'pizza': 266, 'pork_chop': 242, 'poutine': 510, 'prime_rib': 291,
    'pulled_pork_sandwich': 250, 'ramen': 188, 'ravioli': 175, 'red_velvet_cake': 378,
    'risotto': 166, 'samosa': 308, 'sashimi': 140, 'scallops': 137,
    'seaweed_salad': 45, 'shrimp_and_grits': 200, 'spaghetti_bolognese': 151, 'spaghetti_carbonara': 194,
    'spring_rolls': 160, 'steak': 271, 'strawberry_shortcake': 260, 'sushi': 150,
    'tacos': 226, 'takoyaki': 160, 'tiramisu': 240, 'tuna_tartare': 144,
    'waffles': 291
}

class FoodRecognitionModel:
    def __init__(self, input_shape=(224, 224, 3), num_classes=101):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        self.label_encoder = LabelEncoder()
        
    def build_model(self):
        """Build the food recognition model using EfficientNetB0"""
        # Load pre-trained EfficientNetB0 without top layers
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model layers initially
        base_model.trainable = False
        
        # Add custom top layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile the model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_5_accuracy']
        )
        
        return self.model
    
    def prepare_data_generators(self, train_dir, validation_dir, batch_size=32):
        """Prepare data generators for training and validation"""
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Only rescaling for validation
        validation_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        return train_generator, validation_generator
    
    def train_model(self, train_generator, validation_generator, epochs=50):
        """Train the food recognition model"""
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            ModelCheckpoint(
                'best_food_model.h5',
                monitor='val_accuracy',
                save_best_only=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=0.0001
            )
        ]
        
        # Train the model
        self.history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=callbacks
        )
        
        return self.history
    
    def fine_tune_model(self, train_generator, validation_generator, epochs=20):
        """Fine-tune the model by unfreezing some layers"""
        # Unfreeze the top layers of the base model
        base_model = self.model.layers[0]
        base_model.trainable = True
        
        # Fine-tune from this layer onwards
        fine_tune_at = 100
        
        # Freeze all the layers before the `fine_tune_at` layer
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001/10),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_5_accuracy']
        )
        
        # Continue training
        fine_tune_history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=[
                EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )
            ]
        )
        
        return fine_tune_history
    
    def preprocess_image(self, image_path):
        """Preprocess image for prediction"""
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
        else:
            image = image_path
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.input_shape[:2])
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def predict_food(self, image_path, top_k=5):
        """Predict food item from image"""
        if self.model is None:
            raise ValueError("Model not built or trained yet!")
        
        # Preprocess image
        processed_image = self.preprocess_image(image_path)
        
        # Make prediction
        predictions = self.model.predict(processed_image)
        
        # Get top-k predictions
        top_indices = np.argsort(predictions[0])[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            food_name = FOOD_CLASSES[idx]
            confidence = predictions[0][idx]
            calories_per_100g = CALORIE_DATABASE.get(food_name, 200)  # Default 200 if not found
            
            results.append({
                'food_name': food_name.replace('_', ' ').title(),
                'confidence': float(confidence),
                'calories_per_100g': calories_per_100g
            })
        
        return results
    
    def estimate_portion_size(self, image_path):
        """Estimate portion size using basic image analysis"""
        # This is a simplified approach - in practice, you'd need more sophisticated methods
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Find contours to estimate food area
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour (assumed to be the food item)
            largest_contour = max(contours, key=cv2.contourArea)
            food_area = cv2.contourArea(largest_contour)
            
            # Estimate portion size based on area (simplified)
            # This would need calibration in a real system
            image_area = image.shape[0] * image.shape[1]
            portion_ratio = food_area / image_area
            
            # Rough estimation of portion size in grams
            estimated_weight = min(max(portion_ratio * 300, 50), 500)  # Between 50-500g
            
            return estimated_weight
        
        return 150  # Default portion size
    
    def get_nutritional_info(self, food_name, portion_size_grams):
        """Get detailed nutritional information"""
        calories_per_100g = CALORIE_DATABASE.get(food_name.lower().replace(' ', '_'), 200)
        
        # Calculate calories for the portion
        total_calories = (calories_per_100g * portion_size_grams) / 100
        
        # Rough estimates for other nutrients (would need a proper nutrition database)
        nutrition_info = {
            'calories': round(total_calories),
            'protein_g': round(total_calories * 0.15 / 4),  # Rough estimate
            'carbs_g': round(total_calories * 0.45 / 4),
            'fat_g': round(total_calories * 0.35 / 9),
            'fiber_g': round(portion_size_grams * 0.03),
            'sugar_g': round(total_calories * 0.15 / 4)
        }
        
        return nutrition_info
    
    def analyze_food_image(self, image_path):
        """Complete food analysis pipeline"""
        # Get food predictions
        predictions = self.predict_food(image_path)
        
        # Get the most likely food item
        top_prediction = predictions[0]
        
        # Estimate portion size
        portion_size = self.estimate_portion_size(image_path)
        
        # Get nutritional information
        nutrition = self.get_nutritional_info(
            top_prediction['food_name'],
            portion_size
        )
        
        result = {
            'predictions': predictions,
            'top_prediction': top_prediction,
            'estimated_portion_grams': round(portion_size),
            'nutrition': nutrition
        }
        
        return result
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath):
        """Save the trained model"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")

# Example usage and demo
def create_demo_analysis():
    """Create a demo of the food recognition system"""
    # Initialize the model
    food_model = FoodRecognitionModel()
    
    # Build the model architecture
    model = food_model.build_model()
    
    print("Food Recognition and Calorie Estimation Model")
    print("=" * 50)
    print(f"Model Architecture:")
    print(f"- Input Shape: {food_model.input_shape}")
    print(f"- Number of Classes: {food_model.num_classes}")
    print(f"- Base Model: EfficientNetB0")
    print(f"- Total Parameters: {model.count_params():,}")
    
    # Display model summary
    model.summary()
    
    return food_model

def demo_prediction():
    """Demo function to show how predictions would work"""
    print("\nDemo: Food Recognition and Calorie Estimation")
    print("=" * 50)
    
    # Simulate prediction results
    sample_results = [
        {
            'food_name': 'Pizza',
            'confidence': 0.89,
            'calories_per_100g': 266
        },
        {
            'food_name': 'Hamburger',
            'confidence': 0.78,
            'calories_per_100g': 295
        },
        {
            'food_name': 'French Fries',
            'confidence': 0.65,
            'calories_per_100g': 365
        }
    ]
    
    print("Top 3 Food Predictions:")
    for i, result in enumerate(sample_results, 1):
        print(f"{i}. {result['food_name']}")
        print(f"   Confidence: {result['confidence']:.2%}")
        print(f"   Calories per 100g: {result['calories_per_100g']}")
        print()
    
    # Simulate nutritional analysis
    portion_size = 200  # grams
    top_food = sample_results[0]
    
    total_calories = (top_food['calories_per_100g'] * portion_size) / 100
    
    print(f"Nutritional Analysis for {portion_size}g of {top_food['food_name']}:")
    print(f"- Total Calories: {total_calories:.0f}")
    print(f"- Protein: {total_calories * 0.15 / 4:.0f}g")
    print(f"- Carbohydrates: {total_calories * 0.45 / 4:.0f}g")
    print(f"- Fat: {total_calories * 0.35 / 9:.0f}g")
    print(f"- Fiber: {portion_size * 0.03:.0f}g")

# Training instructions
training_instructions = """
TRAINING INSTRUCTIONS:
=====================

1. Data Preparation:
   - Download the Food-101 dataset from Kaggle
   - Organize images into train/validation folders
   - Each food class should have its own subfolder

2. Model Training:
   # Initialize and build model
   food_model = FoodRecognitionModel()
   food_model.build_model()
   
   # Prepare data generators
   train_gen, val_gen = food_model.prepare_data_generators(
       train_dir='path/to/train',
       validation_dir='path/to/validation'
   )
   
   # Train the model
   history = food_model.train_model(train_gen, val_gen, epochs=50)
   
   # Fine-tune (optional)
   fine_tune_history = food_model.fine_tune_model(train_gen, val_gen, epochs=20)
   
   # Save the model
   food_model.save_model('food_recognition_model.h5')

3. Model Usage:
   # Load trained model
   food_model.load_model('food_recognition_model.h5')
   
   # Analyze a food image
   results = food_model.analyze_food_image('path/to/food_image.jpg')
   
   # Print results
   print(f"Food: {results['top_prediction']['food_name']}")
   print(f"Confidence: {results['top_prediction']['confidence']:.2%}")
   print(f"Estimated portion: {results['estimated_portion_grams']}g")
   print(f"Calories: {results['nutrition']['calories']}")

4. Performance Optimization:
   - Use data augmentation to improve generalization
   - Implement transfer learning with pre-trained weights
   - Fine-tune hyperparameters (learning rate, batch size)
   - Consider ensemble methods for better accuracy
"""

if __name__ == "__main__":
    # Create demo
    food_model = create_demo_analysis()
    demo_prediction()
    print(training_instructions)
