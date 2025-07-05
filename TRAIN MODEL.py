import tensorflow as tf
from food_model import FoodRecognitionModel
import os
from datetime import datetime

def setup_gpu():
    """Setup GPU if available"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"🔥 Using GPU: {len(gpus)} device(s)")
        except RuntimeError as e:
            print(f"⚠️ GPU setup error: {e}")
    else:
        print("💻 Using CPU (training will be slower)")

def train_food_model():
    """Train the food recognition model"""
    
    # Setup
    setup_gpu()
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Initialize model
    print("🏗️ Building model...")
    food_model = FoodRecognitionModel()
    model = food_model.build_model()
    
    print(f"✅ Model built with {model.count_params():,} parameters")
    
    # Check if data exists
    if not os.path.exists('data/train') or not os.path.exists('data/validation'):
        print("❌ ERROR: Training data not found!")
        print("Please run 'python prepare_data.py' first")
        return
    
    # Prepare data generators
    print("📊 Preparing data generators...")
    train_gen, val_gen = food_model.prepare_data_generators(
        train_dir='data/train',
        validation_dir='data/validation',
        batch_size=32
    )
    
    print(f"📈 Training samples: {train_gen.samples}")
    print(f"📉 Validation samples: {val_gen.samples}")
    
    # Start training
    print("🚀 Starting training...")
    print("⏰ This will take several hours...")
    
    start_time = datetime.now()
    
    # Phase 1: Initial training
    print("\n🏋️ Phase 1: Initial Training (50 epochs)")
    history = food_model.train_model(
        train_gen, 
        val_gen, 
        epochs=50
    )
    
    # Phase 2: Fine-tuning
    print("\n🎯 Phase 2: Fine-tuning (20 epochs)")
    fine_tune_history = food_model.fine_tune_model(
        train_gen, 
        val_gen, 
        epochs=20
    )
    
    # Save model
    model_path = f'models/food_model_{datetime.now().strftime("%Y%m%d_%H%M")}.h5'
    food_model.save_model(model_path)
    
    # Training complete
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n🎉 TRAINING COMPLETE!")
    print(f"⏱️ Duration: {duration}")
    print(f"💾 Model saved: {model_path}")
    
    # Show training results
    final_acc = history.history['val_accuracy'][-1]
    print(f"📊 Final validation accuracy: {final_acc:.2%}")
    
    # Plot training history
    try:
        food_model.plot_training_history()
    except:
        print("📊 Training plots saved (display not available)")
    
    return food_model, model_path

if __name__ == "__main__":
    trained_model, model_path = train_food_model()
