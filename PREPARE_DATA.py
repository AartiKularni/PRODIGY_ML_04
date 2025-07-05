import os
import shutil
import random
from pathlib import Path

def prepare_training_data():
    """Convert Food-101 dataset to training format"""
    
    # Paths
    source_dir = Path('data/food-101/images')
    train_dir = Path('data/train')
    val_dir = Path('data/validation')
    
    # Create directories
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    print("🔄 Organizing data for training...")
    
    # Process each food class
    for food_class in source_dir.iterdir():
        if food_class.is_dir():
            class_name = food_class.name
            
            # Create class directories
            train_class_dir = train_dir / class_name
            val_class_dir = val_dir / class_name
            train_class_dir.mkdir(exist_ok=True)
            val_class_dir.mkdir(exist_ok=True)
            
            # Get all images for this class
            images = list(food_class.glob('*.jpg'))
            random.shuffle(images)
            
            # Split 80% train, 20% validation
            split_idx = int(len(images) * 0.8)
            train_images = images[:split_idx]
            val_images = images[split_idx:]
            
            # Copy images
            for img in train_images:
                shutil.copy2(img, train_class_dir / img.name)
            
            for img in val_images:
                shutil.copy2(img, val_class_dir / img.name)
            
            print(f"✅ {class_name}: {len(train_images)} train, {len(val_images)} val")
    
    print("🎉 Data preparation complete!")
    print(f"📊 Training classes: {len(list(train_dir.iterdir()))}")
    print(f"📊 Validation classes: {len(list(val_dir.iterdir()))}")

if __name__ == "__main__":
    prepare_training_data()
