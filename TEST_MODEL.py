from food_model import FoodRecognitionModel
import sys
import os

def test_trained_model(model_path, image_path):
    """Test your trained model"""
    
    # Load model
    print(f"📥 Loading model: {model_path}")
    food_model = FoodRecognitionModel()
    food_model.load_model(model_path)
    
    # Analyze image
    print(f"🔍 Analyzing image: {image_path}")
    results = food_model.analyze_food_image(image_path)
    
    # Display results
    print(f"\n🍽️ FOOD RECOGNITION RESULTS")
    print("=" * 40)
    
    top_pred = results['top_prediction']
    print(f"🥘 Food: {top_pred['food_name']}")
    print(f"🎯 Confidence: {top_pred['confidence']:.1%}")
    print(f"⚖️ Estimated portion: {results['estimated_portion_grams']}g")
    
    print(f"\n🔥 NUTRITION INFO:")
    nutrition = results['nutrition']
    print(f"Calories: {nutrition['calories']}")
    print(f"Protein: {nutrition['protein_g']}g")
    print(f"Carbs: {nutrition['carbs_g']}g")
    print(f"Fat: {nutrition['fat_g']}g")
    
    return results

if __name__ == "__main__":
    # Find the latest model
    models_dir = 'models'
    if os.path.exists(models_dir):
        models = [f for f in os.listdir(models_dir) if f.endswith('.h5')]
        if models:
            latest_model = max(models)
            model_path = os.path.join(models_dir, latest_model)
            
            print(f"🔍 Found model: {latest_model}")
            
            # Test with a food image
            if len(sys.argv) > 1:
                image_path = sys.argv[1]
                test_trained_model(model_path, image_path)
            else:
                print("Usage: python test_model.py <food_image.jpg>")
        else:
            print("❌ No trained models found!")
    else:
        print("❌ Models directory not found!")
