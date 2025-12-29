from ultralytics import YOLO
import sys
import os

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

def train_model():
    print("Initializing Training...")
    
    # Check if data.yaml exists
    data_yaml = os.path.join("data", "animal_dataset", "data.yaml")
    if not os.path.exists(data_yaml):
        print(f"Error: {data_yaml} not found. Please run scripts/data_converter.py first.")
        return

    # Load a model
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    # Train the model
    print(f"Starting training on {data_yaml}...")
    results = model.train(data=data_yaml, epochs=50, imgsz=640, project="models", name="animal_train")
    
    # Save the best model to the expected path
    # YOLO saves to models/animal_train/weights/best.pt
    # We want to copy it to models/animal_best.pt
    
    best_weights = os.path.join("models", "animal_train", "weights", "best.pt")
    target_weights = os.path.join("models", "animal_best.pt")
    
    import shutil
    if os.path.exists(best_weights):
        shutil.copy(best_weights, target_weights)
        print(f"Training Complete. Best model saved to {target_weights}")
    else:
        print("Training finished but could not locate best.pt.")

if __name__ == "__main__":
    train_model()
