import cv2
import sys
import os
from ultralytics import YOLO

# Add project root to path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Author: Tarrush Saxena

try:
    from config.animal_config import CARNIVORE_SPECIES, COLOR_CARNIVORE, COLOR_OTHER, CONFIDENCE_THRESHOLD, LINE_THICKNESS
except ImportError:
    # Fallback or dev mode
    CARNIVORE_SPECIES = ['lion', 'tiger', 'bear', 'wolf', 'leopard', 'cheetah', 'hyena', 'crocodile', 'shark']
    COLOR_CARNIVORE = (0, 0, 255)
    COLOR_OTHER = (255, 0, 0)
    CONFIDENCE_THRESHOLD = 0.45
    LINE_THICKNESS = 2

class AnimalEngine:
    def __init__(self, model_path=None):
        if model_path is None:
            # Default to yolov8n if no custom model provided
            self.model = YOLO('yolov8n.pt') 
        else:
            self.model = YOLO(model_path)
            
        # If we are using a generic model (yolov8n), we might need to map COCO classes to our expected output 
        # Requirement: "Your model should be capable of distinguishing between different species", implying a custom trained model.
        # For 'yolov8n.pt' (COCO), it detects 'bear', 'zebra', 'giraffe', 'elephant', 'cat' (tiger/lion often misclassified as cat/dog in base model).
        # Assuming 'models/animal_best.pt' is passed eventually.

    def is_carnivore(self, label):
        return label.lower() in CARNIVORE_SPECIES

    def process_frame(self, frame):
        """
        Runs inference on a single frame.
        Returns: 
            processed_frame: Image with bounding boxes
            carnivore_count: Number of carnivores detected
        """
        results = self.model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        result = results[0]
        
        carnivore_count = 0
        names = result.names

        for box in result.boxes:
            class_id = int(box.cls[0])
            label = names[class_id]
            conf = float(box.conf[0])
            
            # Coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Logic
            if self.is_carnivore(label):
                color = COLOR_CARNIVORE
                carnivore_count += 1
            else:
                color = COLOR_OTHER
            
            # Draw
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, LINE_THICKNESS)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        return frame, carnivore_count
