"""
Animal Detection Configuration
Defines carnivore species and visual styling for bounding boxes.
"""

# Colors in BGR format (OpenCV standard)
COLOR_CARNIVORE = (0, 0, 255)  # Red
COLOR_OTHER = (255, 0, 0)      # Blue

# Species identified as carnivores for Task 2
CARNIVORE_SPECIES = [
    'lion', 'tiger', 'bear', 'wolf', 'leopard', 
    'cheetah', 'hyena', 'crocodile', 'shark'
]

# Detection settings
CONFIDENCE_THRESHOLD = 0.45
LINE_THICKNESS = 2
