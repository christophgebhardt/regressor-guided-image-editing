from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent # src folder

MODELS_DIR = PROJECT_ROOT / "models"
COCO_DIR = PROJECT_ROOT / "datasets/coco"
IMAGINAIRE_DIR = PROJECT_ROOT / "external/imaginaire"


IMAGE_OUTPUT_DIR = PROJECT_ROOT / "out/generated_images"
OPTIMIZED_OUTPUT_DIR = PROJECT_ROOT / "out/optimised"
IMAGINAIRE_OUTPUT_DIR = PROJECT_ROOT / "out/imaginaire"
