from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent # git project root

#location of trained models
MODELS_DIR = PROJECT_ROOT / "models"

#location of coco images and annotations
DATA_DIR = PROJECT_ROOT / "data"

# output paths for diffrent tasks
OUT_DIR = PROJECT_ROOT / "out"
