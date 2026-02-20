from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent # git project root

#location of trained models
MODELS_DIR = PROJECT_ROOT / "models"

#location of coco images and annotations
COCO_DIR = PROJECT_ROOT / "data/coco"

# output paths for diffrent tasks
OUT_DIR = PROJECT_ROOT / "out"
