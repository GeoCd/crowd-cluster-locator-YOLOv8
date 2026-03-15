from pathlib import Path

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"

MODEL_WEIGHTS = "yolov8s.pt"
PERSON_CLASS_ID = 0
CONFIDENCE_THRESHOLD = 0.35
IOU_THRESHOLD = 0.5

GRID_ROWS = 3
GRID_COLS = 3

VIZ_MODE = "both" # "heatmap", "grid" or "both"

HEATMAP_ALPHA = 0.5
HEATMAP_RADIUS = 60

DENSITY_ALERT_THRESHOLD = 5