import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project paths
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
KAGGLE_DIR = Path.home() / ".kaggle"

# Ensure required directories exist
DATA_DIR.mkdir(exist_ok=True)
KAGGLE_DIR.mkdir(exist_ok=True)

# Kaggle configuration
KAGGLE_USERNAME = os.getenv('KAGGLE_USERNAME')
KAGGLE_KEY = os.getenv('KAGGLE_KEY')

# Create kaggle.json if credentials are available
if KAGGLE_USERNAME and KAGGLE_KEY:
    kaggle_json_path = KAGGLE_DIR / "kaggle.json"
    if not kaggle_json_path.exists():
        import json
        with open(kaggle_json_path, 'w') as f:
            json.dump({
                "username": KAGGLE_USERNAME,
                "key": KAGGLE_KEY
            }, f)
        # Set appropriate permissions
        os.chmod(kaggle_json_path, 0o600)
