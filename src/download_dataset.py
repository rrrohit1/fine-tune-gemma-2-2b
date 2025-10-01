import os
import kaggle
from pathlib import Path

def download_medical_transcriptions():
    # Create a data directory if it doesn't exist
    data_dir = Path('../data')
    data_dir.mkdir(exist_ok=True)
    
    # Download the dataset
    try:
        # Using Kaggle API to download the dataset
        kaggle.api.dataset_download_files(
            'tboyle10/medicaltranscriptions',
            path=str(data_dir),
            unzip=True
        )
        print(f"Dataset downloaded successfully to {data_dir}")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nMake sure you have:")
        print("1. Installed kaggle package: pip install kaggle")
        print("2. Set up your Kaggle API credentials in ~/.kaggle/kaggle.json")
        print("3. Given appropriate permissions: chmod 600 ~/.kaggle/kaggle.json")

if __name__ == "__main__":
    download_medical_transcriptions()