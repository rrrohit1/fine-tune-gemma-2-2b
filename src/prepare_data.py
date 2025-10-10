import pandas as pd
from datasets import Dataset
from config import MEDICAL_TRANSCRIPTIONS_FILE, PROCESSED_DATASET_FILE
import os
from pathlib import Path

def prepare_data_for_pretraining(csv_path: str = MEDICAL_TRANSCRIPTIONS_FILE, 
                               output_path: str = PROCESSED_DATASET_FILE):
    """
    Prepares mtsamples.csv for continued pretraining with a causal LM (like LLaMA).
    
    Args:
        csv_path (str): Path to the mtsamples.csv file.
        output_path (str): Directory where the processed dataset will be saved.
    
    Returns:
        dataset (Dataset): Hugging Face Dataset object with 'text' column.
    """
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Keep only transcription column
    if "transcription" not in df.columns:
        raise ValueError("Expected 'transcription' column in dataset.")
    
    df = df.dropna(subset=["transcription"])
    df = df[["transcription"]].rename(columns={"transcription": "text"})
    
    # Convert to Hugging Face Dataset
    dataset = Dataset.from_pandas(df)
    
    # Save in Hugging Face format
    dataset.save_to_disk(output_path)
    
    print(f"✅ Dataset prepared and saved at: {output_path}")
    print(dataset)
    
    return dataset

def validate_paths(csv_path: str, output_path: str):
    # Check if the input CSV file exists
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Input CSV file not found: {csv_path}")

    # Check if the output directory exists, create if not
    output_dir = Path(output_path).parent
    if not output_dir.exists():
        print(f"Output directory not found. Creating: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

# Wrap the main logic with error handling
if __name__ == "__main__":
    try:
        validate_paths(MEDICAL_TRANSCRIPTIONS_FILE, PROCESSED_DATASET_FILE)
        dataset = prepare_data_for_pretraining()
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
    except ValueError as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
