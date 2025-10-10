import argparse
import os
from pathlib import Path

from download_dataset import download_medical_transcriptions
from prepare_data import prepare_data_for_pretraining
from training import pre_training
from config import LLM_NAME, PROCESSED_DATASET_FILE, MODEL_OUTPUT_DIR

def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the complete pipeline: download data, prepare dataset, and train model'
    )
    
    # Training parameters
    parser.add_argument('--model_name', type=str, default=LLM_NAME["llama"],
                      help='Name of the model to fine-tune')
    parser.add_argument('--max_length', type=int, default=1024,
                      help='Maximum sequence length for tokenization')
    parser.add_argument('--num_train_epochs', type=int, default=1,
                      help='Number of training epochs')
    parser.add_argument('--per_device_train_batch_size', type=int, default=2,
                      help='Training batch size per device')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=2,
                      help='Evaluation batch size per device')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                      help='Learning rate for training')
    
    # Pipeline control
    parser.add_argument('--skip_download', action='store_true',
                      help='Skip the data download step')
    parser.add_argument('--skip_prepare', action='store_true',
                      help='Skip the data preparation step')
    
    return parser.parse_args()

def validate_paths():
    # Check if Kaggle credentials are set
    if not os.getenv("KAGGLE_USERNAME") or not os.getenv("KAGGLE_KEY"):
        raise EnvironmentError("Kaggle credentials are not set. Please add them to your .env file.")

    # Check if raw data directory exists
    raw_data_dir = Path(PROCESSED_DATASET_FILE).parent.parent / "raw"
    if not raw_data_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_data_dir}")

    # Check if processed data directory exists
    processed_data_dir = Path(PROCESSED_DATASET_FILE).parent
    if not processed_data_dir.exists():
        print(f"Processed data directory not found. Creating: {processed_data_dir}")
        processed_data_dir.mkdir(parents=True, exist_ok=True)

    # Check if model output directory exists
    if not Path(MODEL_OUTPUT_DIR).exists():
        print(f"Model output directory not found. Creating: {MODEL_OUTPUT_DIR}")
        Path(MODEL_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def main():
    validate_paths()
    args = parse_args()

    # 1. Download the dataset
    if not args.skip_download:
        print("Step 1: Downloading dataset...")
        try:
            download_medical_transcriptions()
        except Exception as e:
            raise RuntimeError(f"Failed to download dataset: {e}")
    else:
        print("Skipping download step...")

    # 2. Prepare the dataset
    if not args.skip_prepare:
        print("\nStep 2: Preparing dataset...")
        try:
            prepare_data_for_pretraining()
        except Exception as e:
            raise RuntimeError(f"Failed to prepare dataset: {e}")
    else:
        print("Skipping preparation step...")

    # 3. Train the model
    print("\nStep 3: Training model...")
    try:
        pre_training(
            model_name=args.model_name,
            dataset_path=PROCESSED_DATASET_FILE,
            output_dir=MODEL_OUTPUT_DIR,
            max_length=args.max_length,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            learning_rate=args.learning_rate
        )
    except Exception as e:
        raise RuntimeError(f"Failed to train the model: {e}")

if __name__ == "__main__":
    main()