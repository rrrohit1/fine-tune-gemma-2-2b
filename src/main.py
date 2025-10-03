import argparse
from download_dataset import download_medical_transcriptions
from prepare_data import prepare_data_for_pretraining
from training import pre_training
from config import LLM_NAME, PROCESSED_DATASET_FILE, MODEL_OUTPUT_DIR

def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the complete pipeline: download data, prepare dataset, and train model'
    )
    
    # Training parameters
    parser.add_argument('--model_name', type=str, default=LLM_NAME,
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

def main():
    args = parse_args()
    
    # 1. Download the dataset
    if not args.skip_download:
        print("Step 1: Downloading dataset...")
        download_medical_transcriptions()
    else:
        print("Skipping download step...")
    
    # 2. Prepare the dataset
    if not args.skip_prepare:
        print("\nStep 2: Preparing dataset...")
        prepare_data_for_pretraining()
    else:
        print("Skipping preparation step...")
    
    # 3. Train the model
    print("\nStep 3: Training model...")
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

if __name__ == "__main__":
    main()