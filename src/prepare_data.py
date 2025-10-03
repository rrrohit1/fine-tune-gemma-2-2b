import pandas as pd
from datasets import Dataset
from config import MEDICAL_TRANSCRIPTIONS_FILE, PROCESSED_DATASET_FILE

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


if __name__ == "__main__":
    # Process and save the dataset
    dataset = prepare_data_for_pretraining()
