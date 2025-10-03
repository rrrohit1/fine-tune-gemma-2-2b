# Fine-tuning Language Models on Medical Transcriptions

This project fine-tunes large language models (LLaMA and Gemma) on medical transcriptions to improve their capabilities in medical text generation and understanding. The project supports both Google's Gemma 2B and Meta's LLaMA models.

## Project Structure

```bash
├── data/
│   ├── raw/              # Raw medical transcriptions data
│   │   └── mtsamples.csv
│   └── processed/        # Processed dataset for training
├── models/              # Directory for saving trained models
├── notebooks/          
│   └── starter.ipynb    # Data exploration notebook
├── src/
│   ├── config.py        # Configuration and paths
│   ├── download_dataset.py  # Dataset download script
│   ├── prepare_data.py  # Data preprocessing
│   ├── training.py      # Model training script
│   └── main.py         # Main pipeline script
└── environment.yaml    # Conda environment file
```

## Setup

1. Create and activate the conda environment:

```bash
conda env create -f environment.yaml
conda activate gemma-env
```

1. Install required packages:

```bash
pip install transformers datasets pandas numpy matplotlib seaborn python-dotenv kaggle jupyter notebook bitsandbytes peft trl
```

1. Set up Kaggle credentials:
   - Get your Kaggle API token from your Kaggle account settings
   - Create a `.env` file in the project root with:

```bash
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_key
```

## Usage

### Data Exploration

- Check `notebooks/starter.ipynb` for detailed data analysis and visualization

### Running the Pipeline

The complete pipeline can be run using the main script:

```bash
python src/main.py
```

Optional arguments:

- `--model_name`: Model to fine-tune (default: google/gemma-2b, also supports meta-llama/Llama-2-7b-hf)
- `--max_length`: Maximum sequence length (default: 1024)
- `--num_train_epochs`: Number of training epochs (default: 1)
- `--per_device_train_batch_size`: Training batch size (default: 2)
- `--per_device_eval_batch_size`: Evaluation batch size (default: 2)
- `--learning_rate`: Learning rate (default: 2e-5)
- `--skip_download`: Skip downloading the dataset
- `--skip_prepare`: Skip data preparation

### Running Individual Steps

1. Download the dataset:

```bash
python src/download_dataset.py
```

1. Prepare the data:

```bash
python src/prepare_data.py
```

1. Train the model:

```bash
python src/training.py
```

## Data

The project uses the [Medical Transcriptions](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions) dataset from Kaggle, which contains:

- Medical transcriptions across various specialties
- Descriptions and keywords for each transcription
- Different types of medical documentation

## Models

The project supports two powerful language models:

1. **Google's Gemma 2B**: A lightweight yet powerful model, ideal for efficient fine-tuning
2. **Meta's LLaMA**: A highly capable model known for its strong performance on domain-specific tasks

Both models can be fine-tuned using causal language modeling on the medical transcriptions. The choice of model can be specified through command-line arguments or in the configuration file.

### Default Models Available

- `google/gemma-2b`: Gemma 2B base model
- `meta-llama/Llama-3.2-1B`: LLaMA3.2 1B base model

### Training Libraries

The following libraries are used for efficient training:

- `bitsandbytes` for quantization
- `peft` for LoRA adapters
- `transformers` for loading the model
- `datasets` for loading and using the fine-tuning dataset
- `trl` for the trainer class

## License

This project is licensed under the terms of the LICENSE file included in the repository.
