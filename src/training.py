from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from config import LLM_NAME, PROCESSED_DATASET_FILE, MODEL_OUTPUT_DIR

def pre_training(
    model_name: str = LLM_NAME,
    dataset_path: str = PROCESSED_DATASET_FILE,
    output_dir: str = MODEL_OUTPUT_DIR,
    max_length: int = 1024,
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 2,
    per_device_eval_batch_size: int = 2,
    learning_rate: float = 2e-5,
):
    """
    Run continued pretraining on a prepared medical dataset for LLaMA or other causal LMs.
    
    Args:
        model_name (str): Hugging Face model checkpoint (e.g., LLaMA).
        dataset_path (str): Path to dataset prepared by prepare_dataset.py.
        output_dir (str): Where to save the trained model.
        max_length (int): Max sequence length for tokenization.
        num_train_epochs (int): Training epochs.
        per_device_train_batch_size (int): Batch size for training.
        per_device_eval_batch_size (int): Batch size for evaluation.
        learning_rate (float): Optimizer learning rate.
    """
    
    # Load dataset
    dataset = load_from_disk(dataset_path)
    
    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # LLaMA tokenizer sometimes doesn’t have a pad token — set it if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # Data collator for CLM (causal LM → mlm=False)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        evaluation_strategy="steps",
        logging_steps=50,
        save_steps=500,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        fp16=True,  # mixed precision if GPU supports
        save_total_limit=2,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,   # using full dataset since it's unsupervised
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Start training
    trainer.train()
    
    # Save model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"✅ Continued pretraining complete. Model saved at: {output_dir}")


if __name__ == "__main__":
    pre_training()
