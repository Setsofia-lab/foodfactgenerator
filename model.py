import pandas as pd
import torch
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
import os
import json
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FoodFactTrainer:
    def __init__(self, model_name="gpt2", output_dir="./gpt2-food-fact-model"):
        """
        Initialize the Food Fact Generation Trainer
        
        Args:
            model_name (str): Base model to fine-tune
            output_dir (str): Directory to save the trained model
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Special tokens for better training
        self.special_tokens = {
            "pad_token": "<|pad|>",
            "eos_token": "<|endoftext|>",
            "bos_token": "<|startoftext|>",
            "sep_token": " -> "
        }
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        
    def load_and_prepare_data(self, data_path="food_facts_training_dataset.csv"):
        """
        Load and prepare the training data
        
        Args:
            data_path (str): Path to the training dataset CSV
            
        Returns:
            DatasetDict: Train and validation datasets
        """
        logger.info(f"Loading data from {data_path}")
        
        # Load the dataset
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} training examples")
        
        # Clean and filter data
        df = df.dropna().reset_index(drop=True)
        df = df[df['input'].str.len() > 2]  # Remove very short inputs
        df = df[df['output'].str.len() > 10]  # Remove very short outputs
        
        logger.info(f"After cleaning: {len(df)} training examples")
        
        # Split into train and validation
        train_texts, val_texts = train_test_split(
            df, 
            test_size=0.1, 
            random_state=42,
            stratify=None
        )
        
        logger.info(f"Train: {len(train_texts)}, Validation: {len(val_texts)}")
        
        # Create datasets
        train_dataset = Dataset.from_pandas(train_texts)
        val_dataset = Dataset.from_pandas(val_texts)
        
        return DatasetDict({
            'train': train_dataset,
            'validation': val_dataset
        })
    
    def setup_tokenizer(self):
        """
        Setup and configure the tokenizer
        
        Returns:
            GPT2Tokenizer: Configured tokenizer
        """
        logger.info(f"Loading tokenizer: {self.model_name}")
        
        tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        
        # Add special tokens
        special_tokens_dict = {
            'pad_token': self.special_tokens['pad_token'],
            'bos_token': self.special_tokens['bos_token'],
            'eos_token': self.special_tokens['eos_token']
        }
        
        num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
        logger.info(f"Added {num_added_tokens} special tokens")
        
        return tokenizer
    
    def tokenize_function(self, examples, tokenizer, max_length=256):
        """
        Tokenize the input-output pairs
        
        Args:
            examples: Batch of examples from dataset
            tokenizer: The tokenizer to use
            max_length: Maximum sequence length
            
        Returns:
            dict: Tokenized examples
        """
        # Create input-output format with special tokens
        texts = []
        for inp, out in zip(examples['input'], examples['output']):
            # Format: <|startoftext|>Input: {input} -> Output: {output}<|endoftext|>
            formatted_text = (
                f"{self.special_tokens['bos_token']}"
                f"Input: {inp}{self.special_tokens['sep_token']}"
                f"Output: {out}"
                f"{self.special_tokens['eos_token']}"
            )
            texts.append(formatted_text)
        
        # Tokenize
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        # For causal language modeling, labels are the same as input_ids
        tokenized['labels'] = tokenized['input_ids'].clone()
        
        return tokenized
    
    def setup_model(self, tokenizer):
        """
        Setup and configure the model
        
        Args:
            tokenizer: The tokenizer (to resize embeddings)
            
        Returns:
            GPT2LMHeadModel: Configured model
        """
        logger.info(f"Loading model: {self.model_name}")
        
        model = GPT2LMHeadModel.from_pretrained(self.model_name)
        
        # Resize token embeddings to account for new special tokens
        model.resize_token_embeddings(len(tokenizer))
        
        # Move to device
        model = model.to(self.device)
        
        logger.info(f"Model loaded with {model.num_parameters()} parameters")
        
        return model
    
    def create_training_arguments(self, num_train_examples):
        """
        Create training arguments with optimal settings
        
        Args:
            num_train_examples: Number of training examples
            
        Returns:
            TrainingArguments: Configured training arguments
        """
        # Calculate steps
        batch_size = 8 if torch.cuda.is_available() else 4
        num_epochs = 3
        
        steps_per_epoch = num_train_examples // batch_size
        total_steps = steps_per_epoch * num_epochs
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{self.output_dir}_{timestamp}"
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            
            # Training parameters
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=2,
            
            # Optimization
            learning_rate=5e-5,
            weight_decay=0.01,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            max_grad_norm=1.0,
            
            # Scheduling
            lr_scheduler_type="cosine",
            warmup_steps=steps_per_epoch // 4,  # 25% of first epoch
            
            # Evaluation and saving
            eval_strategy="steps",
            eval_steps=steps_per_epoch // 2,  # Evaluate twice per epoch
            save_strategy="steps",
            save_steps=steps_per_epoch // 2,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            
            # Logging
            logging_steps=50,
            logging_dir=f"{output_dir}/logs",
            report_to=None,  # Disable wandb/tensorboard for now
            
            # Hardware optimization
            fp16=torch.cuda.is_available(),
            dataloader_pin_memory=True,
            dataloader_num_workers=4 if torch.cuda.is_available() else 0,
            
            # Reproducibility
            seed=42,
            data_seed=42,
        )
        
        self.final_output_dir = output_dir
        return training_args
    
    def train_model(self, data_path="food_facts_training_dataset.csv"):
        """
        Complete training pipeline
        
        Args:
            data_path (str): Path to training data
            
        Returns:
            tuple: (trainer, tokenizer, model)
        """
        logger.info("Starting training pipeline...")
        
        # 1. Load and prepare data
        datasets = self.load_and_prepare_data(data_path)
        
        # 2. Setup tokenizer
        tokenizer = self.setup_tokenizer()
        
        # 3. Tokenize datasets
        logger.info("Tokenizing datasets...")
        tokenized_datasets = datasets.map(
            lambda examples: self.tokenize_function(examples, tokenizer),
            batched=True,
            remove_columns=datasets['train'].column_names
        )
        
        # 4. Setup model
        model = self.setup_model(tokenizer)
        
        # 5. Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # Causal LM, not masked LM
            return_tensors="pt"
        )
        
        # 6. Setup training arguments
        training_args = self.create_training_arguments(len(datasets['train']))
        
        # 7. Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['validation'],
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        # 8. Start training
        logger.info("Starting training...")
        train_result = trainer.train()
        
        # 9. Save the final model and tokenizer
        logger.info(f"Saving model to {self.final_output_dir}")
        trainer.save_model()
        tokenizer.save_pretrained(self.final_output_dir)
        
        # 10. Save training metrics
        metrics = {
            'train_loss': train_result.training_loss,
            'train_steps': train_result.global_step,
            'eval_loss': trainer.evaluate()['eval_loss'],
            'model_name': self.model_name,
            'training_date': datetime.now().isoformat(),
            'num_parameters': model.num_parameters(),
            'dataset_size': len(datasets['train'])
        }
        
        with open(f"{self.final_output_dir}/training_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info("Training completed successfully!")
        logger.info(f"Final training loss: {train_result.training_loss:.4f}")
        logger.info(f"Model saved to: {self.final_output_dir}")
        
        return trainer, tokenizer, model
    
    def generate_sample_predictions(self, tokenizer, model, num_samples=5):
        """
        Generate sample predictions to test the model
        
        Args:
            tokenizer: Trained tokenizer
            model: Trained model
            num_samples: Number of sample predictions to generate
        """
        logger.info("Generating sample predictions...")
        
        # Sample inputs for testing
        test_inputs = [
            "Chocolate Chip Cookies",
            "Greek Yogurt",
            "Organic Quinoa",
            "Pepperoni Pizza",
            "Green Tea"
        ]
        
        model.eval()
        
        for i, input_text in enumerate(test_inputs[:num_samples]):
            # Format input
            prompt = f"{self.special_tokens['bos_token']}Input: {input_text}{self.special_tokens['sep_token']}Output: "
            
            # Tokenize
            inputs = tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 100,
                    num_return_sequences=1,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the output part
            if "Output: " in generated_text:
                output_part = generated_text.split("Output: ", 1)[1]
            else:
                output_part = generated_text
            
            print(f"\n--- Sample {i+1} ---")
            print(f"Input: {input_text}")
            print(f"Generated: {output_part}")

def main():
    """
    Main training function
    """
    # Initialize trainer
    trainer = FoodFactTrainer(
        model_name="gpt2",
        output_dir="./gpt2-food-fact-model"
    )
    
    # Train the model
    trained_trainer, tokenizer, model = trainer.train_model("food_facts_training_dataset.csv")
    
    # Generate sample predictions
    trainer.generate_sample_predictions(tokenizer, model)
    
    print("\n" + "="*50)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print(f"Model saved to: {trainer.final_output_dir}")
    print("="*50)

if __name__ == "__main__":
    main()