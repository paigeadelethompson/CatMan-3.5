import os
import yaml
import torch
import logging
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from utils.data_processing import prepare_cat_dataset
from utils.model_utils import CatLanguageModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_model(config):
    # Initialize tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(config['model']['base_model'])
    model = CatLanguageModel.from_pretrained(
        config['model']['base_model'],
        config=config
    )

    # Add special tokens for cat-specific behaviors
    special_tokens = {
        'additional_special_tokens': [
            f"<{behavior}>" for behavior in config['cat_specific']['behavior_categories']
        ] + [
            f"<{trait}>" for trait in config['cat_specific']['personality_traits']
        ]
    }
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))

    # Load and prepare dataset
    dataset = prepare_cat_dataset(
        config['data']['train_file'],
        config['data']['validation_file'],
        tokenizer,
        config
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./models/cat_lm",
        num_train_epochs=config['training']['num_epochs'],
        per_device_train_batch_size=config['training']['batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=config['training']['learning_rate'],
        warmup_steps=config['training']['warmup_steps'],
        weight_decay=config['training']['weight_decay'],
        logging_dir="./logs",
        logging_steps=100,
        save_steps=config['training']['save_steps'],
        eval_steps=config['training']['eval_steps'],
        evaluation_strategy="steps",
        load_best_model_at_end=True,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        ),
    )

    # Train the model
    logger.info("Starting training...")
    trainer.train()

    # Save the final model
    trainer.save_model("./models/cat_lm_final")
    tokenizer.save_pretrained("./models/cat_lm_final")
    logger.info("Training completed and model saved!")

if __name__ == "__main__":
    config = load_config("configs/model_config.yaml")
    train_model(config) 