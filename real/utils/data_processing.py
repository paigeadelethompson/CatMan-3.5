import json
import torch
from datasets import Dataset, DatasetDict
from typing import Dict, List, Tuple
import random

def load_json_data(file_path: str) -> List[Dict]:
    """Load and parse JSON data file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def prepare_cat_dataset(
    train_file: str,
    val_file: str,
    tokenizer,
    config: Dict
) -> DatasetDict:
    """
    Prepare dataset for training the cat language model.
    
    The dataset should contain:
    - text: The input text
    - behavior_labels: Cat behavior categories
    - personality_labels: Personality trait indicators
    - decision_labels: Decision-making factors
    """
    
    def process_raw_data(examples):
        # Tokenize the text
        tokenized = tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=config['model']['max_length']
        )
        
        # Process behavior categories
        behavior_mapping = {
            b: i for i, b in enumerate(config['cat_specific']['behavior_categories'])
        }
        
        # Process personality traits
        personality_mapping = {
            t: i for i, t in enumerate(config['cat_specific']['personality_traits'])
        }
        
        # Process decision factors
        decision_mapping = {
            d: i for i, d in enumerate(config['cat_specific']['decision_making_factors'])
        }
        
        # Convert labels to indices
        behavior_labels = [
            behavior_mapping.get(b, 0) for b in examples.get('behaviors', [])
        ]
        personality_labels = [
            personality_mapping.get(t, 0) for t in examples.get('personality_traits', [])
        ]
        decision_labels = [
            decision_mapping.get(d, 0) for d in examples.get('decision_factors', [])
        ]
        
        return {
            **tokenized,
            'behavior_labels': behavior_labels,
            'personality_labels': personality_labels,
            'decision_labels': decision_labels
        }
    
    # Load raw data
    train_data = load_json_data(train_file)
    val_data = load_json_data(val_file)
    
    # Create datasets
    train_dataset = Dataset.from_dict({
        'text': [item['text'] for item in train_data],
        'behaviors': [item.get('behaviors', []) for item in train_data],
        'personality_traits': [item.get('personality_traits', []) for item in train_data],
        'decision_factors': [item.get('decision_factors', []) for item in train_data]
    })
    
    val_dataset = Dataset.from_dict({
        'text': [item['text'] for item in val_data],
        'behaviors': [item.get('behaviors', []) for item in val_data],
        'personality_traits': [item.get('personality_traits', []) for item in val_data],
        'decision_factors': [item.get('decision_factors', []) for item in val_data]
    })
    
    # Process datasets
    train_dataset = train_dataset.map(
        process_raw_data,
        remove_columns=train_dataset.column_names,
        batch_size=1000
    )
    
    val_dataset = val_dataset.map(
        process_raw_data,
        remove_columns=val_dataset.column_names,
        batch_size=1000
    )
    
    return DatasetDict({
        'train': train_dataset,
        'validation': val_dataset
    })

def create_sample_data(output_file: str, config: Dict, num_samples: int = 100):
    """
    Create sample training data for testing the model.
    """
    behaviors = config['cat_specific']['behavior_categories']
    traits = config['cat_specific']['personality_traits']
    factors = config['cat_specific']['decision_making_factors']
    
    sample_data = []
    for _ in range(num_samples):
        sample = {
            'text': f"When my cat encounters {random.choice(['food', 'toys', 'strangers', 'other cats'])}, "
                   f"they typically {random.choice(['play', 'hide', 'investigate', 'meow'])}.",
            'behaviors': [random.choice(behaviors)],
            'personality_traits': [random.choice(traits)],
            'decision_factors': [random.choice(factors)]
        }
        sample_data.append(sample)
    
    with open(output_file, 'w') as f:
        json.dump(sample_data, f, indent=2)

def analyze_dataset_statistics(dataset: Dataset) -> Dict:
    """
    Analyze and return statistics about the dataset.
    """
    stats = {
        'total_samples': len(dataset),
        'behavior_distribution': {},
        'personality_distribution': {},
        'decision_factor_distribution': {}
    }
    
    # Calculate distributions
    for example in dataset:
        for behavior in example['behaviors']:
            stats['behavior_distribution'][behavior] = \
                stats['behavior_distribution'].get(behavior, 0) + 1
                
        for trait in example['personality_traits']:
            stats['personality_distribution'][trait] = \
                stats['personality_distribution'].get(trait, 0) + 1
                
        for factor in example['decision_factors']:
            stats['decision_factor_distribution'][factor] = \
                stats['decision_factor_distribution'].get(factor, 0) + 1
    
    return stats 