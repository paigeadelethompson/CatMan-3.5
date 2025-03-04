import os
import yaml
import json
import random
from typing import List, Dict

def load_config(config_path: str) -> Dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def generate_cat_scenario() -> Dict:
    """Generate a random cat scenario with behaviors and traits."""
    scenarios = [
        "When encountering {stimulus}, my cat {action}.",
        "During {time_of_day}, my cat usually {action} when {stimulus} happens.",
        "If there's {stimulus} nearby, my cat tends to {action}.",
        "My cat's typical response to {stimulus} is to {action}.",
        "When {condition}, my cat prefers to {action}."
    ]

    stimuli = [
        "food", "toys", "strangers", "other cats", "loud noises",
        "birds outside", "unfamiliar objects", "their favorite treats",
        "the sound of their food bowl", "new furniture", "cardboard boxes",
        "the vacuum cleaner", "their reflection", "running water",
        "their favorite human"
    ]

    actions = [
        "investigate cautiously",
        "hide under furniture",
        "meow repeatedly",
        "approach slowly",
        "run away quickly",
        "show curiosity",
        "become very alert",
        "purr contentedly",
        "watch intently",
        "play enthusiastically",
        "groom themselves",
        "seek attention",
        "stretch and yawn",
        "paw at it playfully",
        "ignore it completely"
    ]

    conditions = [
        "it's raining outside",
        "visitors come over",
        "the house is quiet",
        "it's feeding time",
        "the sun is setting",
        "there's movement outside",
        "someone is cooking",
        "the lights are dim",
        "it's very early morning",
        "there's a storm"
    ]

    times = [
        "morning", "afternoon", "evening", "nighttime",
        "meal times", "playtime", "quiet hours", "dawn", "dusk"
    ]

    template = random.choice(scenarios)
    
    if "{time_of_day}" in template:
        text = template.format(
            time_of_day=random.choice(times),
            stimulus=random.choice(stimuli),
            action=random.choice(actions)
        )
    elif "{condition}" in template:
        text = template.format(
            condition=random.choice(conditions),
            action=random.choice(actions)
        )
    else:
        text = template.format(
            stimulus=random.choice(stimuli),
            action=random.choice(actions)
        )

    return text

def generate_dataset(config: Dict, num_samples: int) -> List[Dict]:
    """Generate a dataset of cat behavior scenarios."""
    behaviors = config['cat_specific']['behavior_categories']
    traits = config['cat_specific']['personality_traits']
    factors = config['cat_specific']['decision_making_factors']
    
    dataset = []
    for _ in range(num_samples):
        # Generate base scenario
        text = generate_cat_scenario()
        
        # Randomly assign behaviors, traits, and decision factors
        num_behaviors = random.randint(1, 3)
        num_traits = random.randint(1, 2)
        num_factors = random.randint(1, 3)
        
        sample = {
            'text': text,
            'behaviors': random.sample(behaviors, num_behaviors),
            'personality_traits': random.sample(traits, num_traits),
            'decision_factors': random.sample(factors, num_factors)
        }
        
        dataset.append(sample)
    
    return dataset

def main():
    # Load configuration
    config = load_config('../configs/model_config.yaml')
    
    # Create data directory if it doesn't exist
    os.makedirs('../data', exist_ok=True)
    
    # Generate training data
    train_data = generate_dataset(config, num_samples=1000)
    val_data = generate_dataset(config, num_samples=200)
    test_data = generate_dataset(config, num_samples=100)
    
    # Save datasets
    for name, data in [
        ('train', train_data),
        ('validation', val_data),
        ('test', test_data)
    ]:
        output_file = f'../data/cat_{name}_data.json'
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Generated {len(data)} samples for {name} set")

if __name__ == '__main__':
    main() 