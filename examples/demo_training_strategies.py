#!/usr/bin/env python3
"""
Demonstration script showing how to use different training strategies.
"""

import os
import sys
import logging
from omegaconf import DictConfig, OmegaConf

# Add parent directory to path to import benchmark modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training import TrainerFactory, PerTaskTrainer, MultiTaskTrainer, FineTuneTrainer
from src.model import load_model


def setup_logging():
    """Setup basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def create_sample_config(training_type: str) -> DictConfig:
    """Create sample configuration for different training types."""
    
    base_config = {
        "model": {
            "name": "catboost",
            "mode": "train",
            "mmf": False
        },
        "job": {
            "max_seed": 3,
            "task": ["caco2_wang", "bioavailability_ma"]
        },
        "data": {
            "path": "data/",
            "benchmark": "caco2_wang"
        },
        "device": "cpu"
    }
    
    if training_type == "per_task":
        base_config["model"]["train"] = {"strategy": "per_task"}
        
    elif training_type == "multi_task":
        base_config["model"]["name"] = "mmf"
        base_config["model"]["mmf"] = True
        base_config["model"]["train"] = {"strategy": "multi_task"}
        
    elif training_type == "fine_tune":
        base_config["model"]["mode"] = "pre-trained"
        base_config["model"]["pretrained_path"] = "models/pretrained.pkl"
        base_config["model"]["fine_tune"] = {
            "learning_rate": 1e-4,
            "epochs": 5,
            "data_ratio": 0.8
        }
        base_config["model"]["freeze_layers"] = []
        
    return OmegaConf.create(base_config)


def demo_trainer_selection():
    """Demonstrate automatic trainer selection."""
    print("=== Trainer Selection Demo ===")
    
    training_types = ["per_task", "multi_task", "fine_tune"]
    
    for training_type in training_types:
        print(f"\n--- {training_type.upper()} Training ---")
        
        # Create configuration
        cfg = create_sample_config(training_type)
        print(f"Config: {OmegaConf.to_yaml(cfg)}")
        
        # Create trainer
        trainer = TrainerFactory.create_trainer(cfg)
        print(f"Selected trainer: {trainer.__class__.__name__}")
        
        # Show trainer info
        print(f"Trainer description: {trainer.__class__.__doc__.strip()}")


def demo_manual_trainer_creation():
    """Demonstrate manual trainer creation."""
    print("\n=== Manual Trainer Creation Demo ===")
    
    cfg = create_sample_config("per_task")
    
    # Create different trainers manually
    trainers = [
        PerTaskTrainer(cfg),
        MultiTaskTrainer(cfg),
        FineTuneTrainer(cfg)
    ]
    
    for trainer in trainers:
        print(f"\n{trainer.__class__.__name__}:")
        print(f"  Description: {trainer.__class__.__doc__.strip()}")
        print(f"  Output directory: {trainer.output_dir}")
        
        # Validate configuration (some may fail, which is expected)
        try:
            trainer.validate_config()
            print("  ✓ Configuration valid")
        except Exception as e:
            print(f"  ✗ Configuration error: {e}")


def demo_factory_info():
    """Demonstrate factory information methods."""
    print("\n=== Factory Information Demo ===")
    
    # Show available trainers
    available = TrainerFactory.get_available_trainers()
    print(f"Available trainers: {list(available.keys())}")
    
    # Show detailed info for each trainer
    for name in available.keys():
        info = TrainerFactory.get_trainer_info(name)
        print(f"\n{name}:")
        print(f"  Class: {info['class']}")
        print(f"  Module: {info['module']}")
        print(f"  Description: {info['docstring'][:100]}...")


def main():
    """Run all demonstrations."""
    setup_logging()
    
    print("Training Strategy Architecture Demo")
    print("=" * 50)
    
    try:
        demo_trainer_selection()
        demo_manual_trainer_creation()
        demo_factory_info()
        
        print("\n" + "=" * 50)
        print("Demo completed successfully!")
        print("\nTo use in your code:")
        print("1. Create configuration (YAML or programmatically)")
        print("2. trainer = TrainerFactory.create_trainer(cfg)")
        print("3. result = trainer.train(model)")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
