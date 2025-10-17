# Import all model classes
from .base import ModelWrapper
from .maplight import *

# Export all models for easy importing
__all__ = [
    'ModelWrapper',
    'Catboost',
    # Add new models here as you create them
    # 'XGBoost',
    # 'RandomForest',
    # 'NeuralNetwork',
]