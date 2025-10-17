import hydra
import logging
import os
from abc import ABC, abstractmethod

from typing import Any
from omegaconf import DictConfig, OmegaConf


from .model import *
model_classes = {
    "catboost": Catboost,
    # Add new models here as you create them
    # "xgboost": XGBoost,
    # "randomforest": RandomForest,
    # "neuralnetwork": NeuralNetwork,
}


def load_model(cfg: DictConfig, task: str):
    model_name = OmegaConf.select(cfg, "model.name", default="catboost")
    logging.info(f"Loading model: {model_name}")
    
    if model_name in model_classes:
        model = model_classes[model_name](
            cfg=cfg,
            task=task,
        )
        return model
    
    else:
        available_models = list(model_classes.keys())
        raise ValueError(f"Invalid model name: {model_name}. Available models: {available_models}")
    

def train_model(cfg: DictConfig, model_wrapper: Any, task: str):
    output_dir = f"{hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}/model"
    os.makedirs(output_dir, exist_ok=True)    
    
    logging.info(f"Training model for task: {task}")
    trained_model, results = model_wrapper.train()    
    # model_wrapper.save(output_dir)
    
    return results

def fine_tune_model(cfg: DictConfig, model_wrapper: Any, task: str):
    output_dir = f"{hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}/model"
    os.makedirs(output_dir, exist_ok=True)
    
    logging.info(f"Fine-tuning model for task: {task}")
    finetuned_model, results = model_wrapper.fine_tune()
    print(results)
    
    model_wrapper.save(output_dir)
    
    return results