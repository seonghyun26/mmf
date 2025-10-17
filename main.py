import os
import wandb
import hydra
import logging

import numexpr as ne

from typing import Dict, Any
from omegaconf import DictConfig, OmegaConf
from src import *


def run_model(cfg: DictConfig, model: Any):
    task_list = OmegaConf.select(cfg, "job.task", default=["caco2_wang"])
    logging.info(f"Task list: {task_list}")
    
    for task in task_list:
        logging.info(f"Running model for task: {task}")
        
        # Train or fine-tune pre-trained model
        model_mode = OmegaConf.select(cfg, "model.mode", default="train")
        results = {}
        if model_mode == "pre-trained":
            logging.info("Fine-tuning pre-trained model per task")
            results.update(fine_tune_model(cfg, model, task))
        
        elif model_mode == "train":
            logging.info("Training model per task")
            results.update(train_model(cfg, model, task))

        else:
            raise ValueError(f"Invalid model mode: {model_mode}")
        
        wandb.log(results)

@hydra.main(
    config_path="config",
    config_name="basic",
    version_base=None
)
def main(cfg: DictConfig) -> None:
    # Check configs and set logging
    run_dir = os.getcwd()
    wandb.init(
        project="admet",
        entity="eddy26",
        config=OmegaConf.to_container(cfg, resolve=True),
        dir=run_dir,
    )
    logging.basicConfig(
        format='%(asctime)s - %(module)s - %(levelname)s - %(message)s',
        level=logging.DEBUG,
        datefmt='%m/%d/%Y %I:%M:%S %p',
    )
    logging.info(f"Working directory : {os.getcwd()}")
    logging.info(f"Output directory  : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")
    logging.info(cfg)
    
    # Run model
    model = load_model(cfg)
    run_model(cfg, model)
    wandb.finish()

if __name__ == "__main__":
    ne.set_num_threads(16)
    main()