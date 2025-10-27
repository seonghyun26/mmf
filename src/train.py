import hydra
import wandb
import logging

from omegaconf import DictConfig, OmegaConf

from .model import *
model_classes = {
    "catboost": Catboost,
    "catboost_gnn": CatboostGNN,
    "catboost_gnn_cell": CatboostGNNCell,
    "gradientboost": GradientBoost,
    "histgradientboost": HistGradientBoost,
    "minimol": MiniMol,
    "mmf_catboostgnn": MMFCatboostGNN,
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
    

def run_model(cfg: DictConfig):
    task_list = OmegaConf.select(cfg, "job.tasks", default=[])
    logging.info(f"Task list: {task_list}")
    
    for task in task_list:
        # try:
        model = load_model(cfg, task)
        model_mode = OmegaConf.select(cfg, "model.mode")
        
        # Train or fine-tune pre-trained model
        if model_mode == "pre-trained":
            logging.info(f"Fine-tuning model for task: {task}")
            
            # Configs
            cfg_copy = OmegaConf.create(OmegaConf.to_yaml(cfg))
            cfg_copy.job.tasks = task
            output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            wandb.init(
                project="admet",
                entity="eddy26",
                config=OmegaConf.to_container(cfg_copy, resolve=True),
                dir=output_dir,
            )
            
            # Fine-tune model
            results = model.fine_tune()
            wandb.log(results)
            wandb.finish()
        
        elif model_mode == "train":
            logging.info(f"Training model for task: {task}")
            
            # Configs
            cfg_copy = OmegaConf.create(OmegaConf.to_yaml(cfg))
            cfg_copy.job.tasks = task
            output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            wandb.init(
                project="admet",
                entity="eddy26",
                config=OmegaConf.to_container(cfg_copy, resolve=True),
                dir=output_dir,
            )
            
            # Train model and get results
            results = model.train()    
            wandb.log(results)
            wandb.finish()

        else:
            logging.warning(f"Invalid model mode: {model_mode}")
            
        # except Exception as e:
        #     logging.exception(f"Error running task {task}: {e}")
        #     wandb.finish()