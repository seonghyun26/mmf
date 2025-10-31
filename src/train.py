import hydra
import wandb
import logging

from omegaconf import DictConfig, OmegaConf

from .model import *
model_classes = {
    "maplight": Maplight,
    "maplight_gnn": MaplightGNN,
    "gradientboost": GradientBoost,
    "histgradientboost": HistGradientBoost,
    "minimol": MiniMol,
    "maplight_gnn_ours": MaplightOurs,
}



def load_model(cfg: DictConfig, task: str):
    model_name = OmegaConf.select(cfg, "model.name", default="maplight_gnn")
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
    DEBUG_MODE = cfg.job.debug
    
    for task in task_list:
        if DEBUG_MODE:
            model = load_model(cfg, task)
            model_mode = OmegaConf.select(cfg, "model.mode")
            
            if model_mode == "pretrain":
                logging.info(f"Pretraining model for task: {task}")
                
                # Configs
                cfg_copy = OmegaConf.create(OmegaConf.to_yaml(cfg))
                cfg_copy.job.tasks = task
                output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
                wandb.init(
                    project="admet",
                    entity="eddy26",
                    config=OmegaConf.to_container(cfg_copy, resolve=True),
                    dir=output_dir,
                    tags=["debug", cfg.model.name,]
                )
                
                # Fine-tune model
                results = model.pretrain()
                wandb.log(results)
                results = model.train()
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
                    tags=["debug", cfg.model.name]
                )
                
                # Train model and get results
                results = model.train()    
                wandb.log(results)
                wandb.finish()

            else:
                logging.warning(f"Invalid model mode: {model_mode}")    
        else:
            try:
                model = load_model(cfg, task)
                model_mode = OmegaConf.select(cfg, "model.mode")
                
                if model_mode == "pre-train":
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
                        tags=[cfg.model.name]
                    )
                    
                    # Fine-tune model
                    results = model.pre_train()
                    wandb.log(results)
                    results = model.train()
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
                        tags=[cfg.model.name]
                    )
                    
                    # Train model and get results
                    results = model.train()    
                    wandb.log(results)
                    wandb.finish()

                else:
                    logging.warning(f"Invalid model mode: {model_mode}")
                
            except Exception as e:
                logging.exception(f"Error running task {task}: {e}")
                wandb.finish()