import hydra
import logging
import os


from typing import Any
from omegaconf import DictConfig, OmegaConf


admet_task_config = {
    'caco2_wang': ('regression', False),
    'bioavailability_ma': ('binary', False),
    'lipophilicity_astrazeneca': ('regression', False),
    'solubility_aqsoldb': ('regression', False),
    'hia_hou': ('binary', False),
    'pgp_broccatelli': ('binary', False),
    'bbb_martins': ('binary', False),
    'ppbr_az': ('regression', False),
    'vdss_lombardo': ('regression', True),
    'cyp2c9_veith': ('binary', False),
    'cyp2d6_veith': ('binary', False),
    'cyp3a4_veith': ('binary', False),
    'cyp2c9_substrate_carbonmangels': ('binary', False),
    'cyp2d6_substrate_carbonmangels': ('binary', False),
    'cyp3a4_substrate_carbonmangels': ('binary', False),
    'half_life_obach': ('regression', True),
    'clearance_hepatocyte_az': ('regression', True),
    'clearance_microsome_az': ('regression', True),
    'ld50_zhu': ('regression', False),
    'herg': ('binary', False),
    'ames': ('binary', False),
    'dili': ('binary', False)
}

class ModelWrapper():
    def __init__(self, model: Any):
        self.model = model
        
    def train(self, cfg: DictConfig, task: str):
        pass


def load_model(cfg: DictConfig):
    model_name = OmegaConf.select(cfg, "model.name", default="catboost")
    logging.info(f"Loading model: {model_name}")
    if model_name == "catboost":
        model = ModelWrapper()
        
    elif model_name == "something":
        pass
    
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    
    return model


def train_model(cfg: DictConfig, model: Any, task: str):
    max_seed = OmegaConf.select(cfg, "job.max_seed", default=5)
    output_dir = f"{hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}/model"
    os.makedirs(output_dir, exist_ok=True)
    
    # TODO: Implement training 
    # if task
    # model_name = OmegaConf.select(cfg, "model.name", default="catboost")
    
    pass

def fine_tune_model(cfg: DictConfig, model: Any, task: str):
    output_dir = f"{hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}/model"
    os.makedirs(output_dir, exist_ok=True)
    pass