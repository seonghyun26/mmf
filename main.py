import os
import wandb
import hydra
import logging

import numexpr as ne

from typing import Dict, Any
from omegaconf import DictConfig, OmegaConf
from src import *

@hydra.main(
    config_path="config",
    config_name="basic",
    version_base=None
)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        format='%(asctime)s - %(module)s - %(levelname)s - %(message)s',
        level=logging.DEBUG,
        datefmt='%m/%d/%Y %I:%M:%S %p',
    )
    logging.info(f"Working directory : {os.getcwd()}")
    logging.info(f"Output directory  : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")
    logging.info(cfg)
    
    run_model(cfg)

if __name__ == "__main__":
    ne.set_num_threads(16)
    main()