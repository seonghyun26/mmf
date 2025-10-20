from abc import ABC, abstractmethod

from omegaconf import DictConfig, OmegaConf


class ModelWrapper(ABC):
    def __init__(self, cfg: DictConfig, task: str):
        self.cfg = cfg
        self.task = task
        self.model = None
        
    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def save(self, save_dir: str):
        pass