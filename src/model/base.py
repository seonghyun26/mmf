from abc import ABC, abstractmethod

from omegaconf import DictConfig, OmegaConf


class ModelWrapper(ABC):
    def __init__(self, cfg: DictConfig, task: str):
        self.cfg = cfg
        self.task = task
        
    @abstractmethod
    def train(self, cfg: DictConfig, task: str):
        pass