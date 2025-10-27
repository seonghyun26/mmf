from abc import ABC, abstractmethod
from omegaconf import DictConfig, OmegaConf

from tdc.benchmark_group import admet_group
from tdc.metadata import admet_metrics

from ..util import *


class ModelWrapper(ABC):
    def __init__(self, cfg: DictConfig, task: str):
        self.cfg = cfg
        self.task = task
        
        self.admet_task_config = admet_task_config
        self.group = admet_group(path = './data/')
        self.benchmark = self.group.get(self.task)
        self.metric_name = admet_metrics.get(self.task)
        self.task_type, self.task_log_scale = self.admet_task_config[self.task]
        
    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def save(self, save_dir: str):
        pass