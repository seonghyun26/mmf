import hydra
import os
import pickle
import numpy as np

from tqdm import tqdm
from typing import Any
from omegaconf import DictConfig, OmegaConf

from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

from .base import ModelWrapper
from ..util import *



class GradientBoost(ModelWrapper):
    def __init__(self, cfg: DictConfig, task: str):
        super().__init__(cfg, task)        

        if self.task_type == "regression":
            params = OmegaConf.to_container(self.cfg.model.params, resolve=True)
            self.model = GradientBoostingRegressor(**params)

        elif self.task_type == "binary":
            params = OmegaConf.to_container(self.cfg.model.params, resolve=True)
            self.model = GradientBoostingClassifier(**params)
        
        else:
            raise ValueError(f"Invalid task type: {self.task_type}")
    

    def train(self):
        name = self.benchmark['name']
        train, test = self.benchmark['train_val'], self.benchmark['test']
        X_train = FingerprintManager(self.cfg.model.fingerprint, self.task, self.cfg.model.name, "train", train['Drug']).fingerprints
        X_test = FingerprintManager(self.cfg.model.fingerprint, self.task, self.cfg.model.name, "test", test['Drug']).fingerprints
        
        
        results = {}
        predictions_list = []
        for seed in tqdm(range(self.cfg.job.max_seed)):
            # Initialize a fresh model for each seed to ensure proper randomization
            if self.task_type == "regression":
                model = GradientBoostingRegressor(**self.model.get_params())
            elif self.task_type == "binary":
                model = GradientBoostingClassifier(**self.model.get_params())
            model.set_params(random_state=seed)
            
            predictions = {}
            if self.task_type == "regression":
                Y_scaler = scaler(log=self.task_log_scale)
                Y_scaler.fit(self.benchmark['train_val']['Y'].values)
                train['Y_scale'] = Y_scaler.transform(train['Y'].values)
                model.fit(X_train, train['Y_scale'].values)
                y_pred_test = Y_scaler.inverse_transform(model.predict(X_test)).reshape(-1)
            
            elif self.task_type == "binary":
                model.fit(X_train, train['Y'].values)
                y_pred_test = model.predict_proba(X_test)[:, 1]
            
            predictions[name] = y_pred_test
            single_result = self.group.evaluate(predictions)[self.task]
            single_result[f"{self.metric_name}/{seed}"] = single_result.pop(self.metric_name)
            results.update(single_result)
            predictions_list.append(predictions)
            
            # Save model for each seed
            self.save(model, seed)
        
        averaged_results = self.group.evaluate_many(predictions_list)[self.task]
        results.update({
            f"{self.metric_name}/mean": averaged_results[0],
            f"{self.metric_name}/std": averaged_results[1],
        })
        
        return results

    def save(self, model: Any, seed: int):
        save_dir = f"{hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}/{self.cfg.model.name}"
        os.makedirs(save_dir, exist_ok=True)
        format = self.cfg.model.format
        with open(f"{save_dir}/{seed}.{format}", 'wb') as f:
            pickle.dump(model, f)



class scaler:
    def __init__(self, log=False):
        self.log = log
        self.offset = None
        self.scaler = None

    def fit(self, y):
        # make the values non-negative
        self.offset = np.min([np.min(y), 0.0])
        y = y.reshape(-1, 1) - self.offset

        # scale the input data
        if self.log:
            y = np.log10(y + 1.0)

        self.scaler = preprocessing.StandardScaler().fit(y)

    def transform(self, y):
        y = y.reshape(-1, 1) - self.offset

        # scale the input data
        if self.log:
            y = np.log10(y + 1.0)

        y_scale = self.scaler.transform(y)

        return y_scale

    def inverse_transform(self, y_scale):
        y = self.scaler.inverse_transform(y_scale.reshape(-1, 1))

        if self.log:
            y = 10.0**y - 1.0

        y = y + self.offset

        return y

