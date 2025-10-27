import os
import hydra
import numpy as np
import catboost as cb

from tqdm import tqdm
from typing import Any
from sklearn import preprocessing
from omegaconf import DictConfig, OmegaConf

from tdc.benchmark_group import admet_group
from tdc.metadata import admet_metrics


from .base import ModelWrapper
from ..util import *



class CatboostGNNCell(ModelWrapper):
    def __init__(self, cfg: DictConfig, task: str):
        super().__init__(cfg, task)
        self.admet_task_config = admet_task_config
        self.group = admet_group(path = './data/')
        
        
        task_type, task_log_scale = self.admet_task_config[self.task]
        if task_type == "regression":
            params = OmegaConf.to_container(self.cfg.model.params, resolve=True)
            params['loss_function'] = 'MAE'
            self.model = cb.CatBoostRegressor(**params)
        
        elif task_type == "binary":
            params = OmegaConf.to_container(self.cfg.model.params, resolve=True)
            params['loss_function'] = 'Logloss'
            self.model = cb.CatBoostClassifier(**params)
        
        else:
            raise ValueError(f"Invalid task type: {task_type}")
    

    def train(self):
        metric_name = admet_metrics.get(self.task, )
        predictions_list = []
        results = {}
        plot_dir = f"{hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}/{self.task}"
        os.makedirs(plot_dir, exist_ok=True)
        
        for seed in tqdm(range(self.cfg.job.max_seed)):
            # Initialize a fresh model for each seed to ensure proper randomization
            task_type, task_log_scale = self.admet_task_config[self.task]
            if task_type == "regression":
                model = cb.CatBoostRegressor(**self.model.get_params())
            elif task_type == "binary":
                model = cb.CatBoostClassifier(**self.model.get_params())
            model.set_params(random_seed=seed)
            plot_file = f"{hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}/{self.task}/{seed}.html"
            
            benchmark = self.group.get(self.task)
            predictions = {}
            name = benchmark['name']
            train, test = benchmark['train_val'], benchmark['test']
            X_train = FingerprintManagerGIN(self.cfg.model.fingerprint, self.task, self.cfg.model.name, "train", train['Drug']).fingerprints
            X_test = FingerprintManagerGIN(self.cfg.model.fingerprint, self.task, self.cfg.model.name, "test", test['Drug']).fingerprints
            
            if task_type == "regression":
                Y_scaler = scaler(log=task_log_scale)
                Y_scaler.fit(train['Y'].values)
                train['Y_scale'] = Y_scaler.transform(train['Y'].values)
                model.fit(
                    X_train, train['Y_scale'].values,
                    plot=self.cfg.model.fit.plot,
                    plot_file=plot_file
                )
                y_pred_test = Y_scaler.inverse_transform(model.predict(X_test)).reshape(-1)
            
            elif task_type == "binary":
                model.fit(
                    X_train, train['Y'].values,
                    plot=self.cfg.model.fit.plot,
                    plot_file=plot_file
                )
                y_pred_test = model.predict_proba(X_test)[:, 1]
            
            predictions[name] = y_pred_test
            single_result = self.group.evaluate(predictions)[self.task]
            single_result[f"{metric_name}/{seed}"] = single_result.pop(metric_name)
            results.update(single_result)
            predictions_list.append(predictions)
            
            # Save model for each seed
            self.save(model, seed)
        
        averaged_results = self.group.evaluate_many(predictions_list)[self.task]
        results.update({
            f"{metric_name}/mean": averaged_results[0],
            f"{metric_name}/std": averaged_results[1],
        })
        
        return results

    def save(self, model: Any, seed: int):
        save_dir = f"{hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}/{self.cfg.model.name}"
        os.makedirs(save_dir, exist_ok=True)
        format = self.cfg.model.format
        model.save_model(
            fname=f"{save_dir}/{seed}.{format}",
            format=format,
            export_parameters=None,
            pool=None
        )


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

