import hydra
import catboost as cb
import os
import pickle

from tqdm import tqdm
from typing import Any
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import QuantileTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

from tdc.benchmark_group import admet_group
from tdc.metadata import admet_metrics

from .base import ModelWrapper
from ..util import *


class HistGradientBoost(ModelWrapper):
    def __init__(self, cfg: DictConfig, task: str):
        super().__init__(cfg, task)
        self.admet_task_config = admet_task_config
        self.group = admet_group(path = './data/')
        
        
        task_type, task_log_scale = self.admet_task_config[self.task]
        if task_type == "regression":
            params = OmegaConf.to_container(self.cfg.model.params, resolve=True)
            self.model = HistGradientBoostingRegressor(**params)

        elif task_type == "binary":
            params = OmegaConf.to_container(self.cfg.model.params, resolve=True)
            self.model = HistGradientBoostingClassifier(**params)
        
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
                model = HistGradientBoostingRegressor(**self.model.get_params())
            elif task_type == "binary":
                model = HistGradientBoostingClassifier(**self.model.get_params())
            model.set_params(random_state=seed)
            
            benchmark = self.group.get(self.task)
            predictions = {}
            name = benchmark['name']
            train, test = benchmark['train_val'], benchmark['test']
            X_train = get_fingerprints(train['Drug'])
            X_test = get_fingerprints(test['Drug'])
            
            
            if task_type == "regression":
                Y_scaler = scaler(log=task_log_scale)
                Y_scaler.fit(train['Y'].values)
                train['Y_scale'] = Y_scaler.transform(train['Y'].values)
                model.fit(X_train, train['Y_scale'].values)
                y_pred_test = Y_scaler.inverse_transform(model.predict(X_test)).reshape(-1)
            
            elif task_type == "binary":
                model.fit(X_train, train['Y'].values)
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
        with open(f"{save_dir}/{seed}.{format}", 'wb') as f:
            pickle.dump(model, f)



import numpy as np

from sklearn import preprocessing

from rdkit import Chem
from rdkit import RDLogger

from rdkit.Chem import DataStructs
from rdkit.Chem.rdMolDescriptors import GetHashedMorganFingerprint
from rdkit.Avalon.pyAvalonTools import GetAvalonCountFP
from rdkit.Chem import rdReducedGraphs
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator


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


# from https://github.com/rdkit/rdkit/discussions/3863
def count_to_array(fingerprint):
    array = np.zeros((0,), dtype=np.int8)
    
    DataStructs.ConvertToNumpyArray(fingerprint, array)

    return array


def get_avalon_fingerprints(molecules, n_bits=1024):
    fingerprints = molecules.apply(lambda x: GetAvalonCountFP(x, nBits=n_bits))

    fingerprints = fingerprints.apply(count_to_array)
    
    return np.stack(fingerprints.values)


def get_morgan_fingerprints(molecules, n_bits=1024, radius=2):
    fingerprints = molecules.apply(lambda x: 
        GetHashedMorganFingerprint(x, nBits=n_bits, radius=radius))

    fingerprints = fingerprints.apply(count_to_array)
    
    return np.stack(fingerprints.values)


def get_erg_fingerprints(molecules):
    fingerprints = molecules.apply(rdReducedGraphs.GetErGFingerprint)
    
    return np.stack(fingerprints.values)

# from https://www.blopig.com/blog/2022/06/how-to-turn-a-molecule-into-a-vector-of-physicochemical-descriptors-using-rdkit/
def get_rdkit_features(molecules):
    calculator = MolecularDescriptorCalculator(RDKIT_CHOSEN_DESCRIPTORS)

    X_rdkit = molecules.apply(lambda x: np.array(calculator.CalcDescriptors(x)))
    X_rdkit = np.vstack(X_rdkit.values)

    return X_rdkit


def get_fingerprints(smiles):
    RDLogger.DisableLog('rdApp.*')
    molecules = smiles.apply(Chem.MolFromSmiles)
    
    fingerprints = []

    fingerprints.append(get_morgan_fingerprints(molecules))
    fingerprints.append(get_avalon_fingerprints(molecules))
    fingerprints.append(get_erg_fingerprints(molecules))
    fingerprints.append(get_rdkit_features(molecules))

    return np.concatenate(fingerprints, axis=1)