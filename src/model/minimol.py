import hydra
import os
import math
import logging
import numpy as np

from copy import deepcopy
from typing import Any, Union
from omegaconf import DictConfig, OmegaConf
from contextlib import redirect_stdout, redirect_stderr
from tqdm import tqdm
from importlib import resources


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset


from tdc.benchmark_group import admet_group
from tdc.metadata import admet_metrics


from .base import ModelWrapper

MINIMOL_MODEL_CONFIG = {
    'caco2_wang':                       {'hidden_dim': 2048, 'depth': 4, 'combine': True, 'lr': 0.0005},
    'hia_hou':                          {'hidden_dim': 2048, 'depth': 4, 'combine': True, 'lr': 0.0003},
    'pgp_broccatelli':                  {'hidden_dim': 512,  'depth': 4, 'combine': True, 'lr': 0.0003},
    'bioavailability_ma':               {'hidden_dim': 512,  'depth': 3, 'combine': True, 'lr': 0.0003},
    'lipophilicity_astrazeneca':        {'hidden_dim': 2048, 'depth': 4, 'combine': True, 'lr': 0.0005},
    'solubility_aqsoldb':               {'hidden_dim': 1024, 'depth': 4, 'combine': True, 'lr': 0.0005},
    'bbb_martins':                      {'hidden_dim': 2048, 'depth': 3, 'combine': True, 'lr': 0.0001},
    'ppbr_az':                          {'hidden_dim': 2048, 'depth': 4, 'combine': True, 'lr': 0.0003},
    'vdss_lombardo':                    {'hidden_dim': 1024, 'depth': 4, 'combine': True, 'lr': 0.0001},
    'cyp2d6_veith':                     {'hidden_dim': 512,  'depth': 3, 'combine': True, 'lr': 0.0001},
    'cyp3a4_veith':                     {'hidden_dim': 512,  'depth': 3, 'combine': True, 'lr': 0.0001},
    'cyp2c9_veith':                     {'hidden_dim': 512,  'depth': 3, 'combine': True, 'lr': 0.0001},
    'cyp2d6_substrate_carbonmangels':   {'hidden_dim': 512,  'depth': 3, 'combine': True, 'lr': 0.0001},
    'cyp3a4_substrate_carbonmangels':   {'hidden_dim': 512,  'depth': 3, 'combine': True, 'lr': 0.0001},
    'cyp2c9_substrate_carbonmangels':   {'hidden_dim': 1024, 'depth': 3, 'combine': True, 'lr': 0.0005},
    'half_life_obach':                  {'hidden_dim': 1024, 'depth': 3, 'combine': True, 'lr': 0.0003},
    'clearance_microsome_az':           {'hidden_dim': 1024, 'depth': 4, 'combine': True, 'lr': 0.0005},
    'clearance_hepatocyte_az':          {'hidden_dim': 2048, 'depth': 4, 'combine': True, 'lr': 0.0005},
    'herg':                             {'hidden_dim': 512,  'depth': 3, 'combine': True, 'lr': 0.0003},
    'ames':                             {'hidden_dim': 512,  'depth': 3, 'combine': True, 'lr': 0.0001},
    'dili':                             {'hidden_dim': 512,  'depth': 4, 'combine': True, 'lr': 0.0005},
    'ld50_zhu':                         {'hidden_dim': 1024, 'depth': 4, 'combine': True, 'lr': 0.0001},
}

class MiniMol(ModelWrapper):
    def __init__(self, cfg: DictConfig, task: str):
        super().__init__(cfg, task)
        self.admet_task_config = {
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
        self.group = admet_group(path = './data/')
        self.cfg = cfg
        self.EPOCHS = self.cfg.model.epochs
        self.REPETITIONS = self.cfg.model.repetitions
        self.ENSEMBLE_SIZE = self.cfg.model.ensemble_size
        
        self.model = MinimolEncoder(cfg=self.cfg)
    
    
    def train(self):
        metric_name = admet_metrics.get(self.task, )
        predictions_list = []
        results = {}
        plot_dir = f"{hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}/{self.task}"
        os.makedirs(plot_dir, exist_ok=True)
        
        # LOOP1: ensemble on seeds
        pbar = tqdm(total=self.REPETITIONS * self.ENSEMBLE_SIZE * self.EPOCHS, desc="Training")
        for rep_i, seed1 in enumerate(range(self.REPETITIONS)):
            predictions = {}
            task_type, task_log_scale = self.admet_task_config[self.task]
            benchmark = self.group.get(self.task)
            name = benchmark['name']
            train, test = benchmark['train_val'], benchmark['test']
            with open(os.devnull, 'w') as fnull, redirect_stdout(fnull), redirect_stderr(fnull): # suppress output
                test['Embedding'] = self.model(list(test['Drug']))
            test_loader = DataLoader(AdmetDataset(test), batch_size=128, shuffle=False)
            
            best_models = []
            # LOOP2: ensemble on folds
            for fold_i, seed2 in enumerate(range(self.cfg.job.max_seed, self.cfg.job.max_seed+self.ENSEMBLE_SIZE)):
                seed = cantor_pairing(seed1, seed2)
                with open(os.devnull, 'w') as fnull, redirect_stdout(fnull), redirect_stderr(fnull): # suppress output
                    mols_train, mols_valid = self.group.get_train_valid_split(benchmark = name, split_type = 'default', seed = seed)
                    mols_train['Embedding'] = self.model(list(mols_train['Drug']))
                    mols_valid['Embedding'] = self.model(list(mols_valid['Drug']))
                val_loader   = DataLoader(AdmetDataset(mols_valid), batch_size=128, shuffle=False)
                train_loader = DataLoader(AdmetDataset(mols_train), batch_size=32, shuffle=True)
                hparams = MINIMOL_MODEL_CONFIG[self.task]
                # Load task head model
                model, optimiser, lr_scheduler, loss_fn = model_factory(**hparams, task=task_type)
                best_epoch = {"model": None, "result": None}
                
                # LOOP3: training loop
                for epoch in range(self.EPOCHS):
                    model = train_one_epoch(model, train_loader, optimiser, lr_scheduler, loss_fn, epoch, task_type)
                    val_loss = evaluate(model, val_loader, loss_fn, task=task_type)
                    if best_epoch['model'] is None:
                        best_epoch['model'] = deepcopy(model)
                        best_epoch['result'] = deepcopy(val_loss)
                    else:
                        best_epoch['model'] = best_epoch['model'] if best_epoch['result'] <= val_loss else deepcopy(model)
                        best_epoch['result'] = best_epoch['result'] if best_epoch['result'] <= val_loss else deepcopy(val_loss)
                pbar.set_description(
                    f"Rep {rep_i + 1} / {self.REPETITIONS} | "
                    f"Fold {fold_i + 1} / {self.ENSEMBLE_SIZE} | "
                    f"Epoch {epoch + 1} / {self.EPOCHS} | "
                    f"Loss {val_loss:.3f}"
                )
                pbar.update(1)
                best_models.append(deepcopy(best_epoch['model']))

            y_pred_test = evaluate_ensemble(best_models, test_loader, task_type)
            predictions[name] = y_pred_test
            single_result = self.group.evaluate(predictions)[self.task]
            single_result[f"{metric_name}/{seed}"] = single_result.pop(metric_name)
            results.update(single_result)
            predictions_list.append(predictions)
            
            # Save model for each seed
            self.save(best_epoch['model'], seed)
        
        averaged_results = self.group.evaluate_many(predictions_list)[self.task]
        results.update({
            f"{metric_name}/mean": averaged_results[0],
            f"{metric_name}/std": averaged_results[1],
        })
        
        return results
    
    def save(self, model: Any, seed: int):
        save_dir = f"{hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}/{self.cfg.model.name}"
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), f"{save_dir}/{seed}.pt")
        logging.info(f"Model saved to {save_dir}/{seed}.pt")
    
    
class TaskHead(nn.Module):
    def __init__(self, hidden_dim=512, input_dim=512, dropout=0.1, depth=3, combine=True):
        super(TaskHead, self).__init__()
        self.dense1 = nn.Linear(input_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        self.dense3 = nn.Linear(hidden_dim, hidden_dim)
        self.final_dense = nn.Linear(input_dim + hidden_dim, 1) if combine else nn.Linear(hidden_dim, 1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.combine = combine
        self.depth = depth

    def forward(self, x):
        original_x = x

        x = self.dense1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.dense2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        if self.depth == 4:
            x = self.dense3(x)
            x = self.bn3(x)
            x = F.relu(x)
            x = self.dropout(x)

        x = torch.cat((x, original_x), dim=1) if self.combine else x
        x = self.final_dense(x)
        
        return x


def model_factory(hidden_dim, depth, combine, task, lr, epochs=25, warmup=5, weight_decay=0.0001):
    model = TaskHead(hidden_dim=hidden_dim, depth=depth, combine=combine)
    optimiser = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCELoss() if task == 'classification' else nn.MSELoss()        

    def lr_fn(epoch):
        if epoch < warmup: return epoch / warmup
        else: return (1 + math.cos(math.pi * (epoch - warmup) / (epochs - warmup))) / 2

    lr_scheduler = LambdaLR(optimiser, lr_lambda=lr_fn)
    return model, optimiser, lr_scheduler, loss_fn


def cantor_pairing(a, b):
    """
    We have two loops one with repetitions and one with folds;
    To ensure that each innermost execution is seeded with a unique seed,
    we use Cantor Pairing function to combine two seeds into a unique number.
    """
    return (a + b) * (a + b + 1) // 2 + b


def evaluate(predictor, dataloader, loss_fn, task):
    predictor.eval()
    total_loss = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            logits = predictor(inputs).squeeze()
            loss = loss_fn(torch.sigmoid(logits), targets) if task == 'classification' else loss_fn(logits, targets)
            total_loss += loss.item()

    loss = total_loss / len(dataloader)
    
    return loss


def evaluate_ensemble(predictors, dataloader, task):
    predictions = []
    with torch.no_grad():
        
        for inputs, _ in dataloader:
            ensemble_logits = [predictor(inputs).squeeze() for predictor in predictors]
            averaged_logits = torch.mean(torch.stack(ensemble_logits), dim=0)
            if task == 'classification':
                predictions += torch.sigmoid(averaged_logits)
            else:
                predictions += averaged_logits

    arr = np.array([t.item() for t in predictions])
    return arr


def train_one_epoch(predictor, train_loader, optimiser, lr_scheduler, loss_fn, epoch, task_type):
    predictor.train()        
    train_loss = 0
    
    lr_scheduler.step(epoch)
    
    for inputs, targets in train_loader:
        optimiser.zero_grad()
        logits = predictor(inputs).squeeze()
        loss = loss_fn(torch.sigmoid(logits), targets) if task_type == 'classification' else loss_fn(logits, targets)
        loss.backward()
        optimiser.step()
        train_loss += loss.item()

    return predictor


class AdmetDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples['Embedding'].tolist()
        self.targets = [float(target) for target in samples['Y'].tolist()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = torch.tensor(self.samples[idx])
        target = torch.tensor(self.targets[idx])
        return sample, target





from contextlib import redirect_stdout, redirect_stderr
from torch_geometric.nn import global_max_pool
from torch_geometric.data import Batch
from graphium.finetuning.fingerprinting import Fingerprinter
from graphium.config._loader import (
    load_accelerator,
    load_predictor,
    load_metrics,
    load_architecture,
    load_datamodule,
)


class MinimolEncoder: 
    def __init__(self, cfg: DictConfig, batch_size: int = 100):
        self.batch_size = batch_size
        self.cfg = cfg
        state_dict_path = str(resources.files(self.cfg.model.resource_file.file_path) / self.cfg.model.resource_file.state_dict_file_name)
        base_shape_path = str(resources.files(self.cfg.model.resource_file.file_path) / self.cfg.model.resource_file.base_shape_file_name)
        
        # Load encoder config
        cfg_model = OmegaConf.to_container(self.cfg.model.encoder, resolve=True)
        cfg_model['accelerator']['type'] = 'gpu' if torch.cuda.is_available() else 'cpu'
        self.cfg_model, accelerator_type = load_accelerator(cfg_model)
        self.cfg_model['architecture']['mup_base_path'] = base_shape_path
        self.datamodule = load_datamodule(self.cfg_model, accelerator_type)
        model_class, model_kwargs = load_architecture(self.cfg_model, in_dims=self.datamodule.in_dims)
        metrics = load_metrics(self.cfg_model)
        predictor = load_predictor(
            config=self.cfg_model,
            model_class=model_class,
            model_kwargs=model_kwargs,
            metrics=metrics,
            task_levels=self.datamodule.get_task_levels(),
            accelerator_type=accelerator_type,
            featurization=self.datamodule.featurization,
            task_norms=self.datamodule.task_norms,
            replicas=1,
            gradient_acc=1,
            global_bs=self.datamodule.batch_size_training,
        )
        self.set_training_mode_false(predictor)
        predictor.load_state_dict(torch.load(state_dict_path), strict=False)
        self.predictor = Fingerprinter(predictor, 'gnn:15')
        self.predictor.setup()

    def set_training_mode_false(self, module):
        if isinstance(module, torch.nn.Module):
            module.training = False
            for submodule in module.children():
                self.set_training_mode_false(submodule)
        elif isinstance(module, list):
            for value in module:
                self.set_training_mode_false(value)
        elif isinstance(module, dict):
            for _, value in module.items():
                self.set_training_mode_false(value)

    def load_config(self, config_name):
        hydra.initialize('ckpts/minimol_v1/', version_base=None)
        cfg = hydra.compose(config_name=config_name)
        return cfg

    def __call__(self, smiles: Union[str,list]) -> torch.Tensor:
        smiles = [smiles] if not isinstance(smiles, list) else smiles
        
        batch_size = min(self.batch_size, len(smiles))

        results = []
        for i in tqdm(range(0, len(smiles), batch_size)):
            with open(os.devnull, 'w') as fnull, redirect_stdout(fnull), redirect_stderr(fnull): # suppress output
                input_features, _ = self.datamodule._featurize_molecules(smiles[i:(i + batch_size)])
                input_features = self.to_fp32(input_features)

            batch = Batch.from_data_list(input_features)
            batch = {"features": batch, "batch_indices": batch.batch}
            node_features = self.predictor.get_fingerprints_for_batch(batch)
            fingerprint_graph = global_max_pool(node_features, batch['batch_indices'])
            num_molecules = min(batch_size, fingerprint_graph.shape[0])
            results += [fingerprint_graph[i] for i in range(num_molecules)]

        return results
    
    def to_fp32(self, input_features: list) -> list:
        failures = 0
        for input_feature in tqdm(input_features, desc="Casting to FP32"):
            try:
                if not isinstance(input_feature, str):
                    for k, v in input_feature.items():
                        if isinstance(v, torch.Tensor):
                            if v.dtype == torch.half:
                                input_feature[k] = v.float()
                            elif v.dtype == torch.int32:
                                input_feature[k] = v.long()
                else:
                    failures += 1
            except Exception as e:
                print(f"{input_feature = }")
                raise e

        if failures != 0:
            print(f"{failures = }")
        return input_features