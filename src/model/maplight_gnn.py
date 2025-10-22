import hydra
import catboost as cb
import os

from tqdm import tqdm
from typing import Any
from omegaconf import DictConfig, OmegaConf

from tdc.benchmark_group import admet_group
from tdc.metadata import admet_metrics


from .base import ModelWrapper

group = admet_group(path = './data/')


class CatboostGNN(ModelWrapper):
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
            
            benchmark = group.get(self.task)
            predictions = {}
            name = benchmark['name']
            train, test = benchmark['train_val'], benchmark['test']
            X_train = get_fingerprints(train['Drug'])
            X_test = get_fingerprints(test['Drug'])
            
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
            single_result = group.evaluate(predictions)[self.task]
            single_result[f"{metric_name}/{seed}"] = single_result.pop(metric_name)
            results.update(single_result)
            predictions_list.append(predictions)
            
            # Save model for each seed
            self.save(model, seed)
        
        averaged_results = group.evaluate_many(predictions_list)[self.task]
        results.update({
            f"{metric_name}/mean": averaged_results[0],
            f"{metric_name}/std": averaged_results[1],
        })
        
        return results

    def save(self, model: Any, seed: int):
        save_dir = f"{hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}/model"
        os.makedirs(save_dir, exist_ok=True)
        format = self.cfg.model.format
        model.save_model(
            fname=f"{save_dir}/{seed}.{format}",
            format=format,
            export_parameters=None,
            pool=None
        )



import numpy as np

from sklearn import preprocessing

from rdkit import Chem
from rdkit import RDLogger

from rdkit.Chem import DataStructs
from rdkit.Chem.rdMolDescriptors import GetHashedMorganFingerprint
from rdkit.Avalon.pyAvalonTools import GetAvalonCountFP
from rdkit.Chem import rdReducedGraphs
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

from molfeat.trans.pretrained import PretrainedDGLTransformer


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
def get_chosen_descriptors():
    chosen_descriptors = ['BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 
        'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v', 
        'EState_VSA1', 'EState_VSA10', 'EState_VSA11', 'EState_VSA2', 'EState_VSA3', 
        'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 
        'EState_VSA9', 'ExactMolWt', 'FpDensityMorgan1', 'FpDensityMorgan2', 
        'FpDensityMorgan3', 'FractionCSP3', 'HallKierAlpha', 'HeavyAtomCount', 
        'HeavyAtomMolWt', 'Ipc', 'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA', 
        'MaxAbsEStateIndex', 'MaxAbsPartialCharge', 'MaxEStateIndex', 'MaxPartialCharge', 
        'MinAbsEStateIndex', 'MinAbsPartialCharge', 'MinEStateIndex', 'MinPartialCharge', 
        'MolLogP', 'MolMR', 'MolWt', 'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles', 
        'NumAliphaticHeterocycles', 'NumAliphaticRings', 'NumAromaticCarbocycles', 
        'NumAromaticHeterocycles', 'NumAromaticRings', 'NumHAcceptors', 'NumHDonors', 
        'NumHeteroatoms', 'NumRadicalElectrons', 'NumRotatableBonds', 
        'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 'NumSaturatedRings', 
        'NumValenceElectrons', 'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 
        'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 
        'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'RingCount', 'SMR_VSA1', 
        'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 
        'SMR_VSA8', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA11', 'SlogP_VSA12', 
        'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 
        'SlogP_VSA8', 'SlogP_VSA9', 'TPSA', 'VSA_EState1', 'VSA_EState10', 'VSA_EState2', 
        'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 
        'VSA_EState8', 'VSA_EState9', 'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN', 
        'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O', 
        'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0', 'fr_NH1', 'fr_NH2', 
        'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_Nhpyrrole', 'fr_SH', 
        'fr_aldehyde', 'fr_alkyl_carbamate', 'fr_alkyl_halide', 'fr_allylic_oxid', 
        'fr_amide', 'fr_amidine', 'fr_aniline', 'fr_aryl_methyl', 'fr_azide', 'fr_azo', 
        'fr_barbitur', 'fr_benzene', 'fr_benzodiazepine', 'fr_bicyclic', 'fr_diazo', 
        'fr_dihydropyridine', 'fr_epoxide', 'fr_ester', 'fr_ether', 'fr_furan', 'fr_guanido', 
        'fr_halogen', 'fr_hdrzine', 'fr_hdrzone', 'fr_imidazole', 'fr_imide', 'fr_isocyan', 
        'fr_isothiocyan', 'fr_ketone', 'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 
        'fr_methoxy', 'fr_morpholine', 'fr_nitrile', 'fr_nitro', 'fr_nitro_arom', 
        'fr_nitro_arom_nonortho', 'fr_nitroso', 'fr_oxazole', 'fr_oxime', 
        'fr_para_hydroxylation', 'fr_phenol', 'fr_phenol_noOrthoHbond', 'fr_phos_acid', 
        'fr_phos_ester', 'fr_piperdine', 'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd', 
        'fr_pyridine', 'fr_quatN', 'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 
        'fr_term_acetylene', 'fr_tetrazole', 'fr_thiazole', 'fr_thiocyan', 'fr_thiophene', 
        'fr_unbrch_alkane', 'fr_urea', 'qed']
    
    return chosen_descriptors


def get_rdkit_features(molecules):
    calculator = MolecularDescriptorCalculator(get_chosen_descriptors())

    X_rdkit = molecules.apply(
        lambda x: np.array(calculator.CalcDescriptors(x)))

    X_rdkit = np.vstack(X_rdkit.values)

    return X_rdkit


def get_gin_supervised_masking(molecules):
    transformer = PretrainedDGLTransformer(kind='gin_supervised_masking', dtype=float)

    return transformer(molecules)


def get_fingerprints(smiles):
    RDLogger.DisableLog('rdApp.*')
    molecules = smiles.apply(Chem.MolFromSmiles)
    
    fingerprints = []

    fingerprints.append(get_morgan_fingerprints(molecules))
    fingerprints.append(get_avalon_fingerprints(molecules))
    fingerprints.append(get_erg_fingerprints(molecules))
    fingerprints.append(get_rdkit_features(molecules))
    fingerprints.append(get_gin_supervised_masking(molecules))

    return np.concatenate(fingerprints, axis=1)

