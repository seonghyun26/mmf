import numpy as np
import pandas as pd
import logging
import os
import pickle
import swifter
from typing import List
from omegaconf import OmegaConf

from molfeat.trans.pretrained import PretrainedDGLTransformer
from molfeat.trans.pretrained import PretrainedHFTransformer

from rdkit import Chem, RDLogger
from rdkit.Chem import DataStructs
from rdkit.Chem.rdMolDescriptors import GetHashedMorganFingerprint
from rdkit.Avalon.pyAvalonTools import GetAvalonCountFP
from rdkit.Chem.rdReducedGraphs import GetErGFingerprint
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

PRETRAINED_MODELS = [
    "PretrainedDGLTransformer"
]

PRETRAINED_DGL_MODELS = [
    'gin_supervised_contextpred',
    'gin_supervised_infomax',
    'gin_supervised_edgepred',
    'gin_supervised_masking',
    'jtvae_zinc_no_kl',
] 

RDKIT_CHOSEN_DESCRIPTORS = [
    'BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 
    'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v', 
    'EState_VSA1', 'EState_VSA10', 'EState_VSA11', 'EState_VSA2', 'EState_VSA3', 
    'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 
    'EState_VSA9', 'ExactMolWt', 'FpDensityMorgan1', 'FpDensityMorgan2', 
    'FpDensityMorgan3', 'FractionCSP3', 'HallKierAlpha', 'HeavyAtomCount', 
    'HeavyAtomMolWt', 
    # 'Ipc',
    'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA', 
    'MaxAbsEStateIndex',
    # 'MaxAbsPartialCharge',
    'MaxEStateIndex',
    # 'MaxPartialCharge', 
    'MinAbsEStateIndex',
    # 'MinAbsPartialCharge',
    'MinEStateIndex',
    # 'MinPartialCharge', 
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
    'fr_unbrch_alkane', 'fr_urea', 'qed'
]

class FingerprintManager:
    def __init__(self, cfg: OmegaConf, taskname: str, split: str, data:pd.Series = None):
        RDLogger.DisableLog('rdApp.*')
        self.cfg = cfg
        self.taskname = taskname
        self.split = split
        self.data = data
        
        self._fingerprints = self.smiles2fingerprint(self.data)


    @property
    def fingerprints(self):
        return self._fingerprints

    def _check_cache(self, method: str):
        if (
            "cachedir" in self.cfg.keys()
            and "cachefile" in self.cfg.keys()
            and os.path.exists(os.path.join(self.cfg.cachedir, method, self.taskname, f"{self.split}.pkl"))
        ):
            return True
        else:
            return False
    
    def _save_cache(self, method: str, data):
        cache_path = os.path.join(self.cfg.cachedir, method, self.taskname, f"{self.split}.pkl")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)
    
    def count_to_array(self, fingerprint):
        array = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fingerprint, array)

        return array

    def get_morgan_fingerprints(self, molecules, nBits=1024, radius=2):
        if self._check_cache("morgan"):
            cache_path = os.path.join(self.cfg.cachedir, "morgan", self.taskname, f"{self.split}.pkl")
            with open(cache_path, "rb") as f:
                fingerprints_morgan = pickle.load(f)
        else:
            fingerprints_morgan = molecules.swifter.progress_bar(desc='Morgan fingerprints (compute)').apply(lambda x: GetHashedMorganFingerprint(x, nBits=nBits, radius=radius))
            fingerprints_morgan = fingerprints_morgan.swifter.progress_bar(desc='Morgan fingerprints (type conversion)').apply(self.count_to_array)
            fingerprints_morgan = np.stack(fingerprints_morgan.values)
            self._save_cache("morgan", fingerprints_morgan)
        
        return fingerprints_morgan
    
    def get_avalon_fingerprints(self, molecules, nBits=1024):
        if self._check_cache("avalon"):
            cache_path = os.path.join(self.cfg.cachedir, "avalon", self.taskname, f"{self.split}.pkl")
            with open(cache_path, "rb") as f:
                fingerprints_avalon = pickle.load(f)
        else:
            fingerprints_avalon = molecules.swifter.progress_bar(desc='Avalon fingerprints (compute)').apply(lambda x: GetAvalonCountFP(x, nBits=nBits))
            fingerprints_avalon = fingerprints_avalon.swifter.progress_bar(desc='Avalon fingerprints (type conversion)').apply(self.count_to_array)
            fingerprints_avalon = np.stack(fingerprints_avalon.values)
            self._save_cache("avalon", fingerprints_avalon)
        
        return fingerprints_avalon

    def get_erg_fingerprints(self, molecules):
        if self._check_cache("erg"):
            cache_path = os.path.join(self.cfg.cachedir, "erg", self.taskname, f"{self.split}.pkl")
            with open(cache_path, "rb") as f:
                fingerprints_erg = pickle.load(f)
        else:
            fingerprints_erg = molecules.swifter.progress_bar(desc='ERG fingerprints (compute)').apply(GetErGFingerprint)
            fingerprints_erg = np.stack(fingerprints_erg.values)
            self._save_cache("erg", fingerprints_erg)
        
        return fingerprints_erg

    def get_rdkit_features(self, molecules):
        # from https://www.blopig.com/blog/2022/06/how-to-turn-a-molecule-into-a-vector-of-physicochemical-descriptors-using-rdkit/
        if self._check_cache("rdkit"):
            cache_path = os.path.join(self.cfg.cachedir, "rdkit", self.taskname, f"{self.split}.pkl")
            with open(cache_path, "rb") as f:
                fingerprints_rdkit = pickle.load(f)
        else:
            calculator = MolecularDescriptorCalculator(RDKIT_CHOSEN_DESCRIPTORS)
            fingerprints_rdkit = molecules.swifter.progress_bar(desc='RDKit fingerprints (compute)').apply(lambda x: np.array(calculator.CalcDescriptors(x)))
            fingerprints_rdkit = np.vstack(fingerprints_rdkit.values)
            self._save_cache("rdkit", fingerprints_rdkit)
        
        return fingerprints_rdkit
    
    def get_pretrained_fingerprints(self, molecules):
        if self._check_cache(self.cfg.pretrained.name):
            cache_path = os.path.join(self.cfg.cachedir, self.cfg.pretrained.name, self.taskname, f"{self.split}.pkl")
            with open(cache_path, "rb") as f:
                fingerprints_pretrained = pickle.load(f)
        else:
            if self.cfg.pretrained.name == "PretrainedDGLTransformer":
                if self.cfg.pretrained.kind not in PRETRAINED_DGL_MODELS:
                    raise ValueError(f"Unsupported pretrained model: {self.cfg.pretrained.kind}. Available models in DGL: {PRETRAINED_DGL_MODELS}")
                transformer = PretrainedDGLTransformer(kind=self.cfg.pretrained.kind, dtype=float)
                fingerprints_pretrained = transformer(molecules)
            else:
                raise ValueError(f"Unsupported pretrained model: {self.cfg.pretrained.name}. Available model: {PRETRAINED_MODELS}")
            self._save_cache(self.cfg.pretrained.name, fingerprints_pretrained)
        
        return fingerprints_pretrained
    
    def smiles2fingerprint(self, smiles) -> np.ndarray:
        molecules = smiles.swifter.progress_bar(desc='Parsing SMILES').apply(lambda x: Chem.MolFromSmiles(x))

        fingerprints = []
        for type in self.cfg.type:
            if type == "morgan":
                fingerprints.append(self.get_morgan_fingerprints(molecules, **self.cfg.morgan))
            elif type == "avalon":
                fingerprints.append(self.get_avalon_fingerprints(molecules, **self.cfg.avalon))
            elif type == "erg":
                fingerprints.append(self.get_erg_fingerprints(molecules))
            elif type == "rdkit":
                fingerprints.append(self.get_rdkit_features(molecules))
            elif type == "pretrained":
                fingerprints.append(self.get_pretrained_fingerprints(molecules))
            else:
                raise ValueError(f"Unsupported fingerprint type: {type}")

        fingerprints = np.concatenate(fingerprints, axis=1)
        self._fingerprints = fingerprints
        
        return self._fingerprints
    
    
class FingerprintManagerHF(FingerprintManager):
    def __init__(self, cfg: OmegaConf, taskname: str, model_name: str, split: str, data:pd.Series):
        RDLogger.DisableLog('rdApp.*')
        self.cfg = cfg
        self.taskname = taskname
        self.name = model_name
        self.split = split
        self.data = data
        
        self.cache_path = os.path.join(self.cfg.cachedir, self.name, f"{self.taskname}_{self.split}.pkl")
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        self._initalize_fingerprints()

    @property
    def fingerprints(self):
        return self._fingerprints
    
    def smiles2fingerprint(self, smiles) -> np.ndarray:
        molecules = smiles.swifter.apply(lambda x: Chem.MolFromSmiles(x))

        fingerprints = []
        fingerprints.append(self.get_morgan_fingerprints(molecules, **self.cfg.morgan))
        fingerprints.append(self.get_avalon_fingerprints(molecules, **self.cfg.avalon))
        fingerprints.append(self.get_erg_fingerprints(molecules))
        fingerprints.append(self.get_rdkit_features(molecules))

        ml_cfg = getattr(self.cfg, "ml", None)
        if ml_cfg and bool(ml_cfg.get("enabled", False)):
            # Use raw SMILES (not RDKit mols) for the language model backend
            ml_featurizer = MLFeaturizer(ml_cfg)
            ml_vecs = ml_featurizer(smiles)           # [N, D]
            fingerprints.append(ml_vecs)

        return np.concatenate(fingerprints, axis=1)
    
    
class MLFeaturizer:
    """
    Wrap ML-based molecular embeddings (Transformer or DGL via molfeat).
    """
    def __init__(self, ml_cfg: OmegaConf):
        self.ml_cfg = ml_cfg
        self.backend = ml_cfg.get("backend", "hf")
        self.model_name = ml_cfg.get("model_name", "ChemBERTa-77M-MLM")
        self.pooling = ml_cfg.get("pooling", "mean")
        self.batch_size = int(ml_cfg.get("batch_size", 64))

        self._hf = None
        self._ready = False
        self._init_models()

    def _init_models(self):
        if self.backend == "hf":
            if PretrainedHFTransformer is None:
                raise ImportError("molfeat[transformers] is required for HF backend.")
            self._hf = PretrainedHFTransformer(
                kind=self.model_name,
                pooling=self.pooling,
                as_numpy=True
            )
            self._ready = True
        elif self.backend == "dgl":
            raise NotImplementedError("Set a valid DGL pretrained model in molfeat store.")
        else:
            raise ValueError(f"Unknown ML featurizer backend: {self.backend}")

    def __call__(self, smiles_series: pd.Series) -> np.ndarray:
        assert self._ready, "MLFeaturizer is not initialized."
        embeds = self._hf.transform(list(smiles_series.values), batch_size=self.batch_size)
        embeds = np.asarray(embeds)
        if embeds.ndim == 1:
            embeds = embeds.reshape(-1, 1)
        return embeds   