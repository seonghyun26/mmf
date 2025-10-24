import hydra
import os

import numpy as np
import pandas as pd

from typing import Dict, List, Tuple, Literal

from scipy.stats import spearmanr, rankdata
from sklearn.metrics import mean_absolute_error, roc_auc_score, average_precision_score

from omegaconf import DictConfig

from tdc.benchmark_group import admet_group
from tdc.metadata import admet_metrics

from .base import ModelWrapper
from ..util.constant import admet_task_config


MetricType = Literal['spearman', 'mae', 'auroc', 'auprc']
TaskType = Literal['regression', 'binary']


class CFA(ModelWrapper):
    def __init__(self, cfg: DictConfig, task: str):
        super().__init__(cfg, task)
        self.group = admet_group(path = './data/')
        self.admet_task_config = admet_task_config

    def train(self):
        metric_name = admet_metrics.get(self.task, )

        # Determine task type and choose fusion metric
        task_type, _ = self.admet_task_config[self.task]
        if task_type == 'regression':
            metric: MetricType = 'spearman'
        elif task_type == 'binary':
            metric = 'auroc'
        else:
            raise ValueError(f'Invalid task type: {task_type}')

        # Prepare dataset information
        benchmark = self.group.get(self.task)
        name = benchmark['name']
        train_val, test = benchmark['train_val'], benchmark['test']

        # Targets
        y_test = np.array(test.Y).reshape(-1)
        y_valid: Dict[int, pd.Series] = {}
        # Follow CFA4DD convention: seeds 1..max_seed
        for seed in range(1, self.cfg.job.max_seed + 1):
            tv = self.group.get_train_valid_split(benchmark=name, split_type='default', seed=seed)
            y_valid[seed] = tv[1].Y

        # Load per-model predictions from ./data using CFA4DD naming
        # Expect files like: f"{name}_predictions_val_<model>.csv" and f"{name}_predictions_test_<model>.csv"
        model_names: List[str] = list(self.cfg.model.models)
        val_dfs_list: List[pd.DataFrame] = []
        test_dfs_list: List[pd.DataFrame] = []
        for m in model_names:
            val_path = os.path.join('./data', f'{name}_predictions_val_{m}.csv')
            test_path = os.path.join('./data', f'{name}_predictions_test_{m}.csv')
            if not os.path.exists(val_path) or not os.path.exists(test_path):
                raise FileNotFoundError(f'Missing prediction files for model {m}: {val_path} or {test_path}')
            val_dfs_list.append(pd.read_csv(val_path))
            test_dfs_list.append(pd.read_csv(test_path))

        summary_df, metrics, fused_predictions = run_cfa(
            model_names=model_names,
            val_dfs_list=val_dfs_list,
            test_dfs_list=test_dfs_list,
            y_test=y_test,
            y_valid=y_valid,
            task_type=task_type,  # type: ignore[arg-type]
            metric=metric,
        )

        # Map metrics to standardized keys
        results: Dict[str, float] = {}
        if task_type == 'regression':
            # Spearman correlation
            results[f'{metric_name}/rank'] = metrics['rank/spearman']
            results[f'{metric_name}/performance'] = metrics['performance/spearman']
            results[f'{metric_name}/average'] = metrics['average/spearman']
        else:
            # Classification AUROC by default
            results[f'{metric_name}/rank'] = metrics['rank/auroc']
            results[f'{metric_name}/performance'] = metrics['performance/auroc']
            results[f'{metric_name}/average'] = metrics['average/auroc']

        # Persist fused predictions for downstream use
        self.save(fused_predictions)

        # Also persist the summary table
        out_dir = f"{hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}/{self.task}"
        os.makedirs(out_dir, exist_ok=True)
        summary_df.to_csv(os.path.join(out_dir, 'cfa_summary.csv'), index=False)

        return results

    def save(self, fused_predictions: Dict[str, np.ndarray]):
        save_dir = f"{hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}/{self.cfg.model.name}"
        os.makedirs(save_dir, exist_ok=True)
        # Save as CSVs for readability
        for key, arr in fused_predictions.items():
            df = pd.DataFrame({key: np.asarray(arr).reshape(-1)})
            df.to_csv(os.path.join(save_dir, f'cfa_{key}.csv'), index=False)

def _validate_inputs(
    model_names: List[str],
    val_dfs_list: List[pd.DataFrame],
    test_dfs_list: List[pd.DataFrame],
    y_valid: Dict[int, pd.Series],
    y_test: np.ndarray,
):
    if len(model_names) == 0:
        raise ValueError('model_names must not be empty')
    if len(val_dfs_list) != len(model_names) or len(test_dfs_list) != len(model_names):
        raise ValueError('val_dfs_list and test_dfs_list lengths must match model_names')
    if len(y_test.shape) != 1:
        raise ValueError('y_test must be a 1D array')
    # Basic alignment checks: each validation df must contain the same number of rows as its corresponding seed set
    # We expect validation predictions to be provided for multiple seeds; concatenate along rows per model.


def _concat_seed_frames(frames: List[pd.DataFrame]) -> pd.Series:
    # Accept a single-column DataFrame or a DataFrame with a column named 'prediction'
    if len(frames) == 0:
        raise ValueError('Empty frames for seeds')
    ser_list = []
    for df in frames:
        if isinstance(df, pd.Series):
            ser = df
        elif df.shape[1] == 1:
            ser = df.iloc[:, 0]
        elif 'prediction' in df.columns:
            ser = df['prediction']
        else:
            # Fallback: use the last column
            ser = df.iloc[:, -1]
        ser_list.append(pd.to_numeric(ser, errors='coerce'))
    return pd.concat(ser_list, axis=0).reset_index(drop=True)


def _combine_val_frames_by_seed(val_df: pd.DataFrame, seeds: List[int]) -> List[pd.DataFrame]:
    # Flexible loader: if the DataFrame contains a 'seed' column, split by it; otherwise assume the whole df is one seed
    if 'seed' in val_df.columns:
        return [val_df[val_df['seed'] == s].drop(columns=[c for c in ['seed'] if c in val_df.columns]) for s in seeds]
    return [val_df]


def _compute_validation_scores(
    task_type: TaskType,
    metric: MetricType,
    y_valid: Dict[int, pd.Series],
    val_predictions_per_model: List[List[pd.Series]],
) -> np.ndarray:
    # Compute an average performance per model across seeds
    num_models = len(val_predictions_per_model)
    seeds = sorted(list(y_valid.keys()))
    scores = np.zeros(num_models, dtype=float)
    for m_idx in range(num_models):
        seed_scores = []
        for s_idx, seed in enumerate(seeds):
            yv = np.asarray(y_valid[seed]).reshape(-1)
            pv = np.asarray(val_predictions_per_model[m_idx][s_idx]).reshape(-1)
            if len(yv) != len(pv):
                raise ValueError('Mismatch between validation targets and predictions for a seed')
            if task_type == 'regression':
                if metric == 'spearman':
                    rho = spearmanr(yv, pv).correlation
                    seed_scores.append(0.0 if rho is None or np.isnan(rho) else float(rho))
                elif metric == 'mae':
                    mae = mean_absolute_error(yv, pv)
                    # Convert to a higher-is-better score
                    seed_scores.append(float(-mae))
                else:
                    raise ValueError('Unsupported regression metric')
            elif task_type == 'binary':
                if metric == 'auroc':
                    seed_scores.append(float(roc_auc_score(yv, pv)))
                elif metric == 'auprc':
                    seed_scores.append(float(average_precision_score(yv, pv)))
                else:
                    raise ValueError('Unsupported binary metric')
            else:
                raise ValueError('Unsupported task type')
        scores[m_idx] = float(np.mean(seed_scores))
    return scores


def _normalize_weights(scores: np.ndarray) -> np.ndarray:
    # Shift to positive and normalize
    finite_scores = np.where(np.isfinite(scores), scores, 0.0)
    min_s = np.min(finite_scores)
    shifted = finite_scores - min_s
    denom = np.sum(shifted)
    if denom <= 1e-12:
        # Fallback to uniform weights
        return np.ones_like(shifted) / float(len(shifted))
    return shifted / denom


def _rank_fusion(test_predictions_per_model: List[pd.Series]) -> np.ndarray:
    # Average of normalized ranks across models
    num_models = len(test_predictions_per_model)
    n = len(test_predictions_per_model[0])
    ranks = np.zeros((num_models, n), dtype=float)
    for m_idx, preds in enumerate(test_predictions_per_model):
        preds_arr = np.asarray(preds).reshape(-1)
        ranks[m_idx] = rankdata(preds_arr, method='average') / float(n)  # normalized to (0,1]
    return np.mean(ranks, axis=0)


def _performance_weighted_fusion(
    weights: np.ndarray,
    test_predictions_per_model: List[pd.Series]
) -> np.ndarray:
    preds_matrix = np.vstack([np.asarray(p).reshape(-1) for p in test_predictions_per_model])
    return np.average(preds_matrix, axis=0, weights=weights)


def run_cfa(
    model_names: List[str],
    val_dfs_list: List[pd.DataFrame],
    test_dfs_list: List[pd.DataFrame],
    y_test: np.ndarray,
    y_valid: Dict[int, pd.Series],
    task_type: TaskType,
    metric: MetricType,
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, np.ndarray]]:
    """
    Perform Combinatorial Fusion Analysis (CFA) inspired by CFA4DD/cfanalysis.

    Returns:
      - summary_df: per-model and per-fusion scores
      - metrics: evaluation of fused predictions on y_test under the chosen metric
      - fused_predictions: dict with keys 'rank', 'performance', 'average'

    Reference: https://github.com/F-LIDM/CFA4DD
    """
    _validate_inputs(model_names, val_dfs_list, test_dfs_list, y_valid, y_test)

    seeds = sorted(list(y_valid.keys()))

    # Prepare per-model, per-seed validation series and single test series
    val_predictions_per_model: List[List[pd.Series]] = []
    test_predictions_per_model: List[pd.Series] = []

    for m_idx, mname in enumerate(model_names):
        val_parts = _combine_val_frames_by_seed(val_dfs_list[m_idx], seeds)
        if len(val_parts) != len(seeds):
            # If only one combined frame, broadcast to all seeds (assume same ordering)
            val_parts = [val_parts[0] for _ in seeds]
        val_series_per_seed = []
        for part in val_parts:
            val_series_per_seed.append(_concat_seed_frames([part]))
        val_predictions_per_model.append(val_series_per_seed)

        # Test predictions
        test_series = _concat_seed_frames([test_dfs_list[m_idx]])
        test_predictions_per_model.append(test_series)

    # Compute validation-based performance per model to derive weights
    val_scores = _compute_validation_scores(task_type, metric, y_valid, val_predictions_per_model)
    weights = _normalize_weights(val_scores)

    # Compute fusions
    fused_rank = _rank_fusion(test_predictions_per_model)
    fused_perf = _performance_weighted_fusion(weights, test_predictions_per_model)
    fused_avg = 0.5 * (fused_rank + fused_perf)

    # Build summary table
    summary_rows = []
    for m_idx, mname in enumerate(model_names):
        summary_rows.append({
            'key': f'{mname}:modelname',
            'score': float(val_scores[m_idx])
        })
    # Diversity strength (1 - average pairwise Spearman on validation concatenated)
    concat_valid_per_model = [
        pd.concat(val_predictions_per_model[m_idx], axis=0).reset_index(drop=True) for m_idx in range(len(model_names))
    ]
    pairwise = []
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            rho = spearmanr(concat_valid_per_model[i], concat_valid_per_model[j]).correlation
            pairwise.append(0.0 if rho is None or np.isnan(rho) else float(rho))
    diversity_strength = float(1.0 - (np.mean(pairwise) if len(pairwise) else 0.0))

    summary_rows.append({'key': 'modelcombination_ds', 'score': diversity_strength})
    summary_rows.append({'key': 'modelcombination_r', 'score': float(np.mean(fused_rank))})
    summary_rows.append({'key': 'modelcombination_ps', 'score': float(np.mean(fused_perf))})
    summary_rows.append({'key': 'modelcombination', 'score': float(np.mean(fused_avg))})

    summary_df = pd.DataFrame(summary_rows)

    # Evaluate fused predictions on test
    metrics: Dict[str, float] = {}
    if task_type == 'regression':
        if metric == 'spearman':
            metrics['rank/spearman'] = float(spearmanr(y_test, fused_rank).correlation)
            metrics['performance/spearman'] = float(spearmanr(y_test, fused_perf).correlation)
            metrics['average/spearman'] = float(spearmanr(y_test, fused_avg).correlation)
        elif metric == 'mae':
            metrics['rank/mae'] = float(mean_absolute_error(y_test, fused_rank))
            metrics['performance/mae'] = float(mean_absolute_error(y_test, fused_perf))
            metrics['average/mae'] = float(mean_absolute_error(y_test, fused_avg))
        else:
            raise ValueError('Unsupported regression metric')
    elif task_type == 'binary':
        if metric == 'auroc':
            metrics['rank/auroc'] = float(roc_auc_score(y_test, fused_rank))
            metrics['performance/auroc'] = float(roc_auc_score(y_test, fused_perf))
            metrics['average/auroc'] = float(roc_auc_score(y_test, fused_avg))
        elif metric == 'auprc':
            metrics['rank/auprc'] = float(average_precision_score(y_test, fused_rank))
            metrics['performance/auprc'] = float(average_precision_score(y_test, fused_perf))
            metrics['average/auprc'] = float(average_precision_score(y_test, fused_avg))
        else:
            raise ValueError('Unsupported binary metric')
    else:
        raise ValueError('Unsupported task type')

    fused_predictions = {
        'rank': fused_rank,
        'performance': fused_perf,
        'average': fused_avg,
    }

    return summary_df, metrics, fused_predictions


