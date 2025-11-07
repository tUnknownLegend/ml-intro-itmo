import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Union, List, Dict, Any, Tuple


def precision_at_k(y_true: Union[np.ndarray, List], y_pred: Union[np.ndarray, List], k: int = 10) -> float:
    """
    Calculate Precision@K for recommendation systems.
    
    Parameters:
    y_true (array-like): True relevance scores (binary or continuous)
    y_pred (array-like): Predicted scores
    k (int): Number of top items to consider
    
    Returns:
    float: Precision@K
    """
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Handle edge case where k is larger than the number of items
    k = min(k, len(y_true))
    
    # Get indices of top k predictions
    top_k_indices = np.argpartition(y_pred, -k)[-k:]
    
    # Calculate precision@k
    relevant_items = np.sum(y_true[top_k_indices] > 0)
    return float(relevant_items / k)


def recall_at_k(y_true: Union[np.ndarray, List], y_pred: Union[np.ndarray, List], k: int = 10) -> float:
    """
    Calculate Recall@K for recommendation systems.
    
    Parameters:
    y_true (array-like): True relevance scores (binary or continuous)
    y_pred (array-like): Predicted scores
    k (int): Number of top items to consider
    
    Returns:
    float: Recall@K
    """
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Handle edge case where k is larger than the number of items
    k = min(k, len(y_true))
    
    # Get indices of top k predictions
    top_k_indices = np.argpartition(y_pred, -k)[-k:]
    
    # Calculate recall@k
    relevant_items_in_top_k = np.sum(y_true[top_k_indices] > 0)
    total_relevant_items = np.sum(y_true > 0)
    
    if total_relevant_items == 0:
        return 0.0
    
    return float(relevant_items_in_top_k / total_relevant_items)


def ndcg_at_k(y_true: Union[np.ndarray, List], y_pred: Union[np.ndarray, List], k: int = 10) -> float:
    """
    Calculate NDCG@K (Normalized Discounted Cumulative Gain) for recommendation systems.
    
    Parameters:
    y_true (array-like): True relevance scores (binary or continuous)
    y_pred (array-like): Predicted scores
    k (int): Number of top items to consider
    
    Returns:
    float: NDCG@K
    """
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Handle edge case where k is larger than the number of items
    k = min(k, len(y_true))
    
    # Get indices of top k predictions
    top_k_indices = np.argpartition(y_pred, -k)[-k:]
    
    # Sort by predicted scores (descending)
    sorted_indices = top_k_indices[np.argsort(-y_pred[top_k_indices])]
    
    # Calculate DCG
    dcg = 0.0
    for i, idx in enumerate(sorted_indices[:k]):
        dcg += (2 ** y_true[idx] - 1) / np.log2(i + 2)
    
    # Calculate IDCG (ideal DCG)
    ideal_sorted_indices = np.argsort(-y_true)[:k]
    idcg = 0.0
    for i, idx in enumerate(ideal_sorted_indices):
        idcg += (2 ** y_true[idx] - 1) / np.log2(i + 2)
    
    if idcg == 0:
        return 0.0
    
    return float(dcg / idcg)


def average_precision_at_k(y_true: Union[np.ndarray, List], y_pred: Union[np.ndarray, List], k: int = 10) -> float:
    """
    Calculate Average Precision@K for recommendation systems.
    
    Parameters:
    y_true (array-like): True relevance scores (binary or continuous)
    y_pred (array-like): Predicted scores
    k (int): Number of top items to consider
    
    Returns:
    float: Average Precision@K
    """
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Handle edge case where k is larger than the number of items
    k = min(k, len(y_true))
    
    # Get indices of top k predictions
    top_k_indices = np.argpartition(y_pred, -k)[-k:]
    
    # Sort by predicted scores (descending)
    sorted_indices = top_k_indices[np.argsort(-y_pred[top_k_indices])]
    
    # Calculate average precision
    ap = 0.0
    relevant_items = 0
    
    for i, idx in enumerate(sorted_indices[:k]):
        if y_true[idx] > 0:
            relevant_items += 1
            ap += relevant_items / (i + 1)
    
    if relevant_items == 0:
        return 0.0
    
    return float(ap / relevant_items)


def mean_reciprocal_rank(y_true: Union[np.ndarray, List], y_pred: Union[np.ndarray, List]) -> float:
    """
    Calculate Mean Reciprocal Rank for recommendation systems.
    
    Parameters:
    y_true (array-like): True relevance scores (binary or continuous)
    y_pred (array-like): Predicted scores
    
    Returns:
    float: Mean Reciprocal Rank
    """
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Sort by predicted scores (descending)
    sorted_indices = np.argsort(-y_pred)
    sorted_true = y_true[sorted_indices]
    
    # Find first relevant item
    for i, relevance in enumerate(sorted_true):
        if relevance > 0:
            return float(1.0 / (i + 1))
    
    return 0.0


def bootstrap_ranking_metrics(
    y_true: Union[np.ndarray, List], 
    y_pred: Union[np.ndarray, List], 
    k_values: List[int] = [5, 10, 20],
    n_bootstrap: int = 1000, 
    confidence: float = 0.95
) -> Dict[str, Dict[str, Union[float, Tuple[float, float]]]]:
    """
    Calculate bootstrap confidence intervals for ranking metrics.
    
    Parameters:
    y_true (array-like): True relevance scores
    y_pred (array-like): Predicted scores
    k_values (list): List of K values to evaluate
    n_bootstrap (int): Number of bootstrap samples
    confidence (float): Confidence level (e.g., 0.95 for 95% confidence interval)
    
    Returns:
    dict: Dictionary containing metrics with confidence intervals
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Initialize storage for bootstrap samples
    precision_scores = {k: [] for k in k_values}
    recall_scores = {k: [] for k in k_values}
    ndcg_scores = {k: [] for k in k_values}
    mrr_scores = []
    
    n_samples = len(y_true)
    
    for _ in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(n_samples, n_samples, replace=True)
        boot_y_true = y_true[indices]
        boot_y_pred = y_pred[indices]
        
        # Calculate metrics
        for k in k_values:
            precision = precision_at_k(boot_y_true, boot_y_pred, k)
            recall = recall_at_k(boot_y_true, boot_y_pred, k)
            ndcg = ndcg_at_k(boot_y_true, boot_y_pred, k)
            
            precision_scores[k].append(precision)
            recall_scores[k].append(recall)
            ndcg_scores[k].append(ndcg)
        
        mrr = mean_reciprocal_rank(boot_y_true, boot_y_pred)
        mrr_scores.append(mrr)
    
    # Calculate confidence intervals
    alpha = 1 - confidence
    lower_percentile = (alpha/2) * 100
    upper_percentile = (1 - alpha/2) * 100
    
    results = {}
    
    # Add MRR results
    mrr_mean = np.mean(mrr_scores)
    mrr_ci = np.percentile(mrr_scores, [lower_percentile, upper_percentile])
    results['mrr'] = {
        'mean': float(mrr_mean),
        'ci_lower': float(mrr_ci[0]),
        'ci_upper': float(mrr_ci[1])
    }
    
    # Add results for each K value
    for k in k_values:
        precision_mean = np.mean(precision_scores[k])
        recall_mean = np.mean(recall_scores[k])
        ndcg_mean = np.mean(ndcg_scores[k])
        
        precision_ci = np.percentile(precision_scores[k], [lower_percentile, upper_percentile])
        recall_ci = np.percentile(recall_scores[k], [lower_percentile, upper_percentile])
        ndcg_ci = np.percentile(ndcg_scores[k], [lower_percentile, upper_percentile])
        
        results[f'precision@{k}'] = {
            'mean': float(precision_mean),
            'ci_lower': float(precision_ci[0]),
            'ci_upper': float(precision_ci[1])
        }
        
        results[f'recall@{k}'] = {
            'mean': float(recall_mean),
            'ci_lower': float(recall_ci[0]),
            'ci_upper': float(recall_ci[1])
        }
        
        results[f'ndcg@{k}'] = {
            'mean': float(ndcg_mean),
            'ci_lower': float(ndcg_ci[0]),
            'ci_upper': float(ndcg_ci[1])
        }
    
    return results


def statistical_significance_test(
    y_true_1: Union[np.ndarray, List], 
    y_pred_1: Union[np.ndarray, List],
    y_true_2: Union[np.ndarray, List], 
    y_pred_2: Union[np.ndarray, List],
    k: int = 10,
    n_bootstrap: int = 1000
) -> Dict[str, Dict[str, Union[float, bool]]]:
    """
    Perform statistical significance testing between two models using bootstrap sampling.
    
    Parameters:
    y_true_1, y_pred_1: True and predicted values for model 1
    y_true_2, y_pred_2: True and predicted values for model 2
    k (int): K value for ranking metrics
    n_bootstrap (int): Number of bootstrap samples
    
    Returns:
    dict: Dictionary containing test results
    """
    y_true_1 = np.array(y_true_1)
    y_pred_1 = np.array(y_pred_1)
    y_true_2 = np.array(y_true_2)
    y_pred_2 = np.array(y_pred_2)
    
    # Ensure same length
    min_len = min(len(y_true_1), len(y_true_2))
    y_true_1, y_pred_1 = y_true_1[:min_len], y_pred_1[:min_len]
    y_true_2, y_pred_2 = y_true_2[:min_len], y_pred_2[:min_len]
    
    # Bootstrap samples for both models
    precision_diffs = []
    recall_diffs = []
    ndcg_diffs = []
    
    for _ in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(min_len, min_len, replace=True)
        
        # Sample for model 1
        boot_y_true_1 = y_true_1[indices]
        boot_y_pred_1 = y_pred_1[indices]
        
        # Sample for model 2
        boot_y_true_2 = y_true_2[indices]
        boot_y_pred_2 = y_pred_2[indices]
        
        # Calculate metrics for both models
        p1 = precision_at_k(boot_y_true_1, boot_y_pred_1, k)
        r1 = recall_at_k(boot_y_true_1, boot_y_pred_1, k)
        n1 = ndcg_at_k(boot_y_true_1, boot_y_pred_1, k)
        
        p2 = precision_at_k(boot_y_true_2, boot_y_pred_2, k)
        r2 = recall_at_k(boot_y_true_2, boot_y_pred_2, k)
        n2 = ndcg_at_k(boot_y_true_2, boot_y_pred_2, k)
        
        # Store differences
        precision_diffs.append(p1 - p2)
        recall_diffs.append(r1 - r2)
        ndcg_diffs.append(n1 - n2)
    
    # Calculate p-values (two-tailed test)
    precision_diffs = np.array(precision_diffs)
    recall_diffs = np.array(recall_diffs)
    ndcg_diffs = np.array(ndcg_diffs)
    
    precision_p_value = 2 * min(
        np.mean(precision_diffs > 0), 
        np.mean(precision_diffs < 0)
    )
    
    recall_p_value = 2 * min(
        np.mean(recall_diffs > 0), 
        np.mean(recall_diffs < 0)
    )
    
    ndcg_p_value = 2 * min(
        np.mean(ndcg_diffs > 0), 
        np.mean(ndcg_diffs < 0)
    )
    
    # Determine significance (p < 0.05)
    precision_significant = precision_p_value < 0.05
    recall_significant = recall_p_value < 0.05
    ndcg_significant = ndcg_p_value < 0.05
    
    return {
        'precision': {
            'p_value': float(precision_p_value),
            'significant': precision_significant
        },
        'recall': {
            'p_value': float(recall_p_value),
            'significant': recall_significant
        },
        'ndcg': {
            'p_value': float(ndcg_p_value),
            'significant': ndcg_significant
        }
    }


def evaluate_ranking_metrics(
    y_true: Union[np.ndarray, List], 
    y_pred: Union[np.ndarray, List], 
    k_values: List[int] = [5, 10, 20]
) -> Dict[str, float]:
    """
    Evaluate multiple ranking metrics at different K values.
    
    Parameters:
    y_true (array-like): True relevance scores
    y_pred (array-like): Predicted scores
    k_values (list): List of K values to evaluate
    
    Returns:
    dict: Dictionary containing all metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    results = {}
    
    # Traditional regression metrics
    results['mae'] = float(mean_absolute_error(y_true, y_pred))
    results['rmse'] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    results['r2'] = float(r2_score(y_true, y_pred))
    
    # Ranking metrics for different K values
    for k in k_values:
        results[f'precision@{k}'] = precision_at_k(y_true, y_pred, k)
        results[f'recall@{k}'] = recall_at_k(y_true, y_pred, k)
        results[f'ndcg@{k}'] = ndcg_at_k(y_true, y_pred, k)
        results[f'ap@{k}'] = average_precision_at_k(y_true, y_pred, k)
    
    # MRR
    results['mrr'] = mean_reciprocal_rank(y_true, y_pred)
    
    return results


def compare_models_ranking_metrics(
    models: List[Any], 
    X_test: Union[np.ndarray, pd.DataFrame], 
    y_test: Union[np.ndarray, List], 
    model_names: List[str] = None, 
    k_values: List[int] = [5, 10, 20]
) -> pd.DataFrame:
    """
    Compare multiple models using ranking metrics.
    
    Parameters:
    models (list): List of trained models
    X_test (array-like): Test features
    y_test (array-like): Test targets
    model_names (list): List of model names (optional)
    k_values (list): List of K values to evaluate
    
    Returns:
    pd.DataFrame: DataFrame containing all metrics for all models
    """
    if model_names is None:
        model_names = [f"Model_{i}" for i in range(len(models))]
    
    results = []
    
    for model, name in zip(models, model_names):
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Evaluate metrics
        metrics = evaluate_ranking_metrics(y_test, y_pred, k_values)
        metrics['model'] = name
        results.append(metrics)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Reorder columns to put model name first
    cols = ['model'] + [col for col in df.columns if col != 'model']
    df = df[cols]
    
    return df


def plot_ranking_metrics(
    metrics_dict: Dict[str, float], 
    k_values: List[int] = [5, 10, 20]
) -> None:
    """
    Plot ranking metrics for different K values.
    
    Parameters:
    metrics_dict (dict): Dictionary containing metrics
    k_values (list): List of K values to plot
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        k_labels = [f'@{k}' for k in k_values]
        
        precision_values = [metrics_dict.get(f'precision@{k}', 0) for k in k_values]
        recall_values = [metrics_dict.get(f'recall@{k}', 0) for k in k_values]
        ndcg_values = [metrics_dict.get(f'ndcg@{k}', 0) for k in k_values]
        
        axes[0].bar(k_labels, precision_values)
        axes[0].set_title('Precision@K')
        axes[0].set_ylabel('Precision')
        
        axes[1].bar(k_labels, recall_values)
        axes[1].set_title('Recall@K')
        axes[1].set_ylabel('Recall')
        
        axes[2].bar(k_labels, ndcg_values)
        axes[2].set_title('NDCG@K')
        axes[2].set_ylabel('NDCG')
        
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("Matplotlib not available for plotting")