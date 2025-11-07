# Technical Specifications for Ranking-Specific Metrics

## 1. Implementation of Precision@K, Recall@K, NDCG@K

### 1.1 Precision@K Implementation
- **Definition**: Proportion of recommended items that are relevant
- **Formula**: Precision@K = (Number of relevant items in top-K) / K
- **Use Case**: Measure relevance of top-K recommendations

### 1.2 Technical Implementation
```python
import numpy as np

def precision_at_k(y_true, y_pred, k=10):
    """
    Calculate Precision@K for recommendation evaluation
    
    Parameters:
    y_true: Array of true relevance scores (1 for relevant, 0 for not relevant)
    y_pred: Array of predicted scores/ranks
    k: Number of top items to consider
    
    Returns:
    Precision@K score
    """
    # Get top K predictions
    top_k_indices = np.argsort(y_pred)[::-1][:k]
    
    # Calculate precision
    relevant_items = np.sum(y_true[top_k_indices])
    precision = relevant_items / k
    
    return precision
```

### 1.3 Recall@K Implementation
- **Definition**: Proportion of relevant items that are recommended
- **Formula**: Recall@K = (Number of relevant items in top-K) / (Total relevant items)
- **Use Case**: Measure coverage of relevant items in top-K recommendations

### 1.4 Technical Implementation
```python
def recall_at_k(y_true, y_pred, k=10):
    """
    Calculate Recall@K for recommendation evaluation
    
    Parameters:
    y_true: Array of true relevance scores (1 for relevant, 0 for not relevant)
    y_pred: Array of predicted scores/ranks
    k: Number of top items to consider
    
    Returns:
    Recall@K score
    """
    # Get top K predictions
    top_k_indices = np.argsort(y_pred)[::-1][:k]
    
    # Calculate recall
    relevant_items = np.sum(y_true[top_k_indices])
    total_relevant = np.sum(y_true)
    
    if total_relevant == 0:
        return 0.0
    
    recall = relevant_items / total_relevant
    return recall
```

### 1.5 NDCG@K Implementation
- **Definition**: Normalized Discounted Cumulative Gain
- **Formula**: NDCG@K = DCG@K / IDCG@K
- **Use Case**: Measure ranking quality considering position of relevant items

### 1.6 Technical Implementation
```python
def ndcg_at_k(y_true, y_pred, k=10):
    """
    Calculate NDCG@K for recommendation evaluation
    
    Parameters:
    y_true: Array of true relevance scores
    y_pred: Array of predicted scores/ranks
    k: Number of top items to consider
    
    Returns:
    NDCG@K score
    """
    # Get top K predictions
    top_k_indices = np.argsort(y_pred)[::-1][:k]
    top_k_true = y_true[top_k_indices]
    
    # Calculate DCG
    dcg = np.sum((2 ** top_k_true - 1) / np.log2(np.arange(2, len(top_k_true) + 2)))
    
    # Calculate IDCG (ideal DCG)
    ideal_indices = np.argsort(y_true)[::-1][:k]
    ideal_true = y_true[ideal_indices]
    idcg = np.sum((2 ** ideal_true - 1) / np.log2(np.arange(2, len(ideal_true) + 2)))
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg
```

## 2. Additional Ranking Metrics

### 2.1 Mean Average Precision@K (MAP@K)
- **Definition**: Mean of average precision scores for all users
- **Use Case**: Comprehensive measure of ranking quality across users

### 2.2 Technical Implementation
```python
def average_precision_at_k(y_true, y_pred, k=10):
    """
    Calculate Average Precision@K for a single user
    """
    top_k_indices = np.argsort(y_pred)[::-1][:k]
    top_k_true = y_true[top_k_indices]
    
    if np.sum(top_k_true) == 0:
        return 0.0
    
    score = 0.0
    num_hits = 0.0
    
    for i, p in enumerate(top_k_true):
        if p == 1:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    
    return score / np.sum(y_true)

def mean_average_precision_at_k(y_true_list, y_pred_list, k=10):
    """
    Calculate Mean Average Precision@K across all users
    """
    ap_scores = []
    for y_true, y_pred in zip(y_true_list, y_pred_list):
        ap = average_precision_at_k(y_true, y_pred, k)
        ap_scores.append(ap)
    
    return np.mean(ap_scores)
```

### 2.3 Mean Reciprocal Rank (MRR)
- **Definition**: Average of reciprocal ranks of first relevant item
- **Use Case**: Measure when first relevant item appears in ranking

### 2.4 Technical Implementation
```python
def mean_reciprocal_rank(y_true, y_pred):
    """
    Calculate Mean Reciprocal Rank
    """
    # Sort by predicted scores
    sorted_indices = np.argsort(y_pred)[::-1]
    sorted_true = y_true[sorted_indices]
    
    # Find first relevant item
    for i, relevance in enumerate(sorted_true):
        if relevance == 1:
            return 1.0 / (i + 1)
    
    return 0.0

def mean_reciprocal_rank_at_k(y_true_list, y_pred_list):
    """
    Calculate Mean Reciprocal Rank across all users
    """
    mrr_scores = []
    for y_true, y_pred in zip(y_true_list, y_pred_list):
        mrr = mean_reciprocal_rank(y_true, y_pred)
        mrr_scores.append(mrr)
    
    return np.mean(mrr_scores)
```

## 3. Integration with Existing Evaluation Framework

### 3.1 Scikit-learn Compatible Scorers
- **Purpose**: Enable use of ranking metrics with scikit-learn's evaluation tools
- **Implementation**: Use `make_scorer` to create custom scorers

### 3.2 Technical Implementation
```python
from sklearn.metrics import make_scorer

# Create scorers for ranking metrics
precision_k_scorer = make_scorer(precision_at_k, k=10)
recall_k_scorer = make_scorer(recall_at_k, k=10)
ndcg_k_scorer = make_scorer(ndcg_at_k, k=10)

# Use with cross-validation
from sklearn.model_selection import cross_validate

scoring = {
    'precision@10': precision_k_scorer,
    'recall@10': recall_k_scorer,
    'ndcg@10': ndcg_k_scorer
}

cv_results = cross_validate(model, X, y, scoring=scoring, cv=5)
```

### 3.3 Comprehensive Evaluation Pipeline
- **Purpose**: Evaluate models using multiple ranking metrics
- **Implementation**: Create pipeline that calculates all relevant metrics

### 3.4 Technical Implementation
```python
def ranking_evaluation_pipeline(model, X_test, y_test, k_values=[5, 10, 20]):
    """
    Comprehensive ranking evaluation pipeline
    """
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics for different K values
    results = {}
    for k in k_values:
        precision = precision_at_k(y_test, y_pred, k)
        recall = recall_at_k(y_test, y_pred, k)
        ndcg = ndcg_at_k(y_test, y_pred, k)
        map_score = average_precision_at_k(y_test, y_pred, k)
        
        results[f'precision@{k}'] = precision
        results[f'recall@{k}'] = recall
        results[f'ndcg@{k}'] = ndcg
        results[f'map@{k}'] = map_score
    
    # Calculate MRR
    mrr = mean_reciprocal_rank(y_test, y_pred)
    results['mrr'] = mrr
    
    return results
```

## 4. User-Level Evaluation

### 4.1 Per-User Metrics
- **Purpose**: Evaluate model performance for individual users
- **Implementation**: Calculate metrics for each user separately

### 4.2 Technical Implementation
```python
def per_user_ranking_evaluation(model, X_test, y_test, user_ids, k=10):
    """
    Calculate ranking metrics per user
    """
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Group by user
    user_metrics = {}
    unique_users = np.unique(user_ids)
    
    for user in unique_users:
        user_mask = user_ids == user
        user_y_true = y_test[user_mask]
        user_y_pred = y_pred[user_mask]
        
        precision = precision_at_k(user_y_true, user_y_pred, k)
        recall = recall_at_k(user_y_true, user_y_pred, k)
        ndcg = ndcg_at_k(user_y_true, user_y_pred, k)
        
        user_metrics[user] = {
            'precision@10': precision,
            'recall@10': recall,
            'ndcg@10': ndcg
        }
    
    return user_metrics
```

### 4.3 Aggregated User Metrics
- **Purpose**: Aggregate per-user metrics for overall evaluation
- **Implementation**: Calculate mean, median, and distribution of user metrics

## 5. Statistical Significance Testing

### 5.1 Bootstrap Confidence Intervals
- **Purpose**: Calculate confidence intervals for ranking metrics
- **Implementation**: Use bootstrap sampling to estimate metric variance

### 5.2 Technical Implementation
```python
def bootstrap_ranking_metrics(y_true, y_pred, k=10, n_bootstrap=1000, confidence=0.95):
    """
    Calculate bootstrap confidence intervals for ranking metrics
    """
    precision_scores = []
    recall_scores = []
    ndcg_scores = []
    
    n_samples = len(y_true)
    
    for _ in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(n_samples, n_samples, replace=True)
        boot_y_true = y_true[indices]
        boot_y_pred = y_pred[indices]
        
        # Calculate metrics
        precision = precision_at_k(boot_y_true, boot_y_pred, k)
        recall = recall_at_k(boot_y_true, boot_y_pred, k)
        ndcg = ndcg_at_k(boot_y_true, boot_y_pred, k)
        
        precision_scores.append(precision)
        recall_scores.append(recall)
        ndcg_scores.append(ndcg)
    
    # Calculate confidence intervals
    alpha = 1 - confidence
    lower_percentile = (alpha/2) * 100
    upper_percentile = (1 - alpha/2) * 100
    
    precision_ci = np.percentile(precision_scores, [lower_percentile, upper_percentile])
    recall_ci = np.percentile(recall_scores, [lower_percentile, upper_percentile])
    ndcg_ci = np.percentile(ndcg_scores, [lower_percentile, upper_percentile])
    
    return {
        'precision@10': {'mean': np.mean(precision_scores), 'ci': precision_ci},
        'recall@10': {'mean': np.mean(recall_scores), 'ci': recall_ci},
        'ndcg@10': {'mean': np.mean(ndcg_scores), 'ci': ndcg_ci}
    }
```

### 5.3 Statistical Tests for Model Comparison
- **Purpose**: Determine if differences in metrics are statistically significant
- **Implementation**: Use paired t-tests or Wilcoxon signed-rank tests

## 6. Business-Oriented Metrics

### 6.1 Click-Through Rate (CTR)
- **Definition**: Proportion of recommended items that are clicked
- **Use Case**: Direct measure of user engagement with recommendations

### 6.2 Technical Implementation
```python
def click_through_rate(clicked_items, recommended_items):
    """
    Calculate Click-Through Rate
    """
    if len(recommended_items) == 0:
        return 0.0
    
    clicked_count = len(set(clicked_items) & set(recommended_items))
    ctr = clicked_count / len(recommended_items)
    
    return ctr
```

### 6.3 Conversion Rate
- **Definition**: Proportion of recommended items that lead to purchases
- **Use Case**: Measure business impact of recommendations

## 7. Visualization and Reporting

### 7.1 Metric Dashboards
- **Purpose**: Visualize ranking metrics over time
- **Implementation**: Create plots showing metric trends

### 7.2 Technical Implementation
```python
import matplotlib.pyplot as plt

def plot_ranking_metrics(metrics_dict, k_values=[5, 10, 20]):
    """
    Plot ranking metrics for different K values
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    k_labels = [f'@{k}' for k in k_values]
    
    precision_values = [metrics_dict[f'precision@{k}'] for k in k_values]
    recall_values = [metrics_dict[f'recall@{k}'] for k in k_values]
    ndcg_values = [metrics_dict[f'ndcg@{k}'] for k in k_values]
    
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
```

### 7.3 Detailed Reporting
- **Purpose**: Generate comprehensive reports on model performance
- **Implementation**: Create structured reports with all relevant metrics

## 8. Performance Considerations

### 8.1 Computational Efficiency
- **Optimization**: Vectorize calculations where possible
- **Memory Management**: Process large datasets in chunks
- **Caching**: Cache intermediate results to avoid recomputation

### 8.2 Scalability
- **Parallel Processing**: Use parallel processing for large-scale evaluation
- **Approximation Methods**: Use sampling for very large datasets
- **Incremental Updates**: Update metrics incrementally for streaming data

## 9. Validation and Testing

### 9.1 Unit Testing
- **Individual Metrics**: Test each metric implementation with known inputs
- **Edge Cases**: Test with edge cases (no relevant items, all relevant items, etc.)
- **Consistency**: Ensure metrics are consistent with mathematical definitions

### 9.2 Integration Testing
- **Pipeline Integration**: Test integration with model evaluation pipeline
- **Cross-Validation**: Test use with temporal cross-validation
- **Scikit-learn Compatibility**: Test compatibility with scikit-learn tools

### 9.3 Performance Testing
- **Large Datasets**: Test performance with large datasets
- **Memory Usage**: Monitor memory usage during evaluation
- **Execution Time**: Measure execution time for different dataset sizes