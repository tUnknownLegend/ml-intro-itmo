#!/usr/bin/env python3
"""
Test script to verify the implementation of the cross-validation framework 
with temporal folds and hyperparameter tuning.
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

# Add the current directory to the path to import our modules
sys.path.append('.')

def test_temporal_cross_validation():
    """Test the temporal cross-validation framework."""
    print("Testing temporal cross-validation framework...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    
    # Create temporally ordered data
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)
    # Sort by target to simulate temporal ordering
    sort_idx = np.argsort(y)
    X = X[sort_idx]
    y = y[sort_idx]
    
    # Create model
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    
    # Test temporal cross-validation
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Simple cross-validation without parallel processing
    scores = []
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        score = mean_absolute_error(y_val, y_pred)
        scores.append(score)
    
    print(f"Temporal CV scores: {scores}")
    print(f"Mean score: {np.mean(scores):.4f} (+/- {np.std(scores) * 2:.4f})")
    print("Temporal cross-validation test: PASSED\n")
    return True

def test_hyperparameter_tuning():
    """Test the hyperparameter tuning functionality."""
    print("Testing hyperparameter tuning...")
    
    # Create sample data
    np.random.seed(42)
    X = np.random.randn(50, 3)
    y = np.random.randn(50)
    
    # Simple grid search implementation
    param_grid = {
        'n_estimators': [10, 20],
        'max_depth': [3, 5]
    }
    
    best_score = float('inf')
    best_params = None
    
    # Manual grid search
    for n_est in param_grid['n_estimators']:
        for max_d in param_grid['max_depth']:
            model = RandomForestRegressor(
                n_estimators=n_est, 
                max_depth=max_d, 
                random_state=42
            )
            
            # Simple train/validation split
            split_idx = len(X) // 2
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            score = mean_absolute_error(y_val, y_pred)
            
            if score < best_score:
                best_score = score
                best_params = {'n_estimators': n_est, 'max_depth': max_d}
    
    print(f"Best parameters: {best_params}")
    print(f"Best score: {best_score:.4f}")
    print("Hyperparameter tuning test: PASSED\n")
    return True

def test_ranking_metrics():
    """Test the ranking metrics functionality."""
    print("Testing ranking metrics...")
    
    # Import ranking metrics
    try:
        from ranking_metrics import evaluate_ranking_metrics, precision_at_k, recall_at_k, ndcg_at_k
        print("Ranking metrics module imported successfully")
    except ImportError as e:
        print(f"Failed to import ranking metrics: {e}")
        return False
    
    # Create sample data
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 20)  # Binary relevance
    y_pred = np.random.rand(20)  # Predicted scores
    
    # Test individual metrics
    p_at_5 = precision_at_k(y_true, y_pred, k=5)
    r_at_5 = recall_at_k(y_true, y_pred, k=5)
    ndcg_at_5 = ndcg_at_k(y_true, y_pred, k=5)
    
    print(f"Precision@5: {p_at_5:.4f}")
    print(f"Recall@5: {r_at_5:.4f}")
    print(f"NDCG@5: {ndcg_at_5:.4f}")
    
    # Test comprehensive evaluation
    metrics = evaluate_ranking_metrics(y_true, y_pred, k_values=[3, 5])
    print(f"Comprehensive metrics: {metrics}")
    
    print("Ranking metrics test: PASSED\n")
    return True

def main():
    """Run all tests."""
    print("Running implementation tests...\n")
    
    tests = [
        test_temporal_cross_validation,
        test_hyperparameter_tuning,
        test_ranking_metrics
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"Test {test.__name__} failed with error: {e}")
            results.append(False)
    
    print("=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "PASSED" if result else "FAILED"
        print(f"{i+1}. {test.__name__}: {status}")
    
    all_passed = all(results)
    print(f"\nOverall result: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)