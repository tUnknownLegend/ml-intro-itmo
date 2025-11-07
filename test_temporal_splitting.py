import pandas as pd
import numpy as np

def test_temporal_splitting():
    """Test that temporal splitting prevents data leakage."""
    # Create a simple test dataset with timestamps
    np.random.seed(42)
    n_samples = 1000
    
    # Create data with clear temporal pattern
    timestamps = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    # Create target values that increase over time to simulate temporal pattern
    view_count = np.arange(n_samples) + np.random.normal(0, 10, n_samples)
    
    # Create a DataFrame
    df = pd.DataFrame({
        'user_id': np.random.randint(1, 100, n_samples),
        'item_id': np.random.randint(1, 50, n_samples),
        'view_count': view_count,
        'last_view_timestamp': timestamps,
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples)
    })
    
    # Sort by timestamp to ensure temporal order
    df = df.sort_values('last_view_timestamp').reset_index(drop=True)
    
    # Implement temporal splitting (60% train, 20% validation, 20% test)
    n_samples = len(df)
    train_end = int(n_samples * 0.6)
    val_end = int(n_samples * 0.8)
    
    # Split the data temporally
    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]
    
    # Check that timestamps are properly ordered
    train_max_time = train_df['last_view_timestamp'].max()
    val_min_time = val_df['last_view_timestamp'].min()
    val_max_time = val_df['last_view_timestamp'].max()
    test_min_time = test_df['last_view_timestamp'].min()
    
    # Verify no data leakage (train timestamps should be before validation timestamps,
    # and validation timestamps should be before test timestamps)
    assert train_max_time <= val_min_time, "Data leakage detected: train data contains timestamps after validation data"
    assert val_max_time <= test_min_time, "Data leakage detected: validation data contains timestamps after test data"
    
    # Verify the split proportions
    assert len(train_df) == train_end, f"Train set size incorrect: expected {train_end}, got {len(train_df)}"
    assert len(val_df) == val_end - train_end, f"Validation set size incorrect: expected {val_end - train_end}, got {len(val_df)}"
    assert len(test_df) == n_samples - val_end, f"Test set size incorrect: expected {n_samples - val_end}, got {len(test_df)}"
    
    print("Temporal splitting test passed!")
    print(f"Train set: {len(train_df)} samples ({len(train_df)/n_samples*100:.1f}%)")
    print(f"Validation set: {len(val_df)} samples ({len(val_df)/n_samples*100:.1f}%)")
    print(f"Test set: {len(test_df)} samples ({len(test_df)/n_samples*100:.1f}%)")
    print(f"Train time range: {train_df['last_view_timestamp'].min()} to {train_df['last_view_timestamp'].max()}")
    print(f"Validation time range: {val_df['last_view_timestamp'].min()} to {val_df['last_view_timestamp'].max()}")
    print(f"Test time range: {test_df['last_view_timestamp'].min()} to {test_df['last_view_timestamp'].max()}")
    
    # Check for data leakage in target values
    train_mean = train_df['view_count'].mean()
    val_mean = val_df['view_count'].mean()
    test_mean = test_df['view_count'].mean()
    
    print(f"Train target mean: {train_mean:.2f}")
    print(f"Validation target mean: {val_mean:.2f}")
    print(f"Test target mean: {test_mean:.2f}")
    
    # With proper temporal splitting, we expect target values to increase over time
    # This shows that our splitting preserves the temporal pattern
    assert val_mean > train_mean, "Temporal pattern not preserved: validation mean should be higher than train mean"
    assert test_mean > val_mean, "Temporal pattern not preserved: test mean should be higher than validation mean"
    
    print("Temporal pattern preserved correctly!")
    
    return True

if __name__ == "__main__":
    test_temporal_splitting()