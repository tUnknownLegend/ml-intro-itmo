import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


def compare_splitting_methods():
    """Compare random splitting vs temporal splitting to show the importance of proper temporal splitting."""
    # Create a dataset with clear temporal pattern
    np.random.seed(42)
    n_samples = 1000

    # Create timestamps
    timestamps = pd.date_range('2020-01-01', periods=n_samples, freq='D')

    # Create features that change over time
    # Feature values increase over time to simulate concept drift
    feature1 = np.arange(n_samples) * 0.01 + \
        np.random.normal(0, 0.1, n_samples)
    feature2 = np.sin(np.arange(n_samples) * 0.01) + \
        np.random.normal(0, 0.1, n_samples)

    # Create target values that depend on features and time
    # This simulates a scenario where the relationship between features and target changes over time
    time_factor = np.arange(n_samples) * 0.005  # Increasing trend over time
    view_count = (feature1 * 2 + feature2 * 3 + time_factor +
                  np.random.normal(0, 0.5, n_samples))

    # Create a DataFrame
    df = pd.DataFrame({
        'feature1': feature1,
        'feature2': feature2,
        'view_count': view_count,
        'timestamp': timestamps
    })

    print("Dataset created with temporal patterns")
    print(f"Dataset shape: {df.shape}")
    print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Method 1: Random splitting (incorrect approach that causes data leakage)
    print("\n" + "="*50)
    print("METHOD 1: Random Splitting (INCORRECT - causes data leakage)")
    print("="*50)

    X = df[['feature1', 'feature2']]
    y = df['view_count']

    X_train_rand, X_test_rand, y_train_rand, y_test_rand = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model with random split
    model_rand = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rand.fit(X_train_rand, y_train_rand)

    # Evaluate on test set
    y_pred_rand = model_rand.predict(X_test_rand)
    mae_rand = mean_absolute_error(y_test_rand, y_pred_rand)

    print(f"Random splitting - Test MAE: {mae_rand:.4f}")

    # Method 2: Temporal splitting (correct approach)
    print("\n" + "="*50)
    print("METHOD 2: Temporal Splitting (CORRECT - prevents data leakage)")
    print("="*50)

    # Sort by timestamp to ensure temporal order
    df_sorted = df.sort_values('timestamp').reset_index(drop=True)

    # Implement temporal splitting (80% train, 20% test)
    n_samples = len(df_sorted)
    train_end = int(n_samples * 0.8)

    # Split the data temporally
    train_df = df_sorted[:train_end]
    test_df = df_sorted[train_end:]

    X_train_temp = train_df[['feature1', 'feature2']]
    y_train_temp = train_df['view_count']
    X_test_temp = test_df[['feature1', 'feature2']]
    y_test_temp = test_df['view_count']

    print(
        f"Temporal split - Train samples: {len(train_df)} ({len(train_df)/n_samples*100:.1f}%)")
    print(
        f"Temporal split - Test samples: {len(test_df)} ({len(test_df)/n_samples*100:.1f}%)")
    print(
        f"Train time range: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
    print(
        f"Test time range: {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")

    # Train model with temporal split
    model_temp = RandomForestRegressor(n_estimators=100, random_state=42)
    model_temp.fit(X_train_temp, y_train_temp)

    # Evaluate on test set
    y_pred_temp = model_temp.predict(X_test_temp)
    mae_temp = mean_absolute_error(y_test_temp, y_pred_temp)

    print(f"Temporal splitting - Test MAE: {mae_temp:.4f}")

    # Compare results
    print("\n" + "="*50)
    print("COMPARISON")
    print("="*50)
    print(f"Random splitting MAE: {mae_rand:.4f}")
    print(f"Temporal splitting MAE: {mae_temp:.4f}")
    print(f"Difference: {abs(mae_temp - mae_rand):.4f}")

    # With temporal splitting, we expect worse performance because:
    # 1. The model is trained on older data
    # 2. The relationship between features and target may have changed over time
    # 3. This reflects real-world conditions where models need to generalize to future data
    if mae_temp > mae_rand:
        print(
            "\nResult: Temporal splitting shows higher error, which is EXPECTED and CORRECT")
        print("This indicates that the model's performance degrades over time, which is realistic")
        print("Random splitting artificially inflated performance by leaking future information")
    else:
        print("\nResult: Unexpected - temporal splitting shows lower error than random splitting")
        print("This might indicate the temporal pattern is not strong enough in this simulation")

    # Additional analysis: Check if temporal pattern exists
    train_mean_target = train_df['view_count'].mean()
    test_mean_target = test_df['view_count'].mean()

    print(f"\nTemporal pattern analysis:")
    print(f"Train target mean: {train_mean_target:.4f}")
    print(f"Test target mean: {test_mean_target:.4f}")
    print(f"Change in target mean: {test_mean_target - train_mean_target:.4f}")

    return mae_rand, mae_temp


if __name__ == "__main__":
    compare_splitting_methods()
