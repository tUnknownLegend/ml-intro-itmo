# Technical Specifications for Enhanced Feature Engineering

## 1. Categorical Encoding Techniques

### 1.1 One-Hot Encoding Implementation
- **Library**: scikit-learn's `OneHotEncoder`
- **Target Variables**: 
  - `action_type` (events dataset)
  - `os` (events dataset)
  - `subdomain` (events dataset)
- **Parameters**:
  - `sparse_output=False` (for compatibility with other transformers)
  - `handle_unknown='ignore'` (to handle new categories in test data)
- **Implementation Steps**:
  1. Initialize encoder with specified parameters
  2. Fit encoder on training data
  3. Transform both training and test data
  4. Handle missing values by treating them as a separate category

### 1.2 Target Encoding Implementation
- **Custom Implementation**: With smoothing to prevent overfitting
- **Target Variables**:
  - `user_id` (high cardinality)
  - `item_id` (high cardinality)
  - `brand_id` (medium cardinality)
  - `category` (medium cardinality)
  - `subcategory` (high cardinality)
- **Parameters**:
  - `min_samples_leaf=20` (minimum samples per category)
  - `smoothing=10` (smoothing factor for regularization)
- **Implementation Steps**:
  1. Calculate global mean of target variable
  2. Calculate category-wise statistics (count, mean)
  3. Apply smoothing formula: `(count * mean + smoothing * global_mean) / (count + smoothing)`
  4. Map categories to smoothed values
  5. Fill missing categories with global mean

### 1.3 Frequency Encoding Implementation
- **Custom Implementation**: Map categories to their frequency counts
- **Target Variables**: All categorical variables where frequency is informative
- **Implementation Steps**:
  1. Calculate value counts for each category
  2. Create mapping dictionary from category to frequency
  3. Map categories to frequencies
  4. Handle unseen categories with zero or small constant

## 2. Feature Scaling and Normalization

### 2.1 Standardization (Z-score Normalization)
- **Library**: scikit-learn's `StandardScaler`
- **Target Variables**: Numerical features with approximately normal distribution
  - `price` (after handling outliers)
  - `socdem_cluster`
  - `region`
- **Parameters**: Default parameters
- **Implementation Steps**:
  1. Initialize scaler
  2. Fit on training data (mean=0, std=1)
  3. Transform both training and test data

### 2.2 Min-Max Scaling
- **Library**: scikit-learn's `MinMaxScaler`
- **Target Variables**: Numerical features that need to be bounded
  - `user_activity_level`
  - `item_popularity`
  - `price` (alternative to standardization)
- **Parameters**: 
  - `feature_range=(0, 1)` (default)
- **Implementation Steps**:
  1. Initialize scaler
  2. Fit on training data (min=0, max=1)
  3. Transform both training and test data

### 2.3 Robust Scaling
- **Library**: scikit-learn's `RobustScaler`
- **Target Variables**: Numerical features with significant outliers
  - `price` (if outliers are present)
- **Parameters**: Default parameters (uses median and IQR)
- **Implementation Steps**:
  1. Initialize scaler
  2. Fit on training data (median=0, IQR=1)
  3. Transform both training and test data

## 3. Advanced Feature Creation

### 3.1 User Preference Features

#### User Activity Level
- **Features**:
  - `total_events`: Count of all events per user
  - `days_since_last_activity`: Time since user's last interaction
  - `activity_recency_score`: Exponentially decaying score based on recency
- **Implementation**:
  ```python
  user_activity = events_df.groupby('user_id').agg({
      'timestamp': ['count', 'max'],
  }).reset_index()
  user_activity.columns = ['user_id', 'total_events', 'last_activity']
  user_activity['days_since_last_activity'] = (max_timestamp - user_activity['last_activity']).dt.days
  user_activity['activity_recency_score'] = np.exp(-lambda_param * user_activity['days_since_last_activity'])
  ```

#### Category Preferences
- **Features**:
  - `category_preference`: Proportion of views in each category per user
  - `diversity_score`: Number of different categories viewed
- **Implementation**:
  ```python
  user_category_views = events_df.merge(items_df[['item_id', 'category']], on='item_id')
  user_category_views = user_category_views.groupby(['user_id', 'category']).size().reset_index(name='views')
  user_category_views['category_preference'] = user_category_views.groupby('user_id')['views'].transform(lambda x: x / x.sum())
  user_diversity = user_category_views.groupby('user_id')['category'].nunique().reset_index()
  user_diversity.columns = ['user_id', 'diversity_score']
  ```

### 3.2 Item Freshness Features

#### Item Age
- **Features**:
  - `item_age_days`: Days since item was first viewed
  - `is_new_item`: Binary flag for items less than 7 days old
- **Implementation**:
  ```python
  item_first_view = events_df.groupby('item_id')['timestamp'].min().reset_index()
  item_first_view.columns = ['item_id', 'first_view_date']
  items_df = items_df.merge(item_first_view, on='item_id', how='left')
  items_df['item_age_days'] = (max_timestamp - items_df['first_view_date']).dt.days
  items_df['is_new_item'] = (items_df['item_age_days'] <= 7).astype(int)
  ```

#### Recent Popularity
- **Features**:
  - `recent_popularity`: Exponentially weighted view count
  - `trending_score`: Popularity relative to recent average
- **Implementation**:
  ```python
  events_df['time_decay'] = np.exp(-lambda_param * (max_timestamp - events_df['timestamp']).dt.days)
  item_popularity = events_df.groupby('item_id')['time_decay'].sum().reset_index()
  item_popularity.columns = ['item_id', 'recent_popularity']
  ```

### 3.3 User-Item Interaction Features

#### View Count Features
- **Features**:
  - `user_item_view_count`: Number of times user viewed item
  - `view_frequency`: Average views per day
- **Implementation**:
  ```python
  user_item_views = events_df.groupby(['user_id', 'item_id']).size().reset_index(name='view_count')
  user_item_first_view = events_df.groupby(['user_id', 'item_id'])['timestamp'].min().reset_index()
  user_item_first_view.columns = ['user_id', 'item_id', 'first_view_date']
  user_item_views = user_item_views.merge(user_item_first_view, on=['user_id', 'item_id'])
  user_item_views['days_since_first_view'] = (max_timestamp - user_item_views['first_view_date']).dt.days
  user_item_views['view_frequency'] = user_item_views['view_count'] / user_item_views['days_since_first_view']
  ```

#### Temporal Interaction Features
- **Features**:
  - `days_since_last_view`: Time since last view of item
  - `view_recency_score`: Exponentially decaying score
- **Implementation**:
  ```python
  user_item_last_view = events_df.groupby(['user_id', 'item_id'])['timestamp'].max().reset_index()
  user_item_last_view.columns = ['user_id', 'item_id', 'last_view_date']
  user_item_last_view['days_since_last_view'] = (max_timestamp - user_item_last_view['last_view_date']).dt.days
  user_item_last_view['view_recency_score'] = np.exp(-lambda_param * user_item_last_view['days_since_last_view'])
  ```

### 3.4 Temporal Features

#### Cyclical Encoding
- **Features**:
  - `hour_sin`, `hour_cos`: Cyclical encoding of hour
  - `day_sin`, `day_cos`: Cyclical encoding of day of week
- **Implementation**:
  ```python
  events_df['hour_sin'] = np.sin(2 * np.pi * events_df['timestamp'].dt.hour / 24)
  events_df['hour_cos'] = np.cos(2 * np.pi * events_df['timestamp'].dt.hour / 24)
  events_df['day_sin'] = np.sin(2 * np.pi * events_df['timestamp'].dt.dayofweek / 7)
  events_df['day_cos'] = np.cos(2 * np.pi * events_df['timestamp'].dt.dayofweek / 7)
  ```

#### Time-Based Aggregates
- **Features**:
  - `morning_views`: Views during morning hours (6-12)
  - `weekend_views`: Views during weekends
- **Implementation**:
  ```python
  events_df['is_morning'] = (events_df['timestamp'].dt.hour >= 6) & (events_df['timestamp'].dt.hour < 12)
  events_df['is_weekend'] = events_df['timestamp'].dt.dayofweek >= 5
  morning_views = events_df[events_df['is_morning']].groupby(['user_id', 'item_id']).size().reset_index(name='morning_views')
  weekend_views = events_df[events_df['is_weekend']].groupby(['user_id', 'item_id']).size().reset_index(name='weekend_views')
  ```

## 4. Feature Selection Techniques

### 4.1 Correlation-Based Feature Selection
- **Method**: Remove highly correlated features
- **Threshold**: |correlation| > 0.95
- **Implementation**:
  ```python
  correlation_matrix = df.select_dtypes(include=[np.number]).corr().abs()
  upper_triangle = correlation_matrix.where(
      np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
  )
  high_corr_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]
  df_selected = df.drop(columns=high_corr_features)
  ```

### 4.2 Variance Threshold
- **Library**: scikit-learn's `VarianceThreshold`
- **Threshold**: 0.01 (remove features with very low variance)
- **Implementation**:
  ```python
  from sklearn.feature_selection import VarianceThreshold
  selector = VarianceThreshold(threshold=0.01)
  selected_features = selector.fit_transform(df.select_dtypes(include=[np.number]))
  ```

### 4.3 Recursive Feature Elimination
- **Library**: scikit-learn's `RFE`
- **Estimator**: Random Forest or Gradient Boosting
- **Parameters**:
  - `n_features_to_select`: 50 (or appropriate number)
  - `step`: 10 (features to remove per iteration)
- **Implementation**:
  ```python
  from sklearn.feature_selection import RFE
  from sklearn.ensemble import RandomForestRegressor
  estimator = RandomForestRegressor(n_estimators=100, random_state=42)
  selector = RFE(estimator, n_features_to_select=50, step=10)
  selected_features = selector.fit_transform(X, y)
  ```

## 5. Implementation Pipeline

### 5.1 Feature Engineering Pipeline
- **Library**: scikit-learn's `Pipeline` and `ColumnTransformer`
- **Structure**:
  1. Categorical encoding (OneHotEncoder, TargetEncoder)
  2. Numerical scaling (StandardScaler, MinMaxScaler)
  3. Feature creation (custom transformers)
  4. Feature selection (VarianceThreshold, RFE)

### 5.2 Custom Transformers
- **UserPreferenceTransformer**: Creates user preference features
- **ItemFreshnessTransformer**: Creates item freshness features
- **TemporalFeatureTransformer**: Creates temporal features
- **InteractionFeatureTransformer**: Creates user-item interaction features

### 5.3 Pipeline Integration
- **Temporal Constraints**: Ensure features are created using only historical data
- **Cross-Validation Compatibility**: Pipeline works with temporal cross-validation
- **Reproducibility**: Fixed random seeds for all stochastic processes