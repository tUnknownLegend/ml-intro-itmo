# Feature Engineering Plan for T-ECD Dataset

## Executive Summary

This document outlines a comprehensive feature engineering strategy for the T-ECD dataset based on the findings from the Exploratory Data Analysis (EDA). The plan addresses key data quality issues, missing value patterns, and opportunities for creating predictive features that will improve model performance for the personalized product recommendation system.

## Data Quality Issues and Missing Value Patterns

### Key Findings from EDA

1. **Items Dataset**:
   - `category`: 41.56% missing values (966,395 out of 2,325,409)
   - `subcategory`: 53.02% missing values (1,233,023 out of 2,325,409)
   - `price`: 0.12% missing values (2,882 out of 2,322,527)
   - `embedding`: 0.003% missing values (73 out of 2,325,409)

2. **Brands Dataset**:
   - `embedding`: 73.24% missing values (17,954 out of 24,513)

3. **Users Dataset**:
   - `region`: 1.68% missing values (58,917 out of 3,500,000)
   - `socdem_cluster`: 0.15% missing values (5,153 out of 3,494,847)

4. **Events Dataset**:
   - `subdomain`: 0.02% missing values (1,453 out of 6,627,693)

## Feature Engineering Recommendations

### 1. Handling Missing Values in Category and Subcategory Columns

#### Problem
The `category` and `subcategory` columns in the items dataset have significant missing values (41.56% and 53.02% respectively), which presents a challenge for feature creation.

#### Recommended Approaches

1. **Create Missingness Indicators**:
   - Create binary flags indicating whether `category` or `subcategory` is missing
   - This preserves information about the missingness pattern which might be predictive

2. **Hierarchical Imputation**:
   - For items with missing `subcategory` but available `category`, use the mode `subcategory` for that `category`
   - For items with missing `category`, use the mode `category` for items with the same `brand_id`

3. **Cluster-Based Imputation**:
   - Use price and brand information to group similar items
   - Impute missing categories based on the most common category in each cluster

4. **"Unknown" Category**:
   - For items where imputation is not possible, assign a special "unknown" category
   - This preserves the information that these items lack categorization

#### Implementation Strategy
```
# Pseudocode for implementation
items_df['category_missing'] = items_df['category'].isnull().astype(int)
items_df['subcategory_missing'] = items_df['subcategory'].isnull().astype(int)

# Impute based on brand_id
category_mode_by_brand = items_df.groupby('brand_id')['category'].agg(
    lambda x: x.mode().iloc[0] if not x.mode().empty else 'unknown'
)
items_df['category'] = items_df['category'].fillna(
    items_df['brand_id'].map(category_mode_by_brand)
)

# For remaining missing, assign 'unknown'
items_df['category'] = items_df['category'].fillna('unknown')
items_df['subcategory'] = items_df['subcategory'].fillna('unknown')
```

### 2. Utilizing Embedding Data for Brands and Items

#### Problem
The embedding columns for both brands and items have high percentages of missing values (73.24% for brands, 0.003% for items), but when available, embeddings can provide rich feature representations.

#### Recommended Approaches

1. **Dimensionality Reduction**:
   - Apply PCA or t-SNE to reduce embedding dimensions to a manageable size (e.g., 10-50 components)
   - This creates more interpretable features while preserving important information

2. **Statistical Features from Embeddings**:
   - Extract statistical measures from available embeddings:
     - Mean, median, standard deviation of embedding values
     - Min/max values
     - L2 norm of embeddings

3. **Clustering Based on Embeddings**:
   - Use available embeddings to create clusters of similar brands/items
   - Assign cluster IDs as categorical features for all items

4. **Missing Embedding Handling**:
   - Create a binary indicator for missing embeddings
   - For items without embeddings, assign them to a special "no_embedding" cluster
   - Use other available features (price, category, brand) to impute embedding cluster membership

#### Implementation Strategy
```
# Pseudocode for implementation
# Extract statistical features from embeddings
items_df['embedding_mean'] = items_df['embedding'].apply(
    lambda x: np.mean(x) if x is not None and not pd.isna(x) else 0
)
items_df['embedding_std'] = items_df['embedding'].apply(
    lambda x: np.std(x) if x is not None and not pd.isna(x) and len(x) > 1 else 0
)
items_df['embedding_norm'] = items_df['embedding'].apply(
    lambda x: np.linalg.norm(x) if x is not None and not pd.isna(x) else 0
)

# Create missing indicator
items_df['embedding_missing'] = items_df['embedding'].isnull().astype(int)
```

### 3. Temporal Feature Creation from Events Data

#### Problem
The events dataset contains timestamp information that spans approximately 10 days, with potential patterns by action type. These temporal patterns can be leveraged for feature creation.

#### Recommended Approaches

1. **Time-Based Features**:
   - Extract time components from timestamps:
     - Hour of day (cyclical encoding)
     - Day of week (cyclical encoding)
     - Time since start of dataset
     - Time to end of dataset

2. **User Behavior Temporal Patterns**:
   - Calculate user activity patterns:
     - Number of events per hour/day
     - Time since last event
     - Session duration metrics
     - Peak activity times for each user

3. **Item Popularity Over Time**:
   - Track item popularity trends:
     - Views per day
     - Recent popularity score (exponentially weighted)
     - Time since first view
     - Seasonal patterns

4. **Action Sequence Features**:
   - Create features based on action sequences:
     - Previous action type
     - Time between actions
     - Action transition patterns

#### Implementation Strategy
```
# Pseudocode for implementation
events_df['hour'] = events_df['timestamp'].dt.hour
events_df['day_of_week'] = events_df['timestamp'].dt.dayofweek
events_df['hour_sin'] = np.sin(2 * np.pi * events_df['hour'] / 24)
events_df['hour_cos'] = np.cos(2 * np.pi * events_df['hour'] / 24)
events_df['day_sin'] = np.sin(2 * np.pi * events_df['day_of_week'] / 7)
events_df['day_cos'] = np.cos(2 * np.pi * events_df['day_of_week'] / 7)

# User activity features
user_activity = events_df.groupby('user_id').agg({
    'timestamp': ['count', 'min', 'max'],
    'hour': lambda x: x.mode().iloc[0] if not x.mode().empty else -1
}).reset_index()
user_activity.columns = ['user_id', 'event_count', 'first_event', 'last_event', 'peak_hour']
user_activity['activity_duration'] = (
    user_activity['last_event'] - user_activity['first_event']
).dt.total_seconds()
```

### 4. Handling Price Outliers and Relationships

#### Problem
The price variable has outliers (minimum value of -10.0 and maximum of 10.0) that need to be addressed. Additionally, understanding relationships between price and other variables is important.

#### Recommended Approaches

1. **Outlier Detection and Treatment**:
   - Use IQR method to identify outliers
   - Apply winsorization to cap extreme values
   - Consider log transformation for skewed distributions
   - Create outlier indicators as additional features

2. **Price Segmentation**:
   - Create price buckets (low/medium/high/expensive)
   - Use domain knowledge to define meaningful price ranges
   - Create interaction features with categories

3. **Price Relationship Features**:
   - Calculate price relative to category average
   - Calculate price relative to brand average
   - Price rank within category/brand

4. **Missing Price Handling**:
   - Impute missing prices using category and brand medians
   - Create missing indicator feature

#### Implementation Strategy
```
# Pseudocode for implementation
# Outlier detection and treatment
Q1 = items_df['price'].quantile(0.25)
Q3 = items_df['price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

items_df['price_outlier'] = (
    (items_df['price'] < lower_bound) | (items_df['price'] > upper_bound)
).astype(int)

items_df['price_winsorized'] = items_df['price'].clip(
    lower=lower_bound, upper=upper_bound
)

# Price relative features
category_avg_price = items_df.groupby('category')['price'].median()
items_df['price_vs_category_avg'] = (
    items_df['price'] / items_df['category'].map(category_avg_price)
)

# Price segmentation
items_df['price_segment'] = pd.cut(
    items_df['price'], 
    bins=[-np.inf, 0, 1, 5, np.inf], 
    labels=['negative', 'low', 'medium', 'high']
)
```

### 5. Feature Engineering for Categorical Variables

#### Problem
Several categorical variables (region, socdem_cluster, brand_id, action_type, os, subdomain) need appropriate encoding for machine learning models.

#### Recommended Approaches

1. **Target Encoding**:
   - For high cardinality categorical variables, use target encoding with smoothing
   - Apply cross-validation to prevent overfitting
   - Add noise for regularization

2. **Frequency Encoding**:
   - Replace categories with their frequency counts
   - Useful for variables where frequency correlates with importance

3. **Embedding-Based Encoding**:
   - For categories with embeddings, use embedding vectors directly
   - For categories without embeddings, create learned embeddings

4. **Combination Features**:
   - Create interaction features between related categorical variables:
     - region × socdem_cluster
     - brand_id × category
     - action_type × os

#### Implementation Strategy
```
# Pseudocode for implementation
# Frequency encoding
region_counts = users_df['region'].value_counts()
users_df['region_frequency'] = users_df['region'].map(region_counts)

# Target encoding (simplified example)
# In practice, this should use cross-validation
def target_encode(train_df, test_df, column, target, min_samples_leaf=20, smoothing=10):
    # Calculate global mean
    prior = train_df[target].mean()
    
    # Calculate category statistics
    agg = train_df.groupby(column)[target].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']
    
    # Smoothed average
    smooth = (counts * means + smoothing * prior) / (counts + smoothing)
    
    # Apply to test set
    return test_df[column].map(smooth).fillna(prior)

# Combination features
users_df['region_socdem_combo'] = (
    users_df['region'].astype(str) + '_' + users_df['socdem_cluster'].astype(str)
)
```

### 6. Correlation Analysis for Numerical Variables

#### Problem
Understanding relationships between numerical variables can help identify redundant features and create meaningful combinations.

#### Recommended Approaches

1. **Correlation-Based Feature Selection**:
   - Identify highly correlated feature pairs
   - Remove redundant features or combine them
   - Create ratio features from correlated variables

2. **Principal Component Analysis (PCA)**:
   - Apply PCA to groups of correlated numerical features
   - Retain components that explain significant variance
   - Use component scores as new features

3. **Interaction Features**:
   - Create multiplication/division features from correlated variables
   - Create polynomial features for non-linear relationships

#### Implementation Strategy
```
# Pseudocode for implementation
# Calculate correlation matrix
numerical_cols = ['brand_id', 'price', 'socdem_cluster', 'region']
correlation_matrix = df[numerical_cols].corr()

# Identify highly correlated pairs (|r| > 0.7)
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.7:
            high_corr_pairs.append((
                correlation_matrix.columns[i], 
                correlation_matrix.columns[j], 
                correlation_matrix.iloc[i, j]
            ))

# Create interaction features
df['brand_price_interaction'] = df['brand_id'] * df['price']
df['socdem_region_interaction'] = df['socdem_cluster'] * df['region']
```

## Implementation Roadmap

### Phase 1: Data Quality Improvements
1. Handle missing values in category/subcategory columns
2. Process embedding data and extract features
3. Address price outliers and create price-related features

### Phase 2: Temporal Feature Creation
1. Extract time-based features from event timestamps
2. Create user behavior temporal patterns
3. Develop item popularity over time features

### Phase 3: Categorical Encoding and Combinations
1. Implement appropriate encoding strategies for categorical variables
2. Create combination features
3. Apply target encoding where appropriate

### Phase 4: Numerical Feature Engineering
1. Perform correlation analysis
2. Create interaction and ratio features
3. Apply dimensionality reduction techniques

### Phase 5: Validation and Testing
1. Validate feature engineering approaches on a sample
2. Test feature importance
3. Iterate on feature creation based on model performance

## Expected Outcomes

1. **Improved Predictive Power**: New features should capture more nuanced patterns in user behavior and item characteristics
2. **Better Handling of Missing Data**: Systematic approaches to missing values should reduce information loss
3. **Enhanced Model Interpretability**: Well-engineered features should be more interpretable than raw variables
4. **Robustness to Outliers**: Proper treatment of outliers should improve model stability

## Success Metrics

1. Improvement in baseline model performance (MAE, RMSE, R²)
2. Reduction in overfitting
3. Feature importance scores for new features
4. Model interpretability improvements

## Risks and Mitigations

1. **Over-engineering**: Regular validation on holdout sets to ensure features generalize
2. **Computational Complexity**: Feature engineering pipelines should be optimized for performance
3. **Data Leakage**: Careful temporal splits to prevent future information from leaking into training
4. **Feature Redundancy**: Correlation analysis to identify and remove redundant features

## Next Steps

1. Implement Phase 1 features and validate on a small sample
2. Evaluate impact on baseline model performance
3. Iterate on feature engineering approaches based on results
4. Scale implementation to full dataset