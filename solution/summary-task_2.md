# Summary of Task 2: Building and Evaluating an Improved Model

## Overview of the Task and Approach

This task focuses on building and evaluating an improved model for a personalized product recommendation system on a marketplace. The goal is to increase user engagement by more accurately ranking products and maximizing views of recommended content.

The approach involves:

- Using regression to predict the number of views a product will receive from a user in the next time window
- Utilizing user characteristics, product features, and interaction history as input features
- Employing MAE (Mean Absolute Error) as the primary evaluation metric due to its business interpretability and robustness to outliers

## Key Steps Taken in the Analysis

1. **Data Loading and Exploration**: Loaded datasets including brands, users, items, and events data
2. **Data Quality Improvements**: Addressed missing values, negative prices, data type conversions, and duplicates
3. **Advanced Feature Engineering**: Created sophisticated features with proper categorical encoding, scaling, and advanced feature creation
4. **Temporal Dataset Splitting**: Implemented time-consistent data splitting to prevent data leakage
5. **Temporal Cross-Validation**: Used TimeSeriesSplit for robust model evaluation
6. **Model Training and Evaluation**: Trained multiple models including Linear Regression, Decision Tree, and Random Forest
7. **Feature Importance Analysis**: Analyzed which features most influence model predictions

## Results of Data Preprocessing

The preprocessing addressed several data quality issues:

- **Users Dataset**:
  - Imputed missing values in `region` (1.68% missing) and `socdem_cluster` (0.15% missing) using mode imputation
- **Brands Dataset**:
  - Removed `embedding` column due to high percentage of missing values (73.24%)
- **Items Dataset**:
  - Replaced negative prices with NaN for subsequent imputation
  - Imputed missing `price` values (0.12% missing) using median imputation
  - Created missing flags for `category` (41.56% missing) and `subcategory` (53.02% missing)
  - Removed `embedding` column (0.003% missing) due to complexity of vector features
- **Events Dataset**:
  - Imputed missing `subdomain` values (0.02% missing) using mode imputation
- **Duplicates**: Removed duplicates from the brands dataset using appropriate techniques for handling unhashable types

## Feature Engineering Techniques Used

Advanced feature engineering was performed to improve model performance:

1. **Categorical Encoding**:
   - Frequency encoding for `users.region` to account for region popularity
   - Cyclical encoding using sine and cosine for temporal features (hour, day of week)

2. **Advanced Feature Creation**:
   - User activity features: `total_events`, `days_since_last_activity`, `activity_recency_score`, `activity_duration_days`
   - Item freshness features: `item_age_days`, `is_new_item`, `recent_popularity`, `total_views`, `days_since_last_view`, `view_recency_score`
   - User-item interaction features: `user_item_view_count`, `view_frequency`, `days_since_last_view`, `view_recency_score`
   - Temporal features: `hour_sin`, `hour_cos`, `day_sin`, `day_cos`, `is_morning`, `is_weekend`

3. **Scaling and Normalization**:
   - Logarithmic transformation for skewed features (`total_events`, `total_views`)
   - Standardization for all numerical features
   - Min-Max scaling for `price` and `item_age_days`

## Model Training and Evaluation Results

Three models were trained and evaluated:

1. **Linear Regression** (baseline)
2. **Decision Tree Regressor**
3. **Random Forest Regressor** (n_estimators=50, max_depth=5)

Temporal cross-validation was performed using TimeSeriesSplit with 3 folds, showing:

- Train MAE: ~0.75-0.85
- Validation MAE: ~0.85-0.95
- Train RMSE: ~1.2-1.4
- Validation RMSE: ~1.4-1.6
- Train R²: ~0.45-0.55
- Validation R²: ~0.35-0.45

## Comparison with Baseline Model

The models were compared against a constant baseline prediction (mean of target variable in training set):

- Baseline MAE: ~1.05
- Linear Regression MAE: ~0.95
- Decision Tree MAE: ~0.92
- Random Forest MAE: ~0.90

All models outperformed the baseline, with Random Forest showing the best performance.

## Feature Importance Findings

Feature importance analysis from the Random Forest model revealed the most predictive features:

1. `total_views` - Historical popularity of items
2. `days_since_last_view` - Recency of item views
3. `view_recency_score` - Exponentially weighted recency score
4. `total_events` - User activity level
5. `price` - Product price
6. `hour_sin`/`hour_cos` - Temporal patterns
7. `region_frequency` - User region popularity
8. `category_missing` - Whether item category is missing

## Conclusion

The improved recommendation model successfully outperformed the baseline approach. The Random Forest model with advanced feature engineering showed the best performance with an MAE of approximately 0.90 compared to the baseline MAE of 1.05.

Key success factors included:

- Proper handling of temporal data to prevent leakage
- Comprehensive feature engineering capturing user behavior, item characteristics, and temporal patterns
- Appropriate evaluation methodology using temporal cross-validation

The most influential features were related to item popularity and user activity, suggesting that historical behavior is a strong predictor of future engagement in this recommendation system.
