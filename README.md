# ML Intro ITMO: E-Commerce Recommendation System

#### [Task](https://github.com/pacifikus/itmo_ml_for_science_course/blob/main/readme.md)

## Project Overview

### Business Problem

This project aims to build a recommendation system for e-commerce platforms to increase sales and improve user experience by providing personalized product recommendations. The system focuses on maximizing user engagement through accurate ranking of products based on predicted views.

### ML Problem Formulation

The machine learning task is formulated as a regression problem to predict the number of views a product will receive from a user in the next time window. This approach enables ranking products by their predicted engagement, with higher predicted views receiving priority in recommendations.

## Tasks Completed

### Task 1: EDA and Problem Formulation

- Comprehensive exploratory data analysis (EDA) of the T-ECD dataset
- Business problem identification and ML problem formulation
- Dataset quality assessment and identification of data issues
- Feature selection and metric justification (MAE as primary metric)

### Task 2: Baseline Model Implementation

- Data preprocessing and quality improvements for missing values, duplicates, and outliers
- Advanced feature engineering with categorical encoding, scaling, and temporal features
- Implementation of baseline models (Linear Regression, Decision Tree, Random Forest)
- Temporal cross-validation to prevent data leakage
- Model evaluation and comparison with feature importance analysis

### Task 3: Advanced Ensemble Models

- Implementation of advanced ensemble models (XGBoost, LightGBM, CatBoost)
- Hyperparameter optimization with Optuna to minimize MAE
- Comprehensive model interpretation using SHAP (global) and LIME (local)
- Final model comparison and performance analysis

## Setup Instructions

1. **Create a virtual environment:**

   ```bash
   python3 -m venv venv
   ```

2. **Activate the virtual environment:**
   - On macOS/Linux:

     ```bash
     source venv/bin/activate
     ```

   - On Windows:

     ```bash
     venv\Scripts\activate
     ```

3. **Install dependencies:**

   ```bash
   pip3 install -r requirements.txt
   ```

4. **Set up environment variables:**
   Create a `.env` file with example of `.env.example`

5. **Download the dataset:**

   ```bash
   python3 scripts/download_dataset.py
   ```

6. **macOS users - Install libomp for XGBoost (if needed):**
   If you're on macOS and encounter an import error with XGBoost related to libomp, you'll need to install the OpenMP library:

   ```bash
   brew install libomp
   ```

   This is a dependency required by XGBoost on macOS systems. If you don't have Homebrew installed, you'll need to install it first by following the instructions at [brew.sh](https://brew.sh).

## Project Structure

- `scripts/` - Contains Python scripts for data downloading and processing
- `t_ecd_data/` - Directory where the dataset will be downloaded
- `.env` - Environment variables file (not included in repo)
- `requirements.txt` - Python dependencies
- `eda_visualizations/` - Directory containing EDA results and visualizations
- `EDA.md` - Comprehensive EDA report in Russian
- `solution/` - Contains Jupyter notebooks and summary reports for all tasks

## Dataset Information

The project uses the T-ECD (Temporal E-Commerce Dataset) from Hugging Face. You can choose between:

- Small version: `dataset/small/`
- Full version: `dataset/full/` (uncomment in the download script)

### Dataset Components

The T-ECD dataset consists of four main components:

1. **Users Dataset** (3.5M rows):
   - `user_id`: Unique identifier for each user
   - `socdem_cluster`: Socio-demographic cluster of the user
   - `region`: Geographic region of the user

2. **Items Dataset** (2.3M rows):
   - `item_id`: Unique identifier for each item
   - `brand_id`: Brand identifier
   - `category`: Product category
   - `subcategory`: Product subcategory
   - `price`: Price of the item
   - `embedding`: Vector representation of the item

3. **Events Dataset** (6.6M rows):
   - `timestamp`: Time of the event
   - `user_id`: User who performed the action
   - `item_id`: Item involved in the event
   - `subdomain`: Subdomain where the event occurred
   - `action_type`: Type of user action (view, click, cart, purchase)
   - `os`: Operating system used

4. **Brands Dataset** (24.5K rows):
   - `brand_id`: Unique identifier for each brand
   - `embedding`: Vector representation of the brand

### Data Quality Issues Addressed

- High percentage of missing values in `items.category` (41.6%) and `items.subcategory` (53.0%)
- Missing values in `brands.embedding` (73.2%)
- Missing values in `users.region` (1.68%) and `users.socdem_cluster` (0.15%)
- Negative price values in items dataset
- Duplicate records in brands dataset

## Feature Engineering

Comprehensive feature engineering was performed to capture user behavior patterns, item characteristics, and temporal trends:

### User Activity Features

- `total_events`: Total number of events per user
- `days_since_last_activity`: Time since user's last activity
- `activity_recency_score`: Exponentially weighted recency of user activity
- `activity_duration_days`: Duration of user's activity period

### Item Freshness Features

- `item_age_days`: Age of the item in days
- `is_new_item`: Boolean flag for recently added items
- `recent_popularity`: Popularity of item in recent time window
- `total_views`: Total number of views for the item
- `days_since_last_view`: Time since item was last viewed
- `view_recency_score`: Exponentially weighted recency of item views

### User-Item Interaction Features

- `user_item_view_count`: Number of times user viewed the item
- `view_frequency`: Frequency of user-item interactions
- `days_since_last_view`: Time since user last viewed the item
- `view_recency_score`: Exponentially weighted recency of user-item interactions

### Temporal Features

- `hour_sin`/`hour_cos`: Cyclical encoding of hour of day
- `day_sin`/`day_cos`: Cyclical encoding of day of week
- `is_morning`: Boolean flag for morning hours
- `is_weekend`: Boolean flag for weekend days

### Preprocessing Techniques

- Frequency encoding for `users.region` to account for region popularity
- Logarithmic transformation for skewed features (`total_events`, `total_views`)
- Standardization for numerical features
- Min-Max scaling for `price` and `item_age_days`
- Missing value flags for categorical features with high missing rates

## Models Implemented

### Baseline Models

1. **Linear Regression** - Simple linear model for baseline performance
2. **Decision Tree Regressor** - Non-linear baseline model
3. **Random Forest Regressor** - Ensemble of decision trees (n_estimators=50, max_depth=5)

### Advanced Ensemble Models

1. **XGBoost**:
   - Default: n_estimators=100, max_depth=6, learning_rate=0.1
   - Optimized: n_estimators=214, max_depth=5, learning_rate=0.141, subsample=0.847, colsample_bytree=0.723, gamma=0.729, reg_alpha=2.104, reg_lambda=0.001

2. **LightGBM**: n_estimators=1000, max_depth=10, learning_rate=0.05 with additional regularization parameters

3. **CatBoost**: n_estimators=100, max_depth=6, learning_rate=0.1

## Results

All models were evaluated using MAE (Mean Absolute Error) as the primary metric, with RMSE and R² as secondary metrics:

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Linear Regression (Baseline) | 0.947 | 1.389 | 0.356 |
| Decision Tree | ~0.92 | ~1.3-1.4 | ~0.45-0.55 |
| Random Forest | ~0.90 | ~1.3-1.4 | ~0.45-0.55 |
| XGBoost (Default) | 0.854 | 1.256 | 0.478 |
| CatBoost | 0.841 | 1.238 | 0.489 |
| LightGBM | 0.832 | 1.224 | 0.501 |
| XGBoost (Optimized) | **0.823** | 1.204 | 0.512 |

The optimized XGBoost model achieved the best performance with an MAE of 0.823, representing a 9.9% improvement over the Random Forest baseline from Task 2.

## Model Interpretation

### Global Interpretation (SHAP)

Global interpretation using SHAP values revealed the most important features for the optimized XGBoost model:

1. `total_events` - User activity level
2. `total_views` - Historical popularity of items
3. `days_since_last_view` - Recency of item views
4. `view_recency_score` - Exponentially weighted recency score
5. `hour_sin`/`hour_cos` - Temporal patterns (hour of day)
6. `day_sin`/`day_cos` - Temporal patterns (day of week)
7. `price` - Product price
8. `region_frequency` - User region popularity
9. `category_missing` - Whether item category is missing
10. `subcategory_missing` - Whether item subcategory is missing

### Local Interpretation (LIME)

Local interpretation using LIME showed that feature importance varies across individual predictions:

- For some users, recent activity with an item was the most important factor
- For others, overall user activity level was more influential
- Temporal features had varying importance across different predictions
- Price and item popularity features showed variable importance depending on the specific user-item pair

## Technical Details

### Temporal Cross-Validation

To prevent data leakage, temporal cross-validation was implemented using TimeSeriesSplit with 3 folds. This ensures that training data always precedes validation data chronologically.

### Hyperparameter Optimization

Optuna was used for hyperparameter optimization of the XGBoost model with 30 trials. The optimization objective was to minimize MAE on the validation set. The parameter space included:

- n_estimators: 50-300
- max_depth: 3-10
- learning_rate: 0.01-0.3
- subsample: 0.5-1.0
- colsample_bytree: 0.5-1.0
- gamma: 0-10
- reg_alpha: 0-5
- reg_lambda: 0-5

### Data Handling

- Missing values were handled using appropriate imputation strategies (mode for categorical, median for numerical)
- Duplicate records were removed from the brands dataset
- Negative price values were replaced with NaN for subsequent imputation
- Complex vector features (`embedding` columns) were removed due to processing complexity
- Cyclical encoding was used for temporal features to preserve the circular nature of time

## EDA Analysis

The repository includes an exploratory data analysis (EDA) of the T-ECD dataset:

- **Script**: [`scripts/eda_analysis.py`](scripts/eda_analysis.py) - A comprehensive Python script that performs EDA including data loading, basic statistics, data quality checks, and data visualization
- **Report**: [`EDA.md`](EDA.md) - A detailed report in Russian documenting the findings of the EDA with all generated visualizations
