# Task 2 Plan: Building and Evaluating a Baseline Model

## 1. Task Requirements Summary

Based on the requirements in `tasks/2.md`, the main objectives for Task 2 are:

1. Build and evaluate a baseline model
2. Split the dataset into training and test sets (2 points)
3. Measure the quality of constant prediction (e.g., most frequent class for classification, mean/median for regression) (3 points)
4. Train a baseline model from a simple family (linear models, decision trees, KNN) on the training set, considering data preprocessing specifics (6 points)
5. Measure quality on the holdout set using the previously selected metric (2 points)
6. Ensure solution reproducibility: fix `random_state`, ensure the notebook runs from start to finish without errors (3 points)
7. Follow code style guidelines at the [PEP8](https://peps.python.org/pep-0008/) level and [On writing clean Jupyter notebooks](https://ploomber.io/blog/clean-nbs/) (4 points)
8. Justify decisions and comment on them in markdown cells (10 points)

## 2. References to EDA Findings

From the EDA analysis in `EDA.md` and `.doc/eda_results.md`, we have the following key findings:

### Dataset Overview
- **Brands**: 24,513 rows, 2 columns
- **Users**: 3,500,000 rows, 3 columns
- **Items**: 2,325,409 rows, 6 columns
- **Events**: 6,627,693 rows, 6 columns

### ML Task Formulation
- **Business Problem**: Implement a personalized product recommendation system on a marketplace to increase user engagement through more accurate product ranking and maximizing views of recommended content
- **ML Problem**: Regression task - predict the number of views a product will receive from a user in the next time window based on user features, product features, and interaction history

### Data Quality Issues
1. **Missing Values**:
   - Brands dataset: embedding - 17,954 missing values (73.24%)
   - Users dataset: region - 58,917 missing values (1.68%), socdem_cluster - 5,153 missing values (0.15%)
   - Items dataset: subcategory - 1,233,023 missing values (53.02%), category - 966,395 missing values (41.56%), price - 2,882 missing values (0.12%)
   - Events dataset: subdomain - 1,453 missing values (0.02%)

2. **Duplicates**: 46 duplicates (0.19%) found in brands dataset

3. **Outliers**: Several outliers identified, especially in items.price

### Feature Insights
- Items.price shows skewness (mean=0.49, std=2.46) and contains negative values
- Users.region is multimodal
- Brands.brand_id has a normal distribution
- Event data spans approximately 10 days (from day 1250 to 1259)
- Diverse categories of action_type and subdomain observed

### Temporal Patterns
- Events show temporal patterns with variations by action type
- No clear seasonal patterns observed due to limited time span

## 3. References to Task 1 Solution

Based on the solution approach outlined in `.doc/init.md`, Task 1 established:

### Problem Formulation
- **Business Problem**: Build a recommendation system for e-commerce to increase sales and improve user experience through relevant product recommendations
- **ML Problem**: Binary classification - for a user-item pair, predict whether the user will purchase the item within the next 24 hours

### Metrics Selection
- **ROC-AUC**: Suitable for evaluating binary classification accuracy on imbalanced datasets
- **Precision@K and Recall@K**: Relevant for recommendation systems as they evaluate the relevance of top-K recommendations, which relates to business impact

### EDA Approach
- Load data (preferably a small subset like T-ECD small)
- Calculate basic statistics
- Create visualizations using seaborn and plotly
- Analyze data quality issues and propose preprocessing steps

## 4. Detailed Steps for Baseline Model Implementation

### Step 1: Data Preprocessing
1. Handle missing values:
   - For users.region (1.68% missing): Consider imputation using mode or a separate "unknown" category
   - For users.socdem_cluster (0.15% missing): Consider dropping records or imputing with mode
   - For items.category and items.subcategory (41.6% and 53.0% missing respectively): Priority processing required
   - For items.price (0.12% missing): Consider dropping records or imputing with median
   - For brands.embedding (73.24% missing): Priority processing required
   - For events.subdomain (0.02% missing): Consider dropping records or imputing with "unknown" category

2. Handle duplicates in brands dataset

3. Handle outliers, especially in items.price:
   - Consider applying transformations (e.g., log transformation) to skewed numerical variables

4. Merge datasets:
   - Merge users and events datasets on user_id
   - Merge with items dataset on item_id
   - Create a unified dataset for modeling

### Step 2: Feature Engineering
1. Extract temporal features from timestamp:
   - Hour of day
   - Day of week
   - Time since last interaction

2. Encode categorical variables:
   - action_type, os, subdomain (events)
   - category, subcategory (items)
   - Consider appropriate techniques (one-hot, label encoding, or embedding)

3. Create aggregated user behavior features:
   - Number of interactions per user
   - Number of views, clicks, cart additions, purchases per user
   - Average time between interactions

4. Create item-based features:
   - Popularity of items (number of views)
   - Average price in category

### Step 3: Dataset Splitting
1. Split data into training and test sets:
   - Use an appropriate ratio (e.g., 80/20)
   - Ensure temporal consistency in split (use earlier data for training, later for testing)
   - Set random_state for reproducibility

### Step 4: Constant Prediction Baseline
1. Calculate mean/median of target variable (number of views) for regression
2. Use this constant value as a baseline prediction
3. Evaluate using MAE, RMSE, and R²

### Step 5: Model Selection and Training
1. Select a simple baseline model:
   - Linear regression (as a simple baseline)
   - Decision tree regressor (as an alternative)

2. Train the model on the training set:
   - Apply the same preprocessing to training data
   - Handle any model-specific preprocessing requirements

### Step 6: Model Evaluation
1. Evaluate on the test set using the selected metrics:
   - Primary: MAE (Mean Absolute Error) as justified in EDA
   - Secondary: RMSE and R² for additional insights

2. Compare performance against the constant prediction baseline

## 5. Model Evaluation Approach

### Primary Metric
- **MAE (Mean Absolute Error)**: Chosen because it's easily interpretable in business context (average error in views), robust to outliers (active users don't distort the overall assessment), and evaluates errors evenly across all products, which is important for ranking

### Secondary Metrics
- **RMSE (Root Mean Square Error)**: To track stability and penalize larger errors
- **R² (Coefficient of Determination)**: To measure the explanatory power of the model

### Evaluation Process
1. Calculate metrics for constant prediction baseline
2. Calculate metrics for trained baseline model
3. Compare results to assess model improvement
4. Document findings in markdown cells with justification

## 6. Code Style and Reproducibility Considerations

### Code Style
1. Follow [PEP8](https://peps.python.org/pep-0008/) guidelines:
   - Use 4 spaces per indentation level
   - Limit lines to 79 characters
   - Use meaningful variable and function names
   - Use blank lines to separate functions and class definitions
   - Use spaces around operators and after commas

2. Follow [clean Jupyter notebook practices](https://ploomber.io/blog/clean-nbs/):
   - Use markdown cells to explain the purpose of code sections
   - Keep code cells focused on single tasks
   - Use meaningful cell execution order
   - Include comments for complex code sections
   - Use consistent naming conventions

### Reproducibility
1. Set random_state for all random operations:
   - Data splitting
   - Model initialization
   - Any random sampling

2. Document all steps clearly:
   - Include data loading and preprocessing steps
   - Document feature engineering decisions
   - Explain model selection rationale

3. Ensure notebook runs from start to finish without errors:
   - Test the entire notebook execution
   - Include error handling where appropriate
   - Document any dependencies or setup requirements

### Documentation
1. Justify all decisions in markdown cells:
   - Explain preprocessing choices
   - Justify feature engineering decisions
   - Explain model selection rationale
   - Interpret evaluation results

2. Include explanations for:
   - Why specific missing value treatment methods were chosen
   - Why certain categorical encoding techniques were used
   - Why specific models were selected as baselines
   - How evaluation metrics relate to business objectives

## 7. Implementation Plan for Jupyter Notebook

### Notebook Structure
1. **Introduction**
   - Business and ML problem statement
   - Overview of approach

2. **Data Loading and Exploration**
   - Load datasets
   - Display basic information about datasets
   - Show sample data

3. **Data Preprocessing**
   - Handle missing values
   - Handle duplicates
   - Handle outliers
   - Merge datasets
   - Justify all preprocessing decisions

4. **Feature Engineering**
   - Extract temporal features
   - Encode categorical variables
   - Create aggregated features
   - Justify feature engineering choices

5. **Dataset Splitting**
   - Split into train/test sets
   - Ensure temporal consistency
   - Set random_state for reproducibility

6. **Constant Prediction Baseline**
   - Calculate mean/median of target
   - Evaluate using selected metrics
   - Document results

7. **Baseline Model Implementation**
   - Select and justify model choice
   - Train model on training set
   - Apply same preprocessing to training data

8. **Model Evaluation**
   - Evaluate on test set
   - Compare with constant baseline
   - Document and interpret results

9. **Conclusion**
   - Summarize findings
   - Discuss model performance
   - Suggest next steps

### Best Practices for Implementation
1. Use modular code with functions for reusable components
2. Include error handling for data loading and processing
3. Use clear variable names that reflect their purpose
4. Include progress indicators for long-running operations
5. Document assumptions and limitations
6. Use version control for tracking changes

This comprehensive plan provides a roadmap for implementing the baseline model for Task 2, incorporating insights from the EDA and building on the foundation established in Task 1.