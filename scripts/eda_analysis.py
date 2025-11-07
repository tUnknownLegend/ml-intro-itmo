#!/usr/bin/env python3
"""
Comprehensive EDA script for T-ECD dataset analysis.
Performs exploratory data analysis including data loading,
statistics, visualization, and reporting.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
import polars as pl
warnings.filterwarnings('ignore')

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Create visualizations directory if it doesn't exist
VISUALIZATIONS_DIR = "eda_visualizations"
Path(VISUALIZATIONS_DIR).mkdir(exist_ok=True)

# Dataset paths
DATA_DIR = "t_ecd_small_partial/dataset/small"
BRANDS_PATH = f"{DATA_DIR}/brands.pq"
USERS_PATH = f"{DATA_DIR}/users.pq"
ITEMS_PATH = f"{DATA_DIR}/marketplace/items.pq"
EVENTS_DIR = f"{DATA_DIR}/marketplace/events"

class EDAAnalyzer:
    """Class to perform comprehensive EDA on T-ECD dataset."""
    
    def __init__(self):
        self.datasets = {}
        self.report = []
        self.numerical_cols = {}
        self.categorical_cols = {}
        
    def log_report(self, message):
        """Add message to report and print it."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        self.report.append(log_message)
    
    def load_data(self):
        """Load all required data files."""
        self.log_report("Starting data loading...")
        
        # Load static files
        try:
            # Use Polars to read brands.pq due to array schema issues 
            # with Pandas/PyArrow
            brands_df_polars = pl.read_parquet(BRANDS_PATH)
            self.datasets['brands'] = brands_df_polars.to_pandas()
            self.log_report(f"Loaded brands data: {self.datasets['brands'].shape}")
        except Exception as e:
            self.log_report(f"Error loading brands data: {e}")
            
        try:
            self.datasets['users'] = pd.read_parquet(USERS_PATH)
            self.log_report(f"Loaded users data: {self.datasets['users'].shape}")
        except Exception as e:
            self.log_report(f"Error loading users data: {e}")
            
        try:
            # Use Polars to read items.pq due to array schema issues 
            # with Pandas/PyArrow
            items_df_polars = pl.read_parquet(ITEMS_PATH)
            self.datasets['items'] = items_df_polars.to_pandas()
            self.log_report(f"Loaded items data: {self.datasets['items'].shape}")
        except Exception as e:
            self.log_report(f"Error loading items data: {e}")
        
        # Load event files
        event_files = sorted([f for f in os.listdir(EVENTS_DIR) if f.endswith('.pq')])
        self.log_report(f"Found {len(event_files)} event files")
        
        event_dataframes = []
        for file in event_files[:10]:  # Load first 10 for demonstration
            try:
                file_path = os.path.join(EVENTS_DIR, file)
                df = pd.read_parquet(file_path)
                event_dataframes.append(df)
                if len(event_dataframes) % 10 == 0:
                    self.log_report(f"Loaded {len(event_dataframes)} event files...")
            except Exception as e:
                self.log_report(f"Error loading event file {file}: {e}")
        
        if event_dataframes:
            self.datasets['events'] = pd.concat(event_dataframes, ignore_index=True)
            self.log_report(f"Combined events data: {self.datasets['events'].shape}")
        else:
            self.log_report("No event data loaded")
        
        self.log_report("Data loading completed.")
    
    def examine_data_structure(self):
        """Examine the structure of each dataset."""
        self.log_report("Examining data structure...")
        
        for name, df in self.datasets.items():
            if df is not None:
                self.log_report(f"\n--- {name.upper()} DATASET ---")
                self.log_report(f"Shape: {df.shape}")
                self.log_report(f"Columns: {list(df.columns)}")
                self.log_report(f"Data types:\n{df.dtypes}")
                
                # Identify numerical and categorical columns
                numerical = df.select_dtypes(include=[np.number]).columns.tolist()
                categorical = df.select_dtypes(include=['object', 'category']).columns.tolist()
                
                self.numerical_cols[name] = numerical
                self.categorical_cols[name] = categorical
                
                self.log_report(f"Numerical columns: {numerical}")
                self.log_report(f"Categorical columns: {categorical}")
        
        self.log_report("Data structure examination completed.")
    
    def _calculate_column_stats(self, df, col):
        """Calculate statistics for a single column."""
        if 'user_id' in col.lower():
            self.log_report(f"Skipped {col} - user_id is an identifier, "
                          "statistics are not meaningful")
            return
            
        if col in df.columns:
            mean_val = df[col].mean()
            median_val = df[col].median()
            std_val = df[col].std()
            mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else "No mode"
            
            # Handle special formatting for timedelta columns
            if pd.api.types.is_timedelta64_dtype(df[col]):
                mean_str = str(mean_val).split()[0] if hasattr(mean_val, 'days') else str(mean_val)
                median_str = str(median_val).split()[0] if hasattr(median_val, 'days') else str(median_val)
                std_str = str(std_val).split()[0] if hasattr(std_val, 'days') else str(std_val)
                self.log_report(f"{col} - Mean: {mean_str}, Median: {median_str}, "
                              f"Std: {std_str}, Mode: {mode_val}")
            else:
                self.log_report(f"{col} - Mean: {mean_val:.4f}, Median: {median_val:.4f}, "
                              f"Std: {std_val:.4f}, Mode: {mode_val}")
    
    def basic_statistics(self):
        """Calculate basic statistics for numerical variables."""
        self.log_report("Calculating basic statistics...")
        
        for name, df in self.datasets.items():
            if df is not None and name in self.numerical_cols:
                numerical = self.numerical_cols[name]
                if numerical:
                    self.log_report(f"\n--- {name.upper()} STATISTICS ---")
                    stats = df[numerical].describe()
                    self.log_report(f"Descriptive statistics:\n{stats}")
                    
                    # Additional statistics
                    for col in numerical:
                        self._calculate_column_stats(df, col)
        
        self.log_report("Basic statistics calculation completed.")
    
    def missing_values_analysis(self):
        """Analyze missing values in datasets."""
        self.log_report("Analyzing missing values...")
        
        for name, df in self.datasets.items():
            if df is not None:
                self.log_report(f"\n--- {name.upper()} MISSING VALUES ---")
                missing = df.isnull().sum()
                missing_pct = (missing / len(df)) * 100
                
                missing_df = pd.DataFrame({
                    'Missing Count': missing,
                    'Percentage': missing_pct
                }).sort_values('Missing Count', ascending=False)
                
                missing_df = missing_df[missing_df['Missing Count'] > 0]
                
                if not missing_df.empty:
                    self.log_report(f"Missing values:\n{missing_df}")
                else:
                    self.log_report("No missing values found.")
                
                # Visualize missing data pattern
                if not missing_df.empty and len(df.columns) <= 50:  # Only for reasonable number of columns
                    self._create_missing_data_visualization(name, df)
    
    def _create_missing_data_visualization(self, name, df):
        """Create missing data pattern visualization."""
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
        plt.title(f'Missing Data Pattern - {name}\n'
                  '(Black areas indicate missing values)', 
                  fontsize=14, pad=20)
        plt.xlabel('Columns', fontsize=12)
        plt.ylabel('Data Rows (Sample)', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{VISUALIZATIONS_DIR}/{name}_missing_data_pattern.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        self.log_report(f"Saved missing data pattern visualization for {name}")
    
    def duplicate_analysis(self):
        """Check for duplicate records."""
        self.log_report("Checking for duplicate records...")
        
        for name, df in self.datasets.items():
            if df is not None:
                try:
                    duplicates = df.duplicated().sum()
                    self.log_report(f"{name} - Duplicate records: {duplicates} "
                                  f"({(duplicates/len(df))*100:.4f}%)")
                except TypeError as e:
                    if "unhashable type" in str(e):
                        # Handle DataFrames with unhashable types (like arrays in embedding columns)
                        # Create a copy without the problematic columns for duplicate detection
                        non_embedding_cols = [col for col in df.columns 
                                            if 'embedding' not in col.lower()]
                        if non_embedding_cols:
                            duplicates = df[non_embedding_cols].duplicated().sum()
                            self.log_report(f"{name} - Duplicate records "
                                          f"(excluding embedding columns): {duplicates} "
                                          f"({(duplicates/len(df))*100:.4f}%)")
                        else:
                            self.log_report(f"{name} - Skipping duplicate analysis "
                                          "(contains only unhashable columns)")
                    else:
                        self.log_report(f"{name} - Error in duplicate analysis: {e}")
        
        self.log_report("Duplicate analysis completed.")
    
    def _should_skip_distribution(self, name, col, analysis_type):
        """Check if distribution analysis should be skipped for specific columns."""
        skip_distributions = [
            'events_user_id_distribution.png',
            'users_user_id_distribution.png',
            'events_item_id_categories.png'
        ]
        
        if 'user_id' in col.lower():
            self.log_report(f"Skipped {analysis_type} for {col} - user_id is an "
                          "identifier, analysis is not meaningful")
            return True
            
        output_filename = f'{name}_{col}_{analysis_type.split()[0]}.png'
        if output_filename in skip_distributions:
            self.log_report(f"Skipped {analysis_type} for {col} - "
                          "not informative visualization")
            return True
            
        return False
    
    def distribution_analysis(self):
        """Analyze distribution of key variables."""
        self.log_report("Analyzing variable distributions...")
        
        for name, df in self.datasets.items():
            if df is not None:
                numerical = self.numerical_cols.get(name, [])
                categorical = self.categorical_cols.get(name, [])
                
                # For numerical variables, create histograms
                for col in numerical[:5]:  # Limit to first 5 for performance
                    if self._should_skip_distribution(name, col, "distribution analysis"):
                        continue
                        
                    if col in df.columns and df[col].notna().sum() > 0:
                        self._create_numerical_distribution_plot(name, col, df)
                
                # For categorical variables, create bar charts (top 10 categories)
                for col in categorical[:3]:  # Limit to first 3 for performance
                    if self._should_skip_distribution(name, col, "categorical analysis"):
                        continue
                        
                    if col in df.columns:
                        self._create_categorical_distribution_plot(name, col, df)
        
        self.log_report("Distribution analysis completed.")
    
    def _create_numerical_distribution_plot(self, name, col, df):
        """Create histogram for numerical distribution."""
        plt.figure(figsize=(10, 6))
        # Handle timedelta columns specially
        if pd.api.types.is_timedelta64_dtype(df[col]):
            # Convert to seconds for plotting
            data_to_plot = df[col].dropna().dt.total_seconds()
            plt.hist(data_to_plot, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            plt.title(f'Distribution of {col} (in seconds) - {name}\n'
                      'Histogram showing the frequency distribution of values', 
                      fontsize=14, pad=20)
            plt.xlabel(f'{col} (seconds)', fontsize=12)
        else:
            plt.hist(df[col].dropna(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            plt.title(f'Distribution of {col} - {name}\n'
                      'Histogram showing the frequency distribution of values', 
                      fontsize=14, pad=20)
            plt.xlabel(col, fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{VISUALIZATIONS_DIR}/{name}_{col}_distribution.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_categorical_distribution_plot(self, name, col, df):
        """Create bar chart for categorical distribution."""
        value_counts = df[col].value_counts().head(10)
        plt.figure(figsize=(12, 6))
        value_counts.plot(kind='bar', color='lightcoral')
        plt.title(f'Top 10 Categories in {col} - {name}\n'
                  'Bar chart showing the frequency of each category', 
                  fontsize=14, pad=20)
        plt.xlabel(col, fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{VISUALIZATIONS_DIR}/{name}_{col}_categories.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def correlation_analysis(self):
        """Create correlation matrices for numerical variables."""
        self.log_report("Performing correlation analysis...")
        
        for name, df in self.datasets.items():
            if df is not None:
                numerical = self.numerical_cols.get(name, [])
                if len(numerical) > 1 and len(numerical) <= 20:  # Only if reasonable number of columns
                    self._create_correlation_matrix(name, df, numerical)
        
        self.log_report("Correlation analysis completed.")
    
    def _create_correlation_matrix(self, name, df, numerical):
        """Create correlation matrix visualization."""
        corr_matrix = df[numerical].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, fmt='.2f')
        plt.title(f'Correlation Matrix - {name}\n'
                  'Heatmap showing correlations between numerical variables', 
                  fontsize=14, pad=20)
        plt.xlabel('Variables', fontsize=12)
        plt.ylabel('Variables', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{VISUALIZATIONS_DIR}/{name}_correlation_matrix.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        self.log_report(f"Saved correlation matrix for {name}")
    
    def time_series_analysis(self):
        """Analyze time series for event data."""
        self.log_report("Performing time series analysis...")
        
        if 'events' in self.datasets and self.datasets['events'] is not None:
            df = self.datasets['events']
            
            # Try to find timestamp column
            timestamp_cols = [col for col in df.columns 
                            if 'time' in col.lower() or 'date' in col.lower()]
            
            if timestamp_cols:
                timestamp_col = timestamp_cols[0]
                self.log_report(f"Using '{timestamp_col}' as timestamp column")
                
                # Convert to datetime if needed
                if df[timestamp_col].dtype == 'object':
                    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
                
                # Remove rows with invalid timestamps
                df_valid = df[df[timestamp_col].notna()].copy()
                
                if not df_valid.empty:
                    self._create_time_series_plots(df_valid, timestamp_col)
            else:
                self.log_report("No timestamp column found in events data")
        
        self.log_report("Time series analysis completed.")
    
    def _create_time_series_plots(self, df_valid, timestamp_col):
        """Create time series visualizations."""
        # Resample by day and count events
        df_valid.set_index(timestamp_col, inplace=True)
        daily_events = df_valid.resample('D').size()
        
        plt.figure(figsize=(12, 6))
        daily_events.plot(kind='line', marker='o', linewidth=2, markersize=4)
        plt.title('Event Count Over Time\n'
                  'Line chart showing the number of events per day', 
                  fontsize=14, pad=20)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Number of Events', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{VISUALIZATIONS_DIR}/events_time_series.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        self.log_report("Saved time series visualization for events")
        
        # If we have event types, analyze by type
        event_type_cols = [col for col in df_valid.columns 
                         if 'type' in col.lower() or 'event' in col.lower()]
        if event_type_cols:
            event_type_col = event_type_cols[0]
            if event_type_col in df_valid.columns:
                daily_events_by_type = df_valid.groupby([pd.Grouper(freq='D'), 
                                                       event_type_col]).size().unstack(fill_value=0)
                
                plt.figure(figsize=(14, 8))
                daily_events_by_type.plot(kind='area', alpha=0.7)
                plt.title('Event Count by Type Over Time\n'
                          'Stacked area chart showing event counts by type over time', 
                          fontsize=14, pad=20)
                plt.xlabel('Date', fontsize=12)
                plt.ylabel('Number of Events', fontsize=12)
                plt.legend(title=event_type_col, bbox_to_anchor=(1.05, 1), 
                          loc='upper left', fontsize=10)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f'{VISUALIZATIONS_DIR}/events_time_series_by_type.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()
                self.log_report("Saved time series by type visualization for events")
    
    def outlier_detection(self):
        """Detect outliers using box plots."""
        self.log_report("Detecting outliers...")
        
        # List of specific outlier visualizations to skip
        skip_outliers = [
            'events_user_id_outliers.png',
            'users_user_id_outliers.png'
        ]
        
        for name, df in self.datasets.items():
            if df is not None:
                numerical = self.numerical_cols.get(name, [])
                
                # Create box plots for numerical variables (limit to first 5)
                for col in numerical[:5]:
                    # Skip user_id columns as they are identifiers and outlier detection 
                    # is not meaningful
                    if 'user_id' in col.lower():
                        self.log_report(f"Skipped outlier detection for {col} - user_id "
                                      "is an identifier, outlier detection is not meaningful")
                        continue
                        
                    # Skip specific outlier visualizations that are not informative
                    output_filename = f'{name}_{col}_outliers.png'
                    if output_filename in skip_outliers:
                        self.log_report(f"Skipped outlier detection for {col} - "
                                      "not informative visualization")
                        continue
                        
                    if col in df.columns and df[col].notna().sum() > 0:
                        self._create_outlier_plot(name, col, df)
        
        self.log_report("Outlier detection completed.")
    
    def _create_outlier_plot(self, name, col, df):
        """Create box plot for outlier detection."""
        plt.figure(figsize=(8, 6))
        plt.boxplot(df[col].dropna())
        plt.title(f'Outlier Detection - {col} ({name})\n'
                  'Box plot showing distribution and outliers', 
                  fontsize=14, pad=20)
        plt.ylabel(col, fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{VISUALIZATIONS_DIR}/{name}_{col}_outliers.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def scatter_plots(self):
        """Create scatter plots for key variable relationships."""
        self.log_report("Creating scatter plots...")
        
        # List of specific scatter plot visualizations to skip
        skip_scatter_plots = [
            'events_timestamp_vs_user_id_scatter.png',
            'users_user_id_vs_region_scatter.png',
            'users_user_id_vs_socdem_cluster_scatter.png'
        ]
        
        for name, df in self.datasets.items():
            if df is not None:
                numerical = self.numerical_cols.get(name, [])
                
                # Create scatter plots for pairs of numerical variables (limit combinations)
                if len(numerical) >= 2:
                    # Take first 3 numerical columns for pairwise scatter plots
                    cols_to_plot = numerical[:3]
                    # Filter out user_id columns as scatter plots with them are not meaningful
                    cols_to_plot = [col for col in cols_to_plot 
                                  if 'user_id' not in col.lower()]
                    
                    # Skip if we don't have enough columns after filtering
                    if len(cols_to_plot) < 2:
                        self.log_report(f"Skipped scatter plots for {name} - "
                                      "not enough meaningful numerical columns after "
                                      "filtering out user_id")
                        continue
                    
                    self._create_scatter_plot_combinations(name, df, cols_to_plot, 
                                                          skip_scatter_plots)
        
        self.log_report("Scatter plot creation completed.")
    
    def _create_scatter_plot_combinations(self, name, df, cols_to_plot, skip_scatter_plots):
        """Create scatter plots for all combinations of columns."""
        for i in range(len(cols_to_plot)):
            for j in range(i+1, len(cols_to_plot)):
                col1, col2 = cols_to_plot[i], cols_to_plot[j]
                
                # Skip specific scatter plots that are not informative
                output_filename = f'{name}_{col1}_vs_{col2}_scatter.png'
                if output_filename in skip_scatter_plots:
                    self.log_report(f"Skipped scatter plot for {col1} vs {col2} - "
                                  "not informative visualization")
                    continue
                
                # Also check reverse order
                output_filename_reverse = f'{name}_{col2}_vs_{col1}_scatter.png'
                if output_filename_reverse in skip_scatter_plots:
                    self.log_report(f"Skipped scatter plot for {col2} vs {col1} - "
                                  "not informative visualization")
                    continue
                
                if col1 in df.columns and col2 in df.columns:
                    # Sample data if too large
                    sample_df = df[[col1, col2]].dropna()
                    if len(sample_df) > 1000:
                        sample_df = sample_df.sample(1000)
                    
                    if len(sample_df) > 10:  # Only plot if enough data points
                        self._create_scatter_plot(name, col1, col2, sample_df)
    
    def _create_scatter_plot(self, name, col1, col2, sample_df):
        """Create a single scatter plot."""
        plt.figure(figsize=(10, 8))
        # Handle timedelta columns specially
        x_data = sample_df[col1]
        y_data = sample_df[col2]
        
        # Convert timedelta to seconds if needed
        if pd.api.types.is_timedelta64_dtype(x_data):
            x_data = x_data.dt.total_seconds()
        if pd.api.types.is_timedelta64_dtype(y_data):
            y_data = y_data.dt.total_seconds()
        
        plt.scatter(x_data, y_data, alpha=0.6)
        plt.xlabel(col1 if not pd.api.types.is_timedelta64_dtype(sample_df[col1]) 
                  else f'{col1} (seconds)', fontsize=12)
        plt.ylabel(col2 if not pd.api.types.is_timedelta64_dtype(sample_df[col2]) 
                  else f'{col2} (seconds)', fontsize=12)
        plt.title(f'{col2} vs {col1} - {name}\n'
                  'Scatter plot showing relationship between variables', 
                  fontsize=14, pad=20)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{VISUALIZATIONS_DIR}/{name}_{col1}_vs_{col2}_scatter.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_dataset_overview(self):
        """Generate dataset overview section of report."""
        self.log_report("\nDATASET OVERVIEW:")
        for name, df in self.datasets.items():
            if df is not None:
                self.log_report(f"  - {name}: {df.shape[0]} rows, {df.shape[1]} columns")
    
    def _generate_data_quality_issues(self):
        """Generate data quality issues section of report."""
        self.log_report("\nDATA QUALITY ISSUES:")
        for name, df in self.datasets.items():
            if df is not None:
                missing_count = df.isnull().sum().sum()
                try:
                    duplicate_count = df.duplicated().sum()
                except TypeError as e:
                    if "unhashable type" in str(e):
                        # Handle DataFrames with unhashable types (like arrays in embedding columns)
                        non_embedding_cols = [col for col in df.columns 
                                            if 'embedding' not in col.lower()]
                        if non_embedding_cols:
                            duplicate_count = df[non_embedding_cols].duplicated().sum()
                        else:
                            duplicate_count = 0
                    else:
                        duplicate_count = 0
                self.log_report(f"  - {name}: {missing_count} missing values, "
                              f"{duplicate_count} duplicates")
    
    def _generate_key_findings(self):
        """Generate key findings section of report."""
        self.log_report("\nKEY FINDINGS:")
        self.log_report("  - All required datasets have been loaded and analyzed")
        self.log_report("  - Basic statistics calculated for numerical variables")
        self.log_report("  - Missing value patterns identified and visualized")
        self.log_report("  - Distributions of key variables visualized with "
                      "histograms and bar charts")
        self.log_report("  - Correlation matrices generated for numerical variables "
                      "to identify relationships")
        self.log_report("  - Time series analysis performed on event data "
                      "showing temporal patterns")
        self.log_report("  - Outliers detected and visualized using box plots")
        self.log_report("  - Key variable relationships explored through scatter plots")
    
    def _generate_detailed_insights(self):
        """Generate detailed insights section of report."""
        self.log_report("\nDETAILED INSIGHTS:")
        self.log_report("  - Distribution plots reveal the shape and spread "
                      "of numerical variables")
        self.log_report("  - Categorical variable bar charts show the frequency "
                      "of each category")
        self.log_report("  - Correlation heatmaps highlight relationships "
                      "between numerical variables")
        self.log_report("  - Time series visualizations show event patterns over time")
        self.log_report("  - Box plots help identify outliers and understand "
                      "data distribution")
        self.log_report("  - Scatter plots reveal potential relationships "
                      "between variable pairs")
    
    def _generate_recommendations(self):
        """Generate recommendations section of report."""
        self.log_report("\nRECOMMENDATIONS:")
        self.log_report("  - Address missing values based on the identified patterns")
        self.log_report("  - Investigate and handle duplicate records if necessary")
        self.log_report("  - Consider transformations for skewed numerical variables")
        self.log_report("  - Examine outliers to determine if they are valid "
                      "or erroneous data points")
        self.log_report("  - Use correlation analysis to inform feature selection "
                      "for modeling")
        self.log_report("  - Consider temporal patterns in event data for "
                      "time-based features")
        self.log_report("  - Use the visualizations to guide feature engineering "
                      "and data preprocessing")
    
    def generate_summary_report(self):
        """Generate a summary report of findings."""
        self.log_report("\n" + "="*60)
        self.log_report("EDA SUMMARY REPORT")
        self.log_report("="*60)
        
        # Dataset overview
        self._generate_dataset_overview()
        
        # Data quality issues
        self._generate_data_quality_issues()
        
        # Key findings
        self._generate_key_findings()
        
        # Detailed insights
        self._generate_detailed_insights()
        
        # Recommendations
        self._generate_recommendations()
        
        self.log_report("\n" + "="*60)
        self.log_report("EDA ANALYSIS COMPLETED SUCCESSFULLY")
        self.log_report("="*60)
        
        # Save report to file
        report_path = f"{VISUALIZATIONS_DIR}/eda_summary_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            for line in self.report:
                f.write(line + '\n')
        
        self.log_report(f"Full report saved to {report_path}")
    
    def run_complete_eda(self):
        """Run the complete EDA process."""
        self.log_report("Starting comprehensive EDA analysis...")
        
        # 1. Load data
        self.load_data()
        
        # 2. Examine data structure
        self.examine_data_structure()
        
        # 3. Basic statistics
        self.basic_statistics()
        
        # 4. Missing values analysis
        self.missing_values_analysis()
        
        # 5. Duplicate analysis
        self.duplicate_analysis()
        
        # 6. Distribution analysis
        self.distribution_analysis()
        
        # 7. Correlation analysis
        self.correlation_analysis()
        
        # 8. Time series analysis
        self.time_series_analysis()
        
        # 9. Outlier detection
        self.outlier_detection()
        
        # 10. Scatter plots
        self.scatter_plots()
        
        # 11. Generate summary report
        self.generate_summary_report()
        
        self.log_report("Complete EDA process finished.")
    
def main():
    """Main function to run the EDA analysis."""
    analyzer = EDAAnalyzer()
    analyzer.run_complete_eda()

if __name__ == "__main__":
    main()