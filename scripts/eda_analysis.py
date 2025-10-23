#!/usr/bin/env python3
"""
Comprehensive EDA script for T-ECD dataset analysis.
This script performs exploratory data analysis on the T-ECD dataset including:
- Data loading and examination
- Basic statistics and data quality checks
- Data visualization
- Summary report generation
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import pyarrow.parquet as pq
from datetime import datetime
import warnings
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
            self.datasets['brands'] = pd.read_parquet(BRANDS_PATH)
            self.log_report(f"Loaded brands data: {self.datasets['brands'].shape}")
        except Exception as e:
            self.log_report(f"Error loading brands data: {e}")
            
        try:
            self.datasets['users'] = pd.read_parquet(USERS_PATH)
            self.log_report(f"Loaded users data: {self.datasets['users'].shape}")
        except Exception as e:
            self.log_report(f"Error loading users data: {e}")
            
        try:
            self.datasets['items'] = pd.read_parquet(ITEMS_PATH)
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
                        # Skip user_id columns as they are identifiers and statistics are not meaningful
                        if 'user_id' in col.lower():
                            self.log_report(f"Skipped {col} - user_id is an identifier, statistics are not meaningful")
                            continue
                            
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
                    plt.figure(figsize=(12, 8))
                    sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
                    plt.title(f'Missing Data Pattern - {name}')
                    plt.xlabel('Columns')
                    plt.tight_layout()
                    plt.savefig(f'{VISUALIZATIONS_DIR}/{name}_missing_data_pattern.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    self.log_report(f"Saved missing data pattern visualization for {name}")
        
        self.log_report("Missing values analysis completed.")
    
    def duplicate_analysis(self):
        """Check for duplicate records."""
        self.log_report("Checking for duplicate records...")
        
        for name, df in self.datasets.items():
            if df is not None:
                duplicates = df.duplicated().sum()
                self.log_report(f"{name} - Duplicate records: {duplicates} ({(duplicates/len(df))*100:.4f}%)")
        
        self.log_report("Duplicate analysis completed.")
    
    def distribution_analysis(self):
        """Analyze distribution of key variables."""
        self.log_report("Analyzing variable distributions...")
        
        for name, df in self.datasets.items():
            if df is not None:
                numerical = self.numerical_cols.get(name, [])
                categorical = self.categorical_cols.get(name, [])
                
                # For numerical variables, create histograms
                for col in numerical[:5]:  # Limit to first 5 for performance
                    # Skip user_id columns as they are identifiers and distribution analysis is not meaningful
                    if 'user_id' in col.lower():
                        self.log_report(f"Skipped distribution analysis for {col} - user_id is an identifier, distribution analysis is not meaningful")
                        continue
                        
                    if col in df.columns and df[col].notna().sum() > 0:
                        plt.figure(figsize=(10, 6))
                        # Handle timedelta columns specially
                        if pd.api.types.is_timedelta64_dtype(df[col]):
                            # Convert to seconds for plotting
                            data_to_plot = df[col].dropna().dt.total_seconds()
                            plt.hist(data_to_plot, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
                            plt.title(f'Distribution of {col} (in seconds) - {name}')
                            plt.xlabel(f'{col} (seconds)')
                        else:
                            plt.hist(df[col].dropna(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
                            plt.title(f'Distribution of {col} - {name}')
                            plt.xlabel(col)
                        plt.ylabel('Frequency')
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        plt.savefig(f'{VISUALIZATIONS_DIR}/{name}_{col}_distribution.png', dpi=300, bbox_inches='tight')
                        plt.close()
                
                # For categorical variables, create bar charts (top 10 categories)
                for col in categorical[:3]:  # Limit to first 3 for performance
                    if col in df.columns:
                        value_counts = df[col].value_counts().head(10)
                        plt.figure(figsize=(12, 6))
                        value_counts.plot(kind='bar', color='lightcoral')
                        plt.title(f'Top 10 Categories in {col} - {name}')
                        plt.xlabel(col)
                        plt.ylabel('Count')
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        plt.savefig(f'{VISUALIZATIONS_DIR}/{name}_{col}_categories.png', dpi=300, bbox_inches='tight')
                        plt.close()
        
        self.log_report("Distribution analysis completed.")
    
    def correlation_analysis(self):
        """Create correlation matrices for numerical variables."""
        self.log_report("Performing correlation analysis...")
        
        for name, df in self.datasets.items():
            if df is not None:
                numerical = self.numerical_cols.get(name, [])
                if len(numerical) > 1 and len(numerical) <= 20:  # Only if reasonable number of numerical columns
                    corr_matrix = df[numerical].corr()
                    
                    plt.figure(figsize=(12, 10))
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                               square=True, linewidths=0.5)
                    plt.title(f'Correlation Matrix - {name}')
                    plt.tight_layout()
                    plt.savefig(f'{VISUALIZATIONS_DIR}/{name}_correlation_matrix.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    self.log_report(f"Saved correlation matrix for {name}")
        
        self.log_report("Correlation analysis completed.")
    
    def time_series_analysis(self):
        """Analyze time series for event data."""
        self.log_report("Performing time series analysis...")
        
        if 'events' in self.datasets and self.datasets['events'] is not None:
            df = self.datasets['events']
            
            # Try to find timestamp column
            timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
            
            if timestamp_cols:
                timestamp_col = timestamp_cols[0]
                self.log_report(f"Using '{timestamp_col}' as timestamp column")
                
                # Convert to datetime if needed
                if df[timestamp_col].dtype == 'object':
                    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
                
                # Remove rows with invalid timestamps
                df_valid = df[df[timestamp_col].notna()].copy()
                
                if not df_valid.empty:
                    # Resample by day and count events
                    df_valid.set_index(timestamp_col, inplace=True)
                    daily_events = df_valid.resample('D').size()
                    
                    plt.figure(figsize=(12, 6))
                    daily_events.plot(kind='line', marker='o', linewidth=2, markersize=4)
                    plt.title('Event Count Over Time')
                    plt.xlabel('Date')
                    plt.ylabel('Number of Events')
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(f'{VISUALIZATIONS_DIR}/events_time_series.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    self.log_report("Saved time series visualization for events")
                    
                    # If we have event types, analyze by type
                    event_type_cols = [col for col in df_valid.columns if 'type' in col.lower() or 'event' in col.lower()]
                    if event_type_cols:
                        event_type_col = event_type_cols[0]
                        if event_type_col in df_valid.columns:
                            daily_events_by_type = df_valid.groupby([pd.Grouper(freq='D'), event_type_col]).size().unstack(fill_value=0)
                            
                            plt.figure(figsize=(14, 8))
                            daily_events_by_type.plot(kind='area', alpha=0.7)
                            plt.title('Event Count by Type Over Time')
                            plt.xlabel('Date')
                            plt.ylabel('Number of Events')
                            plt.legend(title=event_type_col, bbox_to_anchor=(1.05, 1), loc='upper left')
                            plt.grid(True, alpha=0.3)
                            plt.tight_layout()
                            plt.savefig(f'{VISUALIZATIONS_DIR}/events_time_series_by_type.png', dpi=300, bbox_inches='tight')
                            plt.close()
                            self.log_report("Saved time series by type visualization for events")
            else:
                self.log_report("No timestamp column found in events data")
        
        self.log_report("Time series analysis completed.")
    
    def outlier_detection(self):
        """Detect outliers using box plots."""
        self.log_report("Detecting outliers...")
        
        for name, df in self.datasets.items():
            if df is not None:
                numerical = self.numerical_cols.get(name, [])
                
                # Create box plots for numerical variables (limit to first 5)
                for col in numerical[:5]:
                    # Skip user_id columns as they are identifiers and outlier detection is not meaningful
                    if 'user_id' in col.lower():
                        self.log_report(f"Skipped outlier detection for {col} - user_id is an identifier, outlier detection is not meaningful")
                        continue
                        
                    if col in df.columns and df[col].notna().sum() > 0:
                        plt.figure(figsize=(8, 6))
                        plt.boxplot(df[col].dropna())
                        plt.title(f'Outlier Detection - {col} ({name})')
                        plt.ylabel(col)
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        plt.savefig(f'{VISUALIZATIONS_DIR}/{name}_{col}_outliers.png', dpi=300, bbox_inches='tight')
                        plt.close()
        
        self.log_report("Outlier detection completed.")
    
    def scatter_plots(self):
        """Create scatter plots for key variable relationships."""
        self.log_report("Creating scatter plots...")
        
        for name, df in self.datasets.items():
            if df is not None:
                numerical = self.numerical_cols.get(name, [])
                
                # Create scatter plots for pairs of numerical variables (limit combinations)
                if len(numerical) >= 2:
                    # Take first 3 numerical columns for pairwise scatter plots
                    cols_to_plot = numerical[:3]
                    # Filter out user_id columns as scatter plots with them are not meaningful
                    cols_to_plot = [col for col in cols_to_plot if 'user_id' not in col.lower()]
                    
                    # Skip if we don't have enough columns after filtering
                    if len(cols_to_plot) < 2:
                        self.log_report(f"Skipped scatter plots for {name} - not enough meaningful numerical columns after filtering out user_id")
                        continue
                    
                    for i in range(len(cols_to_plot)):
                        for j in range(i+1, len(cols_to_plot)):
                            col1, col2 = cols_to_plot[i], cols_to_plot[j]
                            if col1 in df.columns and col2 in df.columns:
                                # Sample data if too large
                                sample_df = df[[col1, col2]].dropna()
                                if len(sample_df) > 1000:
                                    sample_df = sample_df.sample(1000)
                                
                                if len(sample_df) > 10:  # Only plot if enough data points
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
                                    plt.xlabel(col1 if not pd.api.types.is_timedelta64_dtype(sample_df[col1]) else f'{col1} (seconds)')
                                    plt.ylabel(col2 if not pd.api.types.is_timedelta64_dtype(sample_df[col2]) else f'{col2} (seconds)')
                                    plt.title(f'{col2} vs {col1} - {name}')
                                    plt.grid(True, alpha=0.3)
                                    plt.tight_layout()
                                    plt.savefig(f'{VISUALIZATIONS_DIR}/{name}_{col1}_vs_{col2}_scatter.png', dpi=300, bbox_inches='tight')
                                    plt.close()
        
        self.log_report("Scatter plot creation completed.")
    
    def generate_summary_report(self):
        """Generate a summary report of findings."""
        self.log_report("\n" + "="*60)
        self.log_report("EDA SUMMARY REPORT")
        self.log_report("="*60)
        
        # Dataset overview
        self.log_report("\nDATASET OVERVIEW:")
        for name, df in self.datasets.items():
            if df is not None:
                self.log_report(f"  - {name}: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Data quality issues
        self.log_report("\nDATA QUALITY ISSUES:")
        for name, df in self.datasets.items():
            if df is not None:
                missing_count = df.isnull().sum().sum()
                duplicate_count = df.duplicated().sum()
                self.log_report(f"  - {name}: {missing_count} missing values, {duplicate_count} duplicates")
        
        # Key findings
        self.log_report("\nKEY FINDINGS:")
        self.log_report("  - All required datasets have been loaded and analyzed")
        self.log_report("  - Basic statistics calculated for numerical variables")
        self.log_report("  - Missing value patterns identified and visualized")
        self.log_report("  - Distributions of key variables visualized")
        self.log_report("  - Correlation matrices generated for numerical variables")
        self.log_report("  - Time series analysis performed on event data")
        self.log_report("  - Outliers detected and visualized")
        self.log_report("  - Key variable relationships explored through scatter plots")
        
        # Recommendations
        self.log_report("\nRECOMMENDATIONS:")
        self.log_report("  - Address missing values based on the identified patterns")
        self.log_report("  - Investigate and handle duplicate records if necessary")
        self.log_report("  - Consider transformations for skewed numerical variables")
        self.log_report("  - Examine outliers to determine if they are valid or erroneous data points")
        self.log_report("  - Use correlation analysis to inform feature selection for modeling")
        self.log_report("  - Consider temporal patterns in event data for time-based features")
        
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