# ML Intro ITMO

#### [Task](https://github.com/pacifikus/itmo_ml_for_science_course/blob/main/readme.md)

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

## Dataset Information

The project uses the T-ECD dataset from Hugging Face. You can choose between:

- Small version: `dataset/small/`
- Full version: `dataset/full/` (uncomment in the download script)

## EDA Analysis

The repository includes an exploratory data analysis (EDA) of the T-ECD dataset:

- **Script**: [`scripts/eda_analysis.py`](scripts/eda_analysis.py) - A comprehensive Python script that performs EDA including data loading, basic statistics, data quality checks, and data visualization
- **Report**: [`EDA.md`](EDA.md) - A detailed report in Russian documenting the findings of the EDA with all generated visualizations
