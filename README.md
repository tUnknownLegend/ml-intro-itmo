# ML Intro ITMO

#### [Task](https://github.com/pacifikus/itmo_ml_for_science_course/blob/main/HW/hw_1.md)


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

## Project Structure

- `scripts/` - Contains Python scripts for data downloading and processing
- `t_ecd_data/` - Directory where the dataset will be downloaded
- `.env` - Environment variables file (not included in repo)
- `requirements.txt` - Python dependencies

## Dataset Information

The project uses the T-ECD dataset from Hugging Face. You can choose between:

- Small version: `dataset/small/`
- Full version: `dataset/full/` (uncomment in the download script)
