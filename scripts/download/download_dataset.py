import os
from dotenv import load_dotenv
from tecd_downloader import download_dataset

# Load environment variables from .env file
load_dotenv()

token = os.getenv("DS_DOWNLOAD")
if not token:
    raise ValueError("DS_DOWNLOAD environment variable is not set")

download_dataset(
    dataset_path="dataset/small",
    local_dir="t_ecd_small_partial",
    domains=["marketplace"],
    day_begin=1250,
    day_end=1310,
    token=token,
)
