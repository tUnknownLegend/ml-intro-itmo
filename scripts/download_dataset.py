import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="t-tech/T-ECD",
    repo_type="dataset",
    allow_patterns="dataset/small/",  # для небольшой версии; "dataset/full/" — для полной
    local_dir="./t_ecd_data",
    token=os.getenv("DS_DOWNLOAD")
)
