from huggingface_hub import HfApi

print("Uploading to Hugging Face...")
api = HfApi()

# Upload exactly just the required clean files, skipping the 465MB venv folder.
api.upload_folder(
    folder_path=".",
    repo_id="hinex-07/data-cleaning-env",
    repo_type="space",
    ignore_patterns=[
        "venv/*", 
        "__pycache__/*", 
        "*.log", 
        "*.txt", 
        "*.pyc",
        ".git/*",
        "uv.lock"
    ]
)
print("Upload completed successfully!")
