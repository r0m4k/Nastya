import os
import modal

# --- Configuration ---
BASE_MODEL_ID = "meta-llama/Meta-Llama-3-70B-Instruct" # "meta-llama/Meta-Llama-3-70B-Instruct" "meta-llama/Meta-Llama-3.1-8B-Instruct "meta-llama/Meta-Llama-3.3-70B-Instruct"
VOLUME_NAME = "finetuning-volume"
CACHE_PATH = "/basemodel"
# --- End of Configuration ---

# Add an environment variable to disable the fast downloader
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("transformers", "torch", "accelerate", "huggingface_hub")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "0"}) # Disables hf_transfer
)

app = modal.App("model-downloader-app", image=image)

volume = modal.NetworkFileSystem.from_name(VOLUME_NAME, create_if_missing=True)

@app.function(
    network_file_systems={CACHE_PATH: volume},
    secrets=[modal.Secret.from_name("my-huggingface-secret")],
    gpu="any",
    timeout=3600,
)
def download_model_to_cache():
    """
    This function uses snapshot_download to efficiently download model files
    to a persistent volume without loading them into memory.
    """
    from huggingface_hub import snapshot_download

    print(f"Starting download of '{BASE_MODEL_ID}' to cache directory: {CACHE_PATH}")
    print("NOTE: High-speed transfer is disabled. Download may be slower but more stable.")

    try:
        snapshot_download(
            repo_id=BASE_MODEL_ID,
            cache_dir=CACHE_PATH,
            token=os.environ["HF_TOKEN"],
            local_dir_use_symlinks=False,
        )

        print("\n✅ Success! Model files are downloaded and cached in the volume.")
        print(f"  - Volume Name: {VOLUME_NAME}")
        print(f"  - Cached to Path: {CACHE_PATH}")

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        raise e

@app.local_entrypoint()
def main():
    print("🚀 Launching remote job to download and cache the model...")
    download_model_to_cache.remote()