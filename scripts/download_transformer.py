# scripts/download_transformers.py
"""
Download & save transformer model files locally for offline use.
This script downloads models and saves them under models/transformer/<name>/
Run while ONLINE, then you can run the app offline.
"""

from pathlib import Path
from transformers import AutoTokenizer, AutoModel

MODELS = {
    "distilbert_local": "distilbert-base-multilingual-cased"
}

BASE_DIR = Path("models/transformer")

def download_model(local_name: str, hf_name: str):
    target = BASE_DIR / local_name
    target.mkdir(parents=True, exist_ok=True)
    print(f"Downloading tokenizer for {hf_name} -> {target}")
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    tokenizer.save_pretrained(target)
    print(f"Downloading model for {hf_name} -> {target}")
    model = AutoModel.from_pretrained(hf_name)
    model.save_pretrained(target)
    print(f"Saved {hf_name} to {target}")

def main():
    for local_name, hf_name in MODELS.items():
        try:
            download_model(local_name, hf_name)
        except Exception as e:
            print(f"Failed to download {hf_name}: {e}")
            print("You may retry or download manually from Hugging Face and place files under models/transformer/<local_name>/")
    print("Done. Transformer files saved under models/transformer/")

if __name__ == "__main__":
    main()
