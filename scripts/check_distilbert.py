#!/usr/bin/env python
"""
Check if DistilBERT model is available and properly configured.
"""
from pathlib import Path

def check_distilbert():
    print("=" * 60)
    print("DistilBERT Availability Check")
    print("=" * 60)
    
    distilbert_dir = Path("models/transformer/distilbert_local")
    distilbert_clf = Path("models/baseline/logreg_distilbert.joblib")
    
    # Check model directory
    if distilbert_dir.exists():
        print(f"✓ DistilBERT model directory found: {distilbert_dir}")
        
        # Check for required files (pytorch_model.bin OR model.safetensors)
        required_files = ["config.json", "tokenizer_config.json"]
        model_file = (distilbert_dir / "pytorch_model.bin").exists() or (distilbert_dir / "model.safetensors").exists()
        missing_files = []
        
        for file in required_files:
            if not (distilbert_dir / file).exists():
                missing_files.append(file)
        
        if not model_file:
            missing_files.append("pytorch_model.bin or model.safetensors")
        
        if missing_files:
            print(f"✗ Missing files: {', '.join(missing_files)}")
            print("  Run: python scripts/download_transformer.py")
        else:
            print("✓ All required model files present")
    else:
        print(f"✗ DistilBERT model directory not found: {distilbert_dir}")
        print("  Run: python scripts/download_transformer.py")
    
    # Check trained classifier
    if distilbert_clf.exists():
        print(f"✓ DistilBERT classifier found: {distilbert_clf}")
    else:
        print(f"✗ DistilBERT classifier not found: {distilbert_clf}")
        print("  Run: python src/training/train_distilbert.py")
    
    # Try loading the model
    print("\nTesting model loading...")
    try:
        from transformers import AutoTokenizer, AutoModel
        import warnings
        
        if distilbert_dir.exists():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tokenizer = AutoTokenizer.from_pretrained(str(distilbert_dir), local_files_only=True)
                model = AutoModel.from_pretrained(str(distilbert_dir), local_files_only=True)
            print("✓ DistilBERT model loads successfully")
            print(f"  Model type: {model.config.model_type}")
            print(f"  Hidden size: {model.config.hidden_size}")
            print(f"  Vocab size: {tokenizer.vocab_size}")
        else:
            print("✗ Cannot test loading - model directory not found")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
    
    # Try loading classifier
    if distilbert_clf.exists():
        try:
            import joblib
            clf = joblib.load(distilbert_clf)
            print(f"✓ Classifier loads successfully")
            print(f"  Classes: {clf.classes_}")
        except Exception as e:
            print(f"✗ Error loading classifier: {e}")
    
    print("\n" + "=" * 60)
    print("Summary:")
    
    if distilbert_dir.exists() and distilbert_clf.exists():
        print("✓ DistilBERT is ready to use!")
        print("  You can select 'distilbert' in the UI model dropdown")
    elif distilbert_dir.exists():
        print("⚠ DistilBERT model downloaded but not trained")
        print("  Run: python src/training/train_distilbert.py")
    else:
        print("✗ DistilBERT not available")
        print("  1. Run: python scripts/download_transformer.py")
        print("  2. Run: python src/training/train_distilbert.py")
    
    print("=" * 60)

if __name__ == "__main__":
    check_distilbert()
