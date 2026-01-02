#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Check if all models are available and working properly.
Tests each model with sample predictions.
"""
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def check_all_models():
    print("=" * 70)
    print("COMPREHENSIVE MODEL AVAILABILITY CHECK")
    print("=" * 70)
    
    models_status = {}
    
    # Define all models and their paths
    model_definitions = {
        "logistic_regression": "models/baseline/logreg.joblib",
        "naive_bayes": "models/baseline/nb.joblib",
        "svc": "models/baseline/svc.joblib",
        "random_forest": "models/baseline/rf.joblib",
        "cnn": "models/baseline/cnn.pth",
        "bilstm": "models/baseline/bilstm.pth",
        "hecan": "models/baseline/hecan.pth",
        "distilbert": "models/baseline/logreg_distilbert.joblib",
        "ensemble": "ensemble (uses multiple models)"
    }
    
    print("\n1. CHECKING MODEL FILES")
    print("-" * 70)
    
    for model_name, model_path in model_definitions.items():
        if model_name == "ensemble":
            models_status[model_name] = {"exists": True, "path": model_path}
            print(f"[OK] {model_name:20} - Virtual (combines other models)")
            continue
            
        path = Path(model_path)
        exists = path.exists()
        models_status[model_name] = {"exists": exists, "path": model_path}
        
        status = "[OK]" if exists else "[MISSING]"
        print(f"{status} {model_name:20} - {model_path}")
        
        if not exists:
            print(f"  -> Missing: Train with appropriate script")
    
    # Check DistilBERT transformer files
    print("\n2. CHECKING DISTILBERT TRANSFORMER FILES")
    print("-" * 70)
    
    distilbert_dir = Path("models/transformer/distilbert_local")
    if distilbert_dir.exists():
        print(f"[OK] DistilBERT directory found: {distilbert_dir}")
        required_files = ["config.json", "tokenizer_config.json"]
        for file in required_files:
            if (distilbert_dir / file).exists():
                print(f"  [OK] {file}")
            else:
                print(f"  [MISSING] {file}")
        if (distilbert_dir / "pytorch_model.bin").exists() or (distilbert_dir / "model.safetensors").exists():
            print(f"  [OK] Model weights")
        else:
            print(f"  [MISSING] Model weights")
    else:
        print(f"[MISSING] DistilBERT directory not found: {distilbert_dir}")
        print("  -> Run: python scripts/download_transformer.py")
    
    # Test predictions
    print("\n3. TESTING MODEL PREDICTIONS")
    print("-" * 70)
    
    test_texts = [
        "Hello, how are you?",
        "You are stupid",
        "I hate you"
    ]
    
    try:
        from src.api.predict import predict_text_with_model
        
        for model_name in model_definitions.keys():
            if not models_status[model_name]["exists"] and model_name != "ensemble":
                print(f"[SKIP] {model_name:20} - Not trained")
                continue
            
            try:
                result = predict_text_with_model(test_texts[0], model_name=model_name)
                label = result.get("label", -1)
                score = result.get("score", 0.0)
                latency = result.get("latency_ms", 0)
                
                print(f"[OK] {model_name:20} - Label: {label}, Score: {score:.2f}, Latency: {latency}ms")
                models_status[model_name]["working"] = True
            except Exception as e:
                print(f"[ERROR] {model_name:20} - {str(e)[:50]}")
                models_status[model_name]["working"] = False
    
    except Exception as e:
        print(f"[ERROR] Cannot test predictions: {e}")
        print("  -> Ensure all dependencies are installed")
    
    # Summary
    print("\n4. SUMMARY")
    print("=" * 70)
    
    total_models = len([m for m in model_definitions.keys() if m != "ensemble"])
    trained_models = len([m for m, s in models_status.items() if s["exists"] and m != "ensemble"])
    working_models = len([m for m, s in models_status.items() if s.get("working", False)])
    
    print(f"Total Models: {total_models + 1} (including ensemble)")
    print(f"Trained Models: {trained_models}")
    print(f"Working Models: {working_models}")
    
    if trained_models == total_models:
        print("\n[SUCCESS] ALL MODELS ARE TRAINED AND READY!")
    else:
        print(f"\n[WARNING] {total_models - trained_models} model(s) need training")
        print("\nTo train missing models:")
        
        if not models_status["logistic_regression"]["exists"]:
            print("  python -m src.training.train_baseline")
        if not models_status["cnn"]["exists"] or not models_status["bilstm"]["exists"] or not models_status["hecan"]["exists"]:
            print("  python -m src.training.train_deep_models")
        if not models_status["distilbert"]["exists"]:
            print("  python scripts/download_transformer.py")
            print("  python src/training/train_distilbert.py")
    
    print("\n" + "=" * 70)
    
    # Return status for programmatic use
    return models_status

if __name__ == "__main__":
    try:
        status = check_all_models()
        
        # Exit with error code if not all models are working
        all_working = all(s.get("working", False) or not s["exists"] for s in status.values())
        sys.exit(0 if all_working else 1)
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
