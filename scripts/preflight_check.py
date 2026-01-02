"""Preflight checks for installation and offline readiness."""
import sys
import warnings
from pathlib import Path

PROJECT = Path('.').resolve()


def check_dependencies() -> list:
    """Check if required packages are installed."""
    issues = []
    required = ['pandas', 'numpy', 'sklearn', 'torch', 'transformers', 'nicegui']
    
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            issues.append(f"Missing package: {pkg}")
    
    return issues


def check_files() -> list:
    """Check if required files/directories exist."""
    issues = []
    required_files = [
        ('models/baseline/logreg.joblib', True),
        ('data/clean_data.csv', False),
        ('data/app.db', False),
    ]
    
    for fpath, required in required_files:
        p = PROJECT / fpath
        if not p.exists():
            status = "required" if required else "optional"
            issues.append(f"Missing ({status}): {fpath}")
    
    return issues


def check_transformer_offline() -> list:
    """Check if transformer models are available offline."""
    issues = []
    
    try:
        from transformers import AutoTokenizer
        transformer_path = PROJECT / 'models' / 'transformer' / 'distilbert_local'
        
        if transformer_path.exists():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    AutoTokenizer.from_pretrained(str(transformer_path), local_files_only=True)
            except Exception as e:
                issues.append(f"Transformer load failed: {str(e)[:60]}")
        else:
            issues.append("Transformer model not downloaded (optional)")
    except ImportError:
        issues.append("transformers not installed")
    
    return issues


def main():
    """Run all preflight checks."""
    all_issues = []
    
    print("Running preflight checks...\n")
    
    # Check dependencies
    dep_issues = check_dependencies()
    if dep_issues:
        print("❌ Dependencies:")
        for issue in dep_issues:
            print(f"   {issue}")
        all_issues.extend(dep_issues)
    else:
        print("✓ All dependencies installed")
    
    # Check files
    file_issues = check_files()
    if file_issues:
        print("\n⚠ Files:")
        for issue in file_issues:
            print(f"   {issue}")
        all_issues.extend([i for i in file_issues if 'required' in i])
    else:
        print("✓ All required files present")
    
    # Check transformer
    tf_issues = check_transformer_offline()
    if tf_issues:
        print("\n⚠ Transformer:")
        for issue in tf_issues:
            print(f"   {issue}")
    else:
        print("✓ Transformer ready for offline use")
    
    # Summary
    print()
    if all_issues:
        print(f"❌ PREFLIGHT CHECK FAILED ({len(all_issues)} critical issues)")
        return 1
    else:
        print("✅ PREFLIGHT CHECK PASSED")
        return 0


if __name__ == '__main__':
    sys.exit(main())
