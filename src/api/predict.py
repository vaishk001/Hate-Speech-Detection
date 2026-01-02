# src/api/predict.py
from matplotlib.pyplot import text
from src.data.preprocess import preprocess_text
from typing import Dict, Tuple, Optional
from pathlib import Path
import time

# optional imports with fallbacks
try:
    import importlib
    joblib = importlib.import_module("joblib")
except Exception:
    joblib = None

# DB insert helper (optional)
try:
    from src.utils.db import insert_prediction
except Exception:
    def insert_prediction(*args, **kwargs):
        return None

# model paths
BASELINE_PATH = Path("models/baseline/logreg.joblib")
TRANSFORMER_DIR = Path("models/transformer/distilbert_local")
TRANSFORMER_SKLEARN = Path("models/baseline/logreg_transformer.joblib")
DISTILBERT_DIR = Path("models/transformer/distilbert_local")
DISTILBERT_SKLEARN = Path("models/baseline/logreg_distilbert.joblib")

# caches
_baseline_model = None
_transformer_tokenizer = None
_transformer_model = None
_transformer_clf = None
_distilbert_tokenizer = None
_distilbert_model = None
_distilbert_clf = None

def _load_baseline():
    global _baseline_model
    if _baseline_model is None:
        if joblib is None:
            raise RuntimeError("joblib not installed")
        if not BASELINE_PATH.exists():
            raise FileNotFoundError(f"Baseline model not found at {BASELINE_PATH}")
        _baseline_model = joblib.load(str(BASELINE_PATH))
    return _baseline_model

def _try_load_transformer_components():
    """
    Try to load: tokenizer, transformer model (for embeddings), and sklearn clf (trained on embeddings).
    Return (tokenizer, transformer_model, sklearn_clf) or (None, None, None) on failure.
    """
    global _transformer_tokenizer, _transformer_model, _transformer_clf
    if _transformer_clf is None and TRANSFORMER_SKLEARN.exists() and joblib is not None:
        try:
            _transformer_clf = joblib.load(str(TRANSFORMER_SKLEARN))
        except Exception:
            _transformer_clf = None
    if _transformer_model is not None and _transformer_tokenizer is not None:
        return _transformer_tokenizer, _transformer_model, _transformer_clf

    try:
        from transformers import AutoTokenizer, AutoModel
        import warnings
        if TRANSFORMER_DIR.exists():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _transformer_tokenizer = AutoTokenizer.from_pretrained(str(TRANSFORMER_DIR), local_files_only=True, fix_mistral_regex=True)
                _transformer_model = AutoModel.from_pretrained(str(TRANSFORMER_DIR), local_files_only=True)
            return _transformer_tokenizer, _transformer_model, _transformer_clf
    except Exception:
        pass
    return None, None, _transformer_clf

def _embed_texts_transformer(texts, tokenizer, model, device='cpu', batch_size=16, max_length=128):
    """
    Compute mean-pooled embeddings for a list of texts using HF model.
    Returns numpy array shape (len(texts), hidden_size).
    """
    import torch
    import numpy as np
    model.to(device)
    model.eval()
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        with torch.no_grad():
            enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            last = out.last_hidden_state  # (B, S, D)
            mask = enc.get('attention_mask', None)
            if mask is not None:
                mask = mask.unsqueeze(-1).expand(last.size()).float()
                summed = (last * mask).sum(1)
                denom = mask.sum(1).clamp(min=1e-9)
                pooled = summed / denom
            else:
                pooled = last.mean(1)
            embs.append(pooled.cpu().numpy())
    return np.vstack(embs)

def _apply_rule_based_override(text: str, label: int, score: float) -> Tuple[int, float]:
    """Apply rule-based overrides using dynamic word categories."""
    from src.utils.word_categories import detect_words_with_categories
    
    detection = detect_words_with_categories(text)
    
    # If hate words detected, upgrade to hate (label 2)
    if detection["has_hate"]:
        return 2, max(score, 0.85)
    
    # If offensive words detected and currently normal, upgrade to offensive (label 1)
    if detection["has_offensive"] and label == 0:
        return 1, max(score, 0.70)
    
    return label, score

def _predict_with_baseline(text: str) -> Dict:
    model = _load_baseline()
    probs = model.predict_proba([text])[0]
    label = int(probs.argmax())
    score = float(probs.max())
    
    # Apply rule-based override
    label, score = _apply_rule_based_override(text, label, score)
    
    return {"label": label, "score": score, "model_name": "baseline-logreg"}

def predict_text_with_model(text: str, model_name: str = "logistic_regression") -> Dict:
    """
    Predict with specific model selection.
    Supported models: logistic_regression, naive_bayes, svc, ensemble, random_forest, cnn, bilstm, hecan, distilbert
    """
    t0 = time.time()
    txt = str(text).strip()
    cleaned_text, lang = preprocess_text(txt)
    txt = cleaned_text.strip()
    
    model_map = {
        "logistic_regression": "models/baseline/logreg.joblib",
        "naive_bayes": "models/baseline/nb.joblib",
        "svc": "models/baseline/svc.joblib",
        "random_forest": "models/baseline/rf.joblib",
        "cnn": "models/baseline/cnn.pth",
        "bilstm": "models/baseline/bilstm.pth",
        "hecan": "models/baseline/hecan.pth",
        "distilbert": "models/baseline/logreg_distilbert.joblib"
    }
    
    # Ensemble: average predictions from all available baseline models
    if model_name == "ensemble":
        try:
            import numpy as np
            all_probs = []
            baseline_models = ["logistic_regression", "naive_bayes", "svc", "random_forest"]
            for mname in baseline_models:
                mpath = model_map.get(mname)
                if mpath and Path(mpath).exists():
                    model = joblib.load(mpath)
                    probs = model.predict_proba([txt])[0]
                    all_probs.append(probs)
            
            if all_probs:
                avg_probs = np.mean(all_probs, axis=0)
                label = int(avg_probs.argmax())
                score = float(avg_probs.max())
                label, score = _apply_rule_based_override(txt, label, score)
                latency_ms = int((time.time() - t0) * 1000)
                result = {"label": label, "score": score, "model_name": "ensemble", "latency_ms": latency_ms, "lang": lang}
                try:
                    insert_prediction(txt, lang, label, score, "ensemble", latency_ms)
                except Exception:
                    pass
                return result
        except Exception as e:
            print(f"Ensemble error: {e}")
    
    # DistilBERT model
    if model_name == "distilbert":
        global _distilbert_tokenizer, _distilbert_model, _distilbert_clf
        if DISTILBERT_SKLEARN.exists() and DISTILBERT_DIR.exists():
            try:
                if _distilbert_clf is None:
                    _distilbert_clf = joblib.load(str(DISTILBERT_SKLEARN))
                if _distilbert_tokenizer is None or _distilbert_model is None:
                    from transformers import AutoTokenizer, AutoModel
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        _distilbert_tokenizer = AutoTokenizer.from_pretrained(str(DISTILBERT_DIR), local_files_only=True, fix_mistral_regex=True)
                        _distilbert_model = AutoModel.from_pretrained(str(DISTILBERT_DIR), local_files_only=True)
                
                emb = _embed_texts_transformer([txt], _distilbert_tokenizer, _distilbert_model, device='cpu', batch_size=1, max_length=128)
                proba = _distilbert_clf.predict_proba(emb)[0]
                label = int(proba.argmax())
                score = float(proba.max())
                label, score = _apply_rule_based_override(txt, label, score)
                latency_ms = int((time.time() - t0) * 1000)
                result = {"label": label, "score": score, "model_name": "distilbert", "latency_ms": latency_ms, "lang": lang}
                try:
                    insert_prediction(txt, lang, label, score, "distilbert", latency_ms)
                except Exception:
                    pass
                return result
            except Exception as e:
                print(f"DistilBERT error: {e}")
                model_name = "logistic_regression"
        else:
            print("DistilBERT model not found, falling back to logistic regression")
            model_name = "logistic_regression"
    
    # Deep learning models (CNN, BiLSTM, HECAN)
    if model_name in ["cnn", "bilstm", "hecan"]:
        model_path = model_map.get(model_name)
        if model_path and Path(model_path).exists():
            try:
                if model_name == "cnn":
                    from src.models.textcnn import load_model, predict_text as predict_cnn
                    model = load_model(model_path)
                    predictions = predict_cnn(model, [txt])
                    label, score = predictions[0]
                elif model_name == "bilstm":
                    from src.models.bilstm import load_model, predict_text as predict_bilstm
                    model = load_model(model_path)
                    predictions = predict_bilstm(model, [txt])
                    label, score = predictions[0]
                elif model_name == "hecan":
                    from src.models.hecan import load_model, predict_text as predict_hecan
                    model = load_model(model_path)
                    predictions = predict_hecan(model, [txt])
                    label, score = predictions[0]
                
                label, score = _apply_rule_based_override(txt, label, score)
                latency_ms = int((time.time() - t0) * 1000)
                result = {"label": label, "score": score, "model_name": model_name, "latency_ms": latency_ms, "lang": lang}
                try:
                    insert_prediction(txt, lang, label, score, model_name, latency_ms)
                except Exception:
                    pass
                return result
            except Exception as e:
                print(f"{model_name} error: {e}")
                model_name = "logistic_regression"
    
    # Sklearn-based models (LogReg, NB, SVC, RF)
    model_path = model_map.get(model_name, model_map["logistic_regression"])
    if not Path(model_path).exists():
        model_path = model_map["logistic_regression"]
        model_name = "logistic_regression"
    
    try:
        model = joblib.load(model_path)
        probs = model.predict_proba([txt])[0]
        label = int(probs.argmax())
        score = float(probs.max())
        label, score = _apply_rule_based_override(txt, label, score)
        
        latency_ms = int((time.time() - t0) * 1000)
        result = {"label": label, "score": score, "model_name": model_name, "latency_ms": latency_ms, "lang": lang}
        try:
            insert_prediction(txt, lang, label, score, model_name, latency_ms)
        except Exception:
            pass
        return result
    except Exception as e:
        print(f"Model error: {e}")
        return {"label": 0, "score": 0.5, "model_name": "error", "latency_ms": 0, "lang": lang}

def predict_text(text: str) -> Dict:
    """
    Main prediction router.
    - Preprocessing should already be applied by your preprocess module if available; this function expects a raw text string.
    - Chooses transformer classifier when available (and beneficial), otherwise baseline.
    Returns dict with keys: label, score, model_name, lang (if available), latency_ms
    """
    t0 = time.time()
    txt = str(text).strip()
    cleaned_text, lang = preprocess_text(txt)
    txt = cleaned_text.strip()

    # Attempt to use transformer classifier when present:
    tokenizer, transformer_model, sklearn_clf = _try_load_transformer_components()

    # Simple routing heuristics: use transformer clf if present and input longer than threshold
    use_transformer = False
    if sklearn_clf is not None and tokenizer is not None and transformer_model is not None:
        if len(txt.split()) > 8:
            use_transformer = True
        # optional: if contains Devanagari characters, prefer transformer
        if any("\u0900" <= ch <= "\u097F" for ch in txt):
            use_transformer = True

    if use_transformer:
        try:
            # compute embedding (same method as training)
            emb = _embed_texts_transformer([txt], tokenizer, transformer_model, device='cpu', batch_size=1, max_length=128)
            proba = sklearn_clf.predict_proba(emb)[0]
            label = int(proba.argmax())
            score = float(proba.max())
            model_name = "transformer-logreg"
        except Exception:
            # fallback to baseline on error
            r = _predict_with_baseline(txt)
            r.update({'latency_ms': int((time.time() - t0) * 1000)})
            try:
                insert_prediction(txt, 'unknown', int(r['label']), float(r['score']), r['model_name'], r['latency_ms'])
            except Exception:
                pass
            return r
        latency_ms = int((time.time() - t0) * 1000)
        out = {"label": label, "score": score, "model_name": model_name, "latency_ms": latency_ms, "lang": lang}
        try:
            insert_prediction(txt, lang, int(label), float(score), model_name, latency_ms)
        except Exception:
            pass
        return out

    # otherwise baseline
    r = _predict_with_baseline(txt)
    r.update({"latency_ms": int((time.time() - t0) * 1000), "lang": lang})
    try:
        insert_prediction(txt, lang, int(r['label']), float(r['score']), r['model_name'], r['latency_ms'])
    except Exception:
        pass
    return r
