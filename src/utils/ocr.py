# src/utils/ocr.py
import warnings

_reader = None

def get_reader():
    global _reader
    if _reader is None:
        try:
            import easyocr
            import torch
            # Suppress FutureWarnings from easyocr/torch
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                _reader = easyocr.Reader(['en', 'hi'], gpu=(device=='cuda'))
        except ImportError:
            print("EasyOCR is not installed. Please run: pip install easyocr")
            return None
    return _reader

def extract_text_from_image(image_bytes: bytes) -> str:
    """Extracts text from image bytes using EasyOCR."""
    reader = get_reader()
    if reader is None:
        return ""
    
    try:
        import io
        import numpy as np
        from PIL import Image
        
        # Safely convert bytes to numpy array via Pillow
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_np = np.array(image)
        
        # Process with EasyOCR
        result = reader.readtext(image_np, detail=0, paragraph=True)
        return " ".join(result)
    except Exception as e:
        print(f"OCR Error: {e}")
        return ""
