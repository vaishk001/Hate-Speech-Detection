import gradio as gr
from src.api.predict import predict_text
from pathlib import Path
import json

LABEL_NAMES = {0: "Normal", 1: "Offensive", 2: "Hate"}
LABEL_COLORS = {0: "#22C55E", 1: "#f59e0b", 2: "#EF4444"}
LABEL_EMOJI = {0: "✅", 1: "⚠️", 2: "🚫"}

def classify_text(text, model_choice):
    """Classify hate speech in text"""
    if not text.strip():
        return "⚠️ Please enter some text", "", "", ""
    
    try:
        result = predict_text(text, model_name=model_choice)
        label_id = result['label']
        label_name = LABEL_NAMES[label_id]
        emoji = LABEL_EMOJI[label_id]
        confidence = result['score']
        latency = result['latency_ms']
        lang = result.get('lang', 'unknown')
        
        # Format output
        prediction = f"{emoji} **{label_name}**"
        confidence_str = f"{confidence:.1%}"
        details = f"**Language:** {lang.upper()}\n**Model:** {result['model_name']}\n**Latency:** {latency:.0f}ms"
        
        # Color based on prediction
        color = LABEL_COLORS[label_id]
        
        return prediction, confidence_str, details, color
        
    except Exception as e:
        return f"❌ Error: {str(e)}", "", "", "#EF4444"

def batch_classify(file):
    """Process CSV file with multiple texts"""
    if file is None:
        return "⚠️ Please upload a CSV file"
    
    try:
        import pandas as pd
        df = pd.read_csv(file.name)
        
        if 'text' not in df.columns:
            return "❌ CSV must have a 'text' column"
        
        results = []
        for text in df['text']:
            result = predict_text(str(text))
            results.append({
                'text': text,
                'prediction': LABEL_NAMES[result['label']],
                'confidence': f"{result['score']:.2%}",
                'latency_ms': f"{result['latency_ms']:.0f}"
            })
        
        results_df = pd.DataFrame(results)
        return results_df
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Available models
MODELS = [
    "baseline-logreg",
    "baseline-nb", 
    "baseline-svc",
    "baseline-rf",
    "distilbert",
    "ensemble"
]

# Custom CSS
custom_css = """
.prediction-box {
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
}
"""

# Create Gradio interface with tabs
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🚫 Hate Speech Detection System
    
    Multilingual hate speech classifier supporting **English**, **Hindi**, and **Hinglish**.
    
    Uses multiple ML models: Logistic Regression, Naive Bayes, SVM, Random Forest, and DistilBERT transformer.
    """)
    
    with gr.Tabs():
        # Single text classification
        with gr.Tab("Single Text"):
            with gr.Row():
                with gr.Column(scale=2):
                    text_input = gr.Textbox(
                        label="Enter text to analyze",
                        placeholder="Type or paste text here...",
                        lines=5
                    )
                    model_dropdown = gr.Dropdown(
                        choices=MODELS,
                        value="baseline-logreg",
                        label="Select Model",
                        info="Choose ML model for classification"
                    )
                    classify_btn = gr.Button("🔍 Analyze Text", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    prediction_output = gr.Markdown(label="Prediction")
                    confidence_output = gr.Textbox(label="Confidence Score", interactive=False)
                    details_output = gr.Markdown(label="Details")
            
            classify_btn.click(
                fn=classify_text,
                inputs=[text_input, model_dropdown],
                outputs=[prediction_output, confidence_output, details_output, gr.State()]
            )
            
            # Examples
            gr.Examples(
                examples=[
                    ["You are amazing and wonderful!", "baseline-logreg"],
                    ["I hate you so much", "baseline-logreg"],
                    ["तुम बेवकूफ हो", "distilbert"],
                    ["This is a normal message", "ensemble"]
                ],
                inputs=[text_input, model_dropdown]
            )
        
        # Batch processing
        with gr.Tab("Batch Processing"):
            gr.Markdown("Upload a CSV file with a 'text' column to classify multiple texts at once.")
            
            file_input = gr.File(label="Upload CSV", file_types=[".csv"])
            batch_btn = gr.Button("📊 Process Batch", variant="primary")
            batch_output = gr.Dataframe(label="Results")
            
            batch_btn.click(
                fn=batch_classify,
                inputs=file_input,
                outputs=batch_output
            )
        
        # Model info
        with gr.Tab("About Models"):
            gr.Markdown("""
            ## Available Models
            
            ### Baseline Models (Fast)
            - **Logistic Regression** - TF-IDF + LogisticRegression (50-100ms)
            - **Naive Bayes** - TF-IDF + MultinomialNB (5-10ms) 
            - **SVC** - TF-IDF + LinearSVC (5-10ms)
            - **Random Forest** - TF-IDF + RandomForestClassifier (100-150ms)
            
            ### Transformer Model (Accurate)
            - **DistilBERT** - Multilingual transformer (100-200ms)
            
            ### Ensemble
            - Combines multiple models for balanced predictions
            
            ## Prediction Classes
            - ✅ **Normal** - Non-offensive content
            - ⚠️ **Offensive** - Mildly offensive/rude content  
            - 🚫 **Hate** - Severe hate speech
            
            ## Supported Languages
            - English (en)
            - Hindi (hi)
            - Hinglish (mixed Hindi-English)
            """)
    
    gr.Markdown("""
    ---
    Built with [Gradio](https://gradio.app) • Models: scikit-learn, PyTorch, Transformers
    """)

if __name__ == "__main__":
    demo.launch()
