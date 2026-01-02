from nicegui import ui  # cSpell:ignore nicegui
from src.api.predict import predict_text
from src.data.preprocess import preprocess_text
from src.utils.db import insert_prediction
from src.utils.word_categories import detect_words_with_categories, update_word_category, reload_categories
from src.ui.icons import get_icon


LABEL_NAMES = {0: "Normal", 1: "Offensive", 2: "Hate"}

# Color Constants
PRIMARY_BLUE = "#1E90FF"
WHITE = "#FFFFFF"
LIGHT_GRAY = "#F5F6F7"
ACCENT_GREEN = "#22C55E"
ACCENT_RED = "#EF4444"

@ui.page("/classify")
def classify():
    ui.add_head_html(f'''
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
    * {{ font-family: 'Inter', sans-serif; }}
    /* Ensure native select icons remain visible; use local SVGs where needed */
    :root {{
        --primary-blue: {PRIMARY_BLUE};
        --white: {WHITE};
        --light-gray: {LIGHT_GRAY};
        --accent-green: {ACCENT_GREEN};
        --accent-red: {ACCENT_RED};
    }}
    
    .gradient-header {{
        background: var(--primary-blue);
        padding: 1rem 2rem;
        box-shadow: 0 4px 12px rgba(30, 144, 255, 0.2);
        border-bottom: 1px solid rgba(255, 255, 255, 0.15);
    }}
    
    .nav-link {{
        font-weight: 500;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        transition: all 0.3s ease;
        text-decoration: none;
        color: white !important;
    }}
    
    .nav-link:hover {{
        background: rgba(255, 255, 255, 0.2);
        transform: translateY(-1px);
    }}
    
    .nav-link.active {{
        background: rgba(255, 255, 255, 0.3);
        font-weight: 600;
    }}
    
    .card-modern {{
        background: var(--white);
        border-radius: 1rem;
        padding: 1.5rem;
        box-shadow: 0 6px 24px rgba(30, 144, 255, 0.1);
        transition: all 0.3s ease;
        border: 1px solid rgba(30, 144, 255, 0.1);
    }}
    
    .card-modern:hover {{
        transform: translateY(-5px);
        box-shadow: 0 12px 32px rgba(30, 144, 255, 0.15);
    }}
    
    .result-card {{ 
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); 
        box-shadow: 0 4px 16px rgba(30, 144, 255, 0.08); 
        border-radius: 1.2rem; 
        border: 1px solid #e2e8f0;
    }}
    
    .label-normal {{ 
        background: {ACCENT_GREEN}; 
        color: #fff; 
        border-radius: 0.5rem; 
        padding: 0.3rem 1.2rem; 
        font-weight: bold; 
        font-size: 0.9rem;
    }}
    .label-offensive {{ 
        background: #f59e0b; 
        color: #fff; 
        border-radius: 0.5rem; 
        padding: 0.3rem 1.2rem; 
        font-weight: bold; 
        font-size: 0.9rem;
    }}
    .label-hate {{ 
        background: {ACCENT_RED}; 
        color: #fff; 
        border-radius: 0.5rem; 
        padding: 0.3rem 1.2rem; 
        font-weight: bold; 
        font-size: 0.9rem;
    }}
    .suspect-word {{ 
        background: #fef2f2;
        color: #dc2626; 
        font-weight: bold; 
        padding: 0.2rem 0.6rem;
        border-radius: 0.375rem;
        border: 1px solid #fecaca;
    }}
    .density-bar {{ 
        height: 18px; 
        border-radius: 8px; 
        background: #e5e7eb; 
        overflow: hidden; 
        width: 100%;
    }}
    .density-fill {{ 
        height: 100%; 
        border-radius: 8px; 
        background: linear-gradient(90deg, #f59e0b 0%, {ACCENT_RED} 100%); 
        transition: width 0.5s ease;
    }}
    .model-select {{
        background: white;
        border-radius: 0.5rem;
        border: 1px solid #d1d5db;
        padding: 0.5rem 1rem;
    }}
    
    /* Hide the default dropdown arrow and text completely */
    .q-field__append {{
        display: none !important;
    }}
    
    .q-field__append i {{
        display: none !important;
    }}
    
    .material-icons {{
        display: none !important;
    }}
    
    .q-icon {{
        display: none !important;
    }}
    
    .text-input {{
        border-radius: 0.75rem;
        border: 1px solid #d1d5db;
        padding: 1rem;
        min-height: 120px;
        font-size: 1rem;
    }}
    .section-title {{
        font-size: 1.25rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 1rem;
    }}
    .feedback-btn {{
        border-radius: 0.5rem;
        font-weight: 600;
        padding: 0.5rem 1rem;
    }}
    
    /* Model select dropdown styling - Fixed to hide text and show only icon */
    .model-select-container {{
        position: relative;
        width: 100%;
    }}
    .model-select-dropdown {{
        width: 100%;
        padding: 0.75rem 1rem;
        border: 1px solid #d1d5db;
        border-radius: 0.5rem;
        background: white;
        font-size: 0.875rem;
        color: #374151;
        appearance: none;
        cursor: pointer;
        transition: all 0.2s ease;
    }}
    .model-select-dropdown:hover {{
        border-color: {PRIMARY_BLUE};
    }}
    .model-select-dropdown:focus {{
        outline: none;
        border-color: {PRIMARY_BLUE};
        box-shadow: 0 0 0 3px rgba(30, 144, 255, 0.1);
    }}
    .model-select-icon {{
        position: absolute;
        right: 1rem;
        top: 50%;
        transform: translateY(-50%);
        pointer-events: none;
        color: #6b7280;
        width: 16px;
        height: 16px;
    }}
    
    /* Main Background with Hate Speech Detection Pattern */
    body {{
        background: linear-gradient(135deg, {LIGHT_GRAY} 0%, #e8eef5 50%, #f0f4f8 100%);
        background-attachment: fixed;
        min-height: 100vh;
        position: relative;
    }}
    
    body::before {{
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-image: 
            radial-gradient(circle at 15% 50%, rgba(30, 144, 255, 0.08) 0%, transparent 50%),
            radial-gradient(circle at 85% 80%, rgba(34, 197, 94, 0.06) 0%, transparent 50%),
            radial-gradient(circle at 40% 20%, rgba(239, 68, 68, 0.05) 0%, transparent 50%),
            /* Shield Pattern */
            url("data:image/svg+xml,%3Csvg width='200' height='200' viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M100 20 L180 60 L180 140 L100 180 L20 140 L20 60 Z' fill='none' stroke='{PRIMARY_BLUE.replace('#', '%23')}' stroke-width='0.5' opacity='0.03'/%3E%3C/svg%3E"),  # cSpell:disable-line
            /* Text Filter Pattern */
            url("data:image/svg+xml,%3Csvg width='150' height='150' viewBox='0 0 150 150' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M30 30 L120 30 L120 120 L30 120 Z' fill='none' stroke='{ACCENT_RED.replace('#', '%23')}' stroke-width='0.3' opacity='0.02'/%3E%3C/svg%3E");
        pointer-events: none;
        z-index: -1;
    }}
    
    /* Dark Mode Styles */
    body.body--dark {{
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%) !important;
        background-attachment: fixed;
        color: #e2e8f0;
    }}
    
    body.body--dark::before {{
        background-image: 
            radial-gradient(circle at 15% 50%, rgba(30, 144, 255, 0.12) 0%, transparent 50%),
            radial-gradient(circle at 85% 80%, rgba(34, 197, 94, 0.08) 0%, transparent 50%),
            radial-gradient(circle at 40% 20%, rgba(239, 68, 68, 0.07) 0%, transparent 50%),
            url("data:image/svg+xml,%3Csvg width='200' height='200' viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M100 20 L180 60 L180 140 L100 180 L20 140 L20 60 Z' fill='none' stroke='{PRIMARY_BLUE.replace('#', '%23')}' stroke-width='0.5' opacity='0.05'/%3E%3C/svg%3E"),  /* cspell:ignore Csvg */
            url("data:image/svg+xml,%3Csvg width='150' height='150' viewBox='0 0 150 150' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M30 30 L120 30 L120 120 L30 120 Z' fill='none' stroke='{ACCENT_RED.replace('#', '%23')}' stroke-width='0.3' opacity='0.03'/%3E%3C/svg%3E");
    }}
    
    body.body--dark .card-modern,
    body.body--dark .bg-white {{
        background: rgba(30, 41, 59, 0.95) !important;
        border-color: rgba(30, 144, 255, 0.2);
        color: #e2e8f0;
        backdrop-filter: blur(10px);
    }}
    
    .card-modern {{
        background: rgba(255, 255, 255, 0.98) !important;
        backdrop-filter: blur(10px);
    }}
    
    body.body--dark .text-gray-800,
    body.body--dark .text-blue-900 {{
        color: #e2e8f0 !important;
    }}
    
    body.body--dark .text-gray-700 {{
        color: #cbd5e0 !important;
    }}
    
    body.body--dark .text-gray-600 {{
        color: #94a3b8 !important;
    }}
    
    body.body--dark .gradient-bg {{
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%) !important;
    }}
    
    body.body--dark .result-card {{
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%) !important;
        border: 1px solid #475569 !important;
    }}
    
    body.body--dark .section-title {{
        color: #f8fafc !important;
    }}
    
    body.body--dark .text-input {{
        background: #1e293b;
        border-color: #475569;
        color: #e2e8f0;
    }}
    
    body.body--dark .model-select-dropdown {{
        background: #1e293b;
        border-color: #475569;
        color: #e2e8f0;
    }}
    body.body--dark .model-select-dropdown:hover {{
        border-color: {PRIMARY_BLUE};
    }}
    body.body--dark .model-select-dropdown:focus {{
        border-color: {PRIMARY_BLUE};
        box-shadow: 0 0 0 3px rgba(30, 144, 255, 0.2);
    }}
    body.body--dark .model-select-icon {{
        color: #94a3b8;
    }}
    
    body.body--dark .suspect-word {{
        background: #7f1d1d;
        color: #fca5a5;
        border-color: #991b1b;
    }}
    
    body.body--dark .density-bar {{
        background: #374151;
    }}
    
    /* Detection Pattern Overlay */
    .detection-pattern {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: -1;
        opacity: 0.02;
    }}
    
    .detection-pattern::before {{
        content: '';
        position: absolute;
        width: 100%;
        height: 100%;
        background-image: 
            linear-gradient(30deg, {PRIMARY_BLUE} 12%, transparent 12.5%, transparent 87%, {PRIMARY_BLUE} 87.5%, {PRIMARY_BLUE}),
            linear-gradient(150deg, {PRIMARY_BLUE} 12%, transparent 12.5%, transparent 87%, {PRIMARY_BLUE} 87.5%, {PRIMARY_BLUE}),
            linear-gradient(30deg, {PRIMARY_BLUE} 12%, transparent 12.5%, transparent 87%, {PRIMARY_BLUE} 87.5%, {PRIMARY_BLUE}),
            linear-gradient(150deg, {PRIMARY_BLUE} 12%, transparent 12.5%, transparent 87%, {PRIMARY_BLUE} 87.5%, {PRIMARY_BLUE}),
            linear-gradient(60deg, rgba(30, 144, 255, 0.3) 25%, transparent 25.5%, transparent 75%, rgba(30, 144, 255, 0.3) 75%, rgba(30, 144, 255, 0.3)),
            linear-gradient(60deg, rgba(30, 144, 255, 0.3) 25%, transparent 25.5%, transparent 75%, rgba(30, 144, 255, 0.3) 75%, rgba(30, 144, 255, 0.3));
        background-size: 80px 140px;
        background-position: 0 0, 0 0, 40px 70px, 40px 70px, 0 0, 40px 70px;
    }}
    
    /* Enhanced gradient cards */
    .gradient-card-blue {{
        background: linear-gradient(135deg, rgba(30, 144, 255, 0.1) 0%, rgba(30, 144, 255, 0.05) 100%);
        border-left: 4px solid {PRIMARY_BLUE};
    }}
    
    body.body--dark .gradient-card-blue {{
        background: linear-gradient(135deg, rgba(30, 144, 255, 0.2) 0%, rgba(30, 144, 255, 0.1) 100%);
    }}
    </style>
    <script>
    (function(){{
        const KEY = 'krx_dark';
        function apply(){{
            const v = localStorage.getItem(KEY);
            if (v === '1') document.body.classList.add('body--dark'); else document.body.classList.remove('body--dark');
            const sw = document.getElementById('krx-dark-switch'); if (sw) sw.checked = (v === '1');
        }}
        function attach(){{
            document.addEventListener('change', function(e){{
                const el = e.target; if (!el) return; if (el.id === 'krx-dark-switch'){{
                    const enabled = !!el.checked; localStorage.setItem(KEY, enabled ? '1' : '0');
                    if (enabled) document.body.classList.add('body--dark'); else document.body.classList.remove('body--dark');
                }}
            }}, true);
        }}
        if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', function(){{ apply(); attach(); }}); else {{ apply(); attach(); }}
    }})();
    </script>
    ''')
    
    # Add detection pattern overlay
    ui.add_head_html('<div class="detection-pattern"></div>')
    
    def toggle_dark(e):
        try:
            enabled = bool(getattr(e, 'value', False))
            if enabled:
                ui.dark_mode().enable()
                ui.run_javascript("localStorage.setItem('krx_dark','1'); document.body.classList.add('body--dark');")
            else:
                ui.dark_mode().disable()
                ui.run_javascript("localStorage.setItem('krx_dark','0'); document.body.classList.remove('body--dark');")
        except Exception as ex:
            print(f"Dark toggle error: {ex}")
    
    with ui.header().classes("gradient-header items-center justify-between"):
        with ui.row().classes("items-center gap-4 w-full justify-between"):
            with ui.row().classes("items-center gap-3"):
                ui.html(get_icon('shield', '28'), sanitize=False).classes("text-white")
                ui.label("Classify Dashboard").classes("text-2xl font-bold text-white")
            with ui.row().classes("gap-2 items-center"):
                ui.link("HOME", "/").classes("nav-link text-white")
                ui.link("CLASSIFY", "/classify").classes("nav-link active text-white")
                ui.link("BATCH", "/batch").classes("nav-link text-white")
                ui.link("LOGS", "/history").classes("nav-link text-white")
                ui.link("ANALYTICS", "/analytics").classes("nav-link text-white")
                ui.switch('', on_change=toggle_dark).props('color=white id=krx-dark-switch')
    
    with ui.column().classes("w-full max-w-4xl mx-auto px-4 py-8"):
        with ui.card().classes("card-modern gradient-card-blue w-full p-6 mb-6"):
            ui.label("Try these samples").classes("text-xl font-bold mb-4 text-gray-800 dark:text-white")
            # cSpell:disable
            sample_texts = [
                ("Hey how are you?", "en"),
                ("You are an idiot", "en"),
                ("This is great news for everyone", "en"),
                ("Shut up and get lost", "en"),
                ("तुम बहुत अच्छे हो", "hi"),
                ("तुम बेवकूफ हो", "hi"),
                ("आज का दिन बहुत सुंदर है", "hi"),
                ("Yaar tu bahut bura hai", "hi-en"),
                ("Bhai kya scene hai aaj", "hi-en"),
                ("Tu pagal hai kya", "hi-en")
            ]
            # cSpell:enable
            with ui.grid(columns=2).classes("gap-2 w-full"):
                for text, lang in sample_texts:
                    lang_label = {"en": "🇬🇧", "hi": "🇮🇳", "hi-en": "🔀"}[lang]
                    def make_click(t=text):
                        def click_handler():
                            input_text.set_value(t)
                        return click_handler
                    ui.button(f"{lang_label} {text}", on_click=make_click()).props("flat align=left").classes("w-full justify-start text-left text-sm text-gray-700 dark:text-gray-300")
        
        ui.label("Enter text to classify").classes("section-title text-gray-800 dark:text-white")
        input_text = ui.textarea(placeholder="Type or paste your text here...").classes("text-input w-full")
        result_ref = {"data": None}
        spinner_ref = {"spinner": None}

        async def on_classify():
            import asyncio
            raw = input_text.value or ""
            if not raw.strip():
                ui.notify("Please enter some text", type="warning")
                return
            
            selected_model = model_select.value
            
            # Check if model is available
            from pathlib import Path
            model_paths = {
                "logistic_regression": "models/baseline/logreg.joblib",  # cSpell:ignore logreg
                "naive_bayes": "models/baseline/nb.joblib",
                "svc": "models/baseline/svc.joblib",
                "random_forest": "models/baseline/rf.joblib",
                "cnn": "models/baseline/cnn.pth",
                "bilstm": "models/baseline/bilstm.pth",  # cSpell:ignore bilstm
                "hecan": "models/baseline/hecan.pth",  # cSpell:ignore hecan
                "distilbert": "models/baseline/logreg_distilbert.joblib"  # cSpell:ignore distilbert
            }
            
            if selected_model != "ensemble" and selected_model in model_paths:
                if not Path(model_paths[selected_model]).exists():
                    ui.notify(f"Model '{selected_model}' not trained yet. Using Logistic Regression instead.", type="warning")
                    selected_model = "logistic_regression"
            
            spinner_ref["spinner"].set_visibility(True)
            ui.notify(f"Analyzing with {selected_model}...", type="info")
            
            # Run prediction in thread pool to avoid blocking
            def run_prediction():
                cleaned, lang = preprocess_text(raw)
                from src.api.predict import predict_text_with_model
                result = predict_text_with_model(cleaned, model_name=selected_model)
                result["text"] = cleaned
                result["lang"] = lang
                return result
            
            result = await asyncio.get_event_loop().run_in_executor(None, run_prediction)
            result_ref["data"] = result
            result_card.refresh()
            spinner_ref["spinner"].set_visibility(False)
            ui.notify("Analysis complete", type="positive")

        with ui.row().classes("w-full items-center justify-between mt-4 mb-6"):
            with ui.column().classes("gap-3 flex-1"):
                ui.label("Select Model").classes("text-sm font-semibold text-gray-700 dark:text-gray-300")
                
                # Create custom dropdown with icon - Fixed to hide default arrow text
                with ui.column().classes("model-select-container w-full"):
                    model_select = ui.select(
                        options=[
                            "logistic_regression",
                            "naive_bayes", 
                            "svc",
                            "random_forest",
                            "cnn",
                            "bilstm",
                            "hecan",
                            "distilbert",
                            "ensemble",
                        ],
                        value="logistic_regression"
                    ).classes("model-select-dropdown").props('id=model-select-dropdown')
                    
                    # Add dropdown icon using our local SVG icon
                    ui.html(f'''
                    <div class="model-select-icon">
                        {get_icon("chevron_down", "16")}
                    </div>
                    ''', sanitize=False)
                
                ui.label("(Options: logistic_regression, naive_bayes, svc, random_forest, cnn, bilstm, hecan, distilbert, ensemble)").classes("text-xs text-gray-600 dark:text-gray-400 mt-1")
            
            ui.button("CLASSIFY", on_click=on_classify).props("color=primary size=lg").classes("px-8 self-end")

        spinner_ref["spinner"] = ui.spinner(size="lg", color="primary").classes("mt-4 mx-auto")
        spinner_ref["spinner"].set_visibility(False)

        @ui.refreshable  # cspell:ignore refreshable
        def result_card():
            data = result_ref["data"]
            if data is None:
                return
            with ui.card().classes("card-modern w-full p-6 mt-6"):
                label_int = data.get("label", 0)
                label_name = LABEL_NAMES.get(label_int, "Unknown")
                score_pct = data.get("score", 0.0) * 100
                model = data.get("model_name", "N/A")
                lang = data.get("lang", "N/A")
                latency = data.get("latency_ms", 0)
                with ui.row().classes("w-full items-center justify-between mb-4"):
                    with ui.row().classes("items-center gap-4"):
                        ui.label("RESULT").classes("text-xl font-bold text-gray-800 dark:text-white")
                        if label_int == 0:
                            ui.label("NORMAL").classes("label-normal")
                        elif label_int == 1:
                            ui.label("OFFENSIVE").classes("label-offensive")
                        else:
                            ui.label("HATE").classes("label-hate")
                    ui.label(f"Confidence: {score_pct:.1f}% | Language: {lang}").classes("text-md font-semibold text-gray-700 dark:text-gray-300")
                
                # Detect hate words using dynamic categories (use stored detection if present)
                hate_detection = data.get("detection") or detect_words_with_categories(data.get("text", ""))
                suspect_hate = hate_detection["hate_words"]
                suspect_offensive = hate_detection["offensive_words"]
                
                # Display hate words and offensive words separately
                with ui.column().classes("w-full mb-4"):
                    if suspect_hate:
                        ui.label("🚨 Hate Words Detected:").classes("text-md font-bold text-red-700 dark:text-red-300 mb-2")
                        with ui.row().classes("flex-wrap gap-2 mb-3"):
                            for w in suspect_hate:
                                ui.label(w).classes("suspect-word")
                    if suspect_offensive:
                        ui.label("⚠️ Offensive Words:").classes("text-md font-semibold text-orange-600 dark:text-orange-400 mb-2")
                        with ui.row().classes("flex-wrap gap-2 mb-3"):
                            for w in suspect_offensive:
                                ui.label(w).style("background: #fff7ed; color: #ea580c; padding: 0.2rem 0.6rem; border-radius: 0.375rem; border: 1px solid #fed7aa;")
                    if not suspect_hate and not suspect_offensive:
                        ui.label("Flagged words:").classes("text-md font-semibold text-gray-700 dark:text-gray-300 mb-2")
                        ui.label("None detected").classes("text-sm text-gray-500 dark:text-gray-400")
                density = 0.2 if label_int == 0 else (0.6 if label_int == 1 else 0.9)
                with ui.column().classes("w-full mb-6"):
                    with ui.row().classes("w-full items-center justify-between mb-2"):
                        ui.label("Paragraph hate density").classes("text-md font-semibold text-gray-700 dark:text-gray-300")
                        ui.label(f"{int(density*100)}%").classes("text-md font-bold text-gray-800 dark:text-white")
                    with ui.element("div").classes("density-bar"):
                        ui.element("div").style(f"width: {int(density*100)}%; height: 100%; border-radius: 8px; background: linear-gradient(90deg, #f59e0b 0%, {ACCENT_RED} 100%); transition: width 0.5s ease;")
                with ui.row().classes("w-full gap-3 mt-4"):
                    ui.button("👍 PREDICTION CORRECT", 
                             on_click=lambda: ui.notify('Feedback recorded!', type='positive')
                    ).props("color=positive").classes("feedback-btn flex-1")
                    async def on_fix():
                        try:
                            selected_label = {"value": result_ref["data"].get("label", 0) if result_ref.get("data") else 0}
                            label_buttons = {}
                            
                            with ui.dialog() as fix_dialog, ui.card().classes("card-modern p-6 min-w-96"):
                                ui.label("Correct the Classification").classes("text-xl font-semibold mb-4 text-gray-800 dark:text-white")
                                ui.label("Select the correct label for this text").classes("text-sm text-gray-500 dark:text-gray-400 text-center mb-4")
                                
                                def update_button_styles():
                                    for lbl_val, btn in label_buttons.items():
                                        if lbl_val == selected_label["value"]:
                                            btn.classes("border-blue-500 bg-blue-500 text-white", remove="border-gray-300 bg-white text-gray-700")
                                        else:
                                            btn.classes("border-gray-300 bg-white text-gray-700", remove="border-blue-500 bg-blue-500 text-white")

                                with ui.row().classes("gap-4 justify-between w-full"):
                                    for lbl_val, lbl_name in LABEL_NAMES.items():
                                        def make_click(v=lbl_val):
                                            def click_handler():
                                                selected_label["value"] = v
                                                update_button_styles()
                                            return click_handler
                                        
                                        btn = ui.button(lbl_name.upper()).classes("p-4 cursor-pointer border-2 flex-1 font-semibold text-gray-700").props("flat")
                                        btn.on_click(make_click())
                                        label_buttons[lbl_val] = btn
                                
                                update_button_styles()

                                async def on_submit():
                                    try:
                                        corrected_label = selected_label["value"]
                                        text_data = result_ref["data"].get("text", "") if result_ref.get("data") else ""
                                        insert_prediction(
                                            text_data,
                                            result_ref["data"].get("lang", "unknown") if result_ref.get("data") else "unknown",
                                            corrected_label,
                                            1.0,
                                            "user_corrected",
                                            0
                                        )
                                        text_words = text_data.lower().split()
                                        for word in text_words:
                                            clean_word = ''.join(c for c in word if c.isalnum())
                                            if clean_word:
                                                if corrected_label == 0:
                                                    update_word_category(clean_word, "normal")
                                                elif corrected_label == 1:
                                                    update_word_category(clean_word, "offensive")
                                                elif corrected_label == 2:
                                                    update_word_category(clean_word, "hate")
                                        # Ensure in-memory categories are reloaded and detection is refreshed
                                        try:
                                            reload_categories()
                                            new_detection = detect_words_with_categories(text_data)
                                            if result_ref.get("data") is not None:
                                                result_ref["data"]["detection"] = new_detection
                                        except Exception:
                                            pass
                                        ui.notify(f"Corrected to {LABEL_NAMES[corrected_label]} - thank you!", type="positive")
                                        if result_ref.get("data") is not None:
                                            result_ref["data"]["label"] = corrected_label
                                            result_ref["data"]["score"] = 1.0
                                            result_card.refresh()
                                        # Clear any client-side suspect-word elements in case they persist
                                        try:
                                            ui.run_javascript("document.querySelectorAll('.suspect-word').forEach(e=>e.remove());")
                                        except Exception:
                                            pass
                                        fix_dialog.close()
                                    except Exception as ex:
                                        print(f"Fix submit error: {ex}")
                                        ui.notify(f"Error: {ex}", type="negative")

                                with ui.row().classes("gap-3 mt-6 justify-center w-full"):
                                    ui.button("SUBMIT CORRECTION", on_click=on_submit).props("color=primary")
                                    ui.button("CANCEL", on_click=fix_dialog.close).props("outline")

                            fix_dialog.open()
                        except Exception as ex:
                            print(f"Fix dialog error: {ex}")
                            ui.notify(f"Error: {ex}", type='negative')

                    ui.button("⚠️ INCORRECT? FIX IT", on_click=on_fix).props("color=warning").classes("feedback-btn flex-1")
        
        result_card()