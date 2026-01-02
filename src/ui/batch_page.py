from nicegui import ui
import csv
import asyncio
from pathlib import Path
from src.api.predict import predict_text
from src.data.preprocess import preprocess_text


LABEL_NAMES = {0: "Normal", 1: "Offensive", 2: "Hate"}
BATCH_RESULTS_CSV = Path("data/batch_results.csv")
TRAINING_DATA_PATH = Path("data/raw/batch_training_data.csv")

# Color Constants
PRIMARY_BLUE = "#1E90FF"
WHITE = "#FFFFFF"
LIGHT_GRAY = "#F5F6F7"
ACCENT_GREEN = "#22C55E"
ACCENT_RED = "#EF4444"


def register_batch_page():
    @ui.page("/batch")
    def batch():
        results_ref = {"data": []}
        
        ui.add_head_html(f'''
        <style>
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
                    url("data:image/svg+xml,%3Csvg width='200' height='200' viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M100 20 L180 60 L180 140 L100 180 L20 140 L20 60 Z' fill='none' stroke='{PRIMARY_BLUE.replace('#', '%23')}' stroke-width='0.5' opacity='0.03'/%3E%3C/svg%3E"),
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
                    url("data:image/svg+xml,%3Csvg width='200' height='200' viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M100 20 L180 60 L180 140 L100 180 L20 140 L20 60 Z' fill='none' stroke='{PRIMARY_BLUE.replace('#', '%23')}' stroke-width='0.5' opacity='0.05'/%3E%3C/svg%3E"),
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
            
            /* Stats Cards */
            .stats-card {{
                background: linear-gradient(135deg, rgba(30, 144, 255, 0.1) 0%, rgba(30, 144, 255, 0.05) 100%);
                border-left: 4px solid {PRIMARY_BLUE};
            }}
            
            .stats-card-green {{
                background: linear-gradient(135deg, rgba(34, 197, 94, 0.1) 0%, rgba(34, 197, 94, 0.05) 100%);
                border-left: 4px solid {ACCENT_GREEN};
            }}
            
            .stats-card-purple {{
                background: linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(139, 92, 246, 0.05) 100%);
                border-left: 4px solid #8b5cf6;
            }}
            
            body.body--dark .stats-card {{
                background: linear-gradient(135deg, rgba(30, 144, 255, 0.2) 0%, rgba(30, 144, 255, 0.1) 100%);
            }}
            
            body.body--dark .stats-card-green {{
                background: linear-gradient(135deg, rgba(34, 197, 94, 0.2) 0%, rgba(34, 197, 94, 0.1) 100%);
            }}
            
            body.body--dark .stats-card-purple {{
                background: linear-gradient(135deg, rgba(139, 92, 246, 0.2) 0%, rgba(139, 92, 246, 0.1) 100%);
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
                    ui.icon("shield", size="1.8rem").classes("text-white")
                    ui.label("Batch Dashboard").classes("text-2xl font-bold text-white")
                with ui.row().classes("gap-2 items-center"):
                    ui.link("HOME", "/").classes("nav-link text-white")
                    ui.link("CLASSIFY", "/classify").classes("nav-link text-white")
                    ui.link("BATCH", "/batch").classes("nav-link active text-white")
                    ui.link("LOGS", "/history").classes("nav-link text-white")
                    ui.link("ANALYTICS", "/analytics").classes("nav-link text-white")
                    ui.switch("", on_change=toggle_dark).props("color=white icon=dark_mode id=krx-dark-switch")

        with ui.column().classes("w-full max-w-6xl mx-auto mt-8 px-4 gap-6"):
            with ui.row().classes("w-full gap-4 mb-4"):
                with ui.card().classes("card-modern stats-card flex-1 p-4 text-center"):
                    ui.icon("analytics", size="2rem").classes("text-blue-600 dark:text-blue-400 mb-2")
                    ui.label("Batch Prediction").classes("font-bold text-lg text-gray-800 dark:text-white")
                    ui.label("Process multiple texts").classes("text-sm text-gray-600 dark:text-gray-400")
                
                with ui.card().classes("card-modern stats-card-green flex-1 p-4 text-center"):
                    ui.icon("model_training", size="2rem").classes("text-green-600 dark:text-green-400 mb-2")
                    ui.label("Training Data").classes("font-bold text-lg text-gray-800 dark:text-white")
                    ui.label("Upload labeled data").classes("text-sm text-gray-600 dark:text-gray-400")
                
                with ui.card().classes("card-modern stats-card-purple flex-1 p-4 text-center"):
                    ui.icon("language", size="2rem").classes("text-purple-600 dark:text-purple-400 mb-2")
                    ui.label("Multi-Language").classes("font-bold text-lg text-gray-800 dark:text-white")
                    ui.label("Train new languages").classes("text-sm text-gray-600 dark:text-gray-400")
            
            with ui.card().classes("card-modern w-full p-6"):
                ui.label("📊 Batch Process CSV").classes("text-xl font-semibold mb-4 text-gray-800 dark:text-white")
                ui.label("Upload a CSV/TXT file with text samples for batch prediction").classes("text-sm text-gray-500 dark:text-gray-400 mb-2")
                ui.label("Format: One text per line (unlabeled data)").classes("text-sm text-gray-400 dark:text-gray-500 mb-4")
                
                async def handle_upload(e):
                    try:
                        content = e.content.read().decode('utf-8')
                        lines = content.strip().split('\n')
                        results = []
                        
                        ui.notify(f"Processing {len(lines)} texts...", type="info")
                        
                        for i, line in enumerate(lines):
                            if not line.strip():
                                continue
                            try:
                                cleaned, lang = preprocess_text(line.strip())
                                if cleaned.strip():
                                    result = predict_text(cleaned)
                                    result["text"] = cleaned
                                    result["lang"] = lang
                                    results.append(result)
                            except Exception as ex:
                                print(f"Error processing line {i}: {ex}")
                        
                        results_ref["data"] = results
                        results_container.refresh()
                        ui.notify(f"Processed {len(results)} texts successfully", type="positive")
                        
                        BATCH_RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
                        with open(BATCH_RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
                            writer = csv.writer(f)
                            writer.writerow(["text", "label", "label_name", "score", "model", "lang", "latency_ms"])
                            for r in results:
                                writer.writerow([
                                    r.get("text", ""),
                                    r.get("label", 0),
                                    LABEL_NAMES.get(r.get("label", 0), "Unknown"),
                                    r.get("score", 0.0),
                                    r.get("model_name", ""),
                                    r.get("lang", ""),
                                    r.get("latency_ms", 0)
                                ])
                        
                    except Exception as ex:
                        print(f"Upload error: {ex}")
                        ui.notify(f"Upload failed: {ex}", type="negative")
                
                ui.upload(on_upload=handle_upload, auto_upload=True).props("accept=.csv,.txt").classes("w-full")
            
            @ui.refreshable
            def results_container():
                if not results_ref["data"]:
                    return
                
                with ui.card().classes("card-modern w-full p-6 mt-4"):
                    ui.label(f"Results ({len(results_ref['data'])} items)").classes("text-xl font-semibold mb-4 text-gray-800 dark:text-white")
                    
                    columns = [
                        {"name": "text", "label": "Text", "field": "text", "align": "left"},
                        {"name": "label", "label": "Label", "field": "label", "align": "left"},
                        {"name": "score", "label": "Score", "field": "score", "align": "left"},
                        {"name": "lang", "label": "Lang", "field": "lang", "align": "left"},
                        {"name": "latency", "label": "Latency", "field": "latency", "align": "left"},
                    ]
                    
                    rows = []
                    for item in results_ref["data"]:
                        text_raw = item.get("text", "")
                        text_display = (text_raw[:50] + "...") if len(text_raw) > 50 else text_raw
                        label_int = item.get("label", 0)
                        label_name = LABEL_NAMES.get(label_int, "Unknown")
                        score_pct = f"{item.get('score', 0.0) * 100:.1f}%"
                        rows.append({
                            "text": text_display,
                            "label": label_name,
                            "score": score_pct,
                            "lang": item.get("lang", ""),
                            "latency": f"{item.get('latency_ms', 0)} ms",
                        })
                    
                    ui.table(columns=columns, rows=rows).classes("w-full")
                    
                    def download_results():
                        ui.notify(f"Results saved to {BATCH_RESULTS_CSV}", type="positive")
                    
                    ui.button("Download Results", on_click=download_results).props("color=primary outline")
            
            results_container()
            
            with ui.card().classes("card-modern w-full p-6 mt-6"):
                ui.label("🎓 Upload Training Data").classes("text-xl font-semibold mb-4 text-gray-800 dark:text-white")
                ui.label("Upload labeled CSV data to improve model accuracy for new languages").classes("text-sm text-gray-500 dark:text-gray-400 mb-2")
                ui.label("Format: text,label (where label: 0=Normal, 1=Offensive, 2=Hate)").classes("text-sm text-gray-400 dark:text-gray-500 mb-4")
                
                with ui.row().classes("gap-4 items-center mb-4"):
                    ui.label("Target Language:").classes("font-semibold text-gray-700 dark:text-gray-300")
                    lang_select = ui.select(
                        options=["English", "Hindi", "Hinglish", "Tamil", "Bengali", "Urdu", "Other"],
                        value="English"
                    ).classes("w-48")
                
                training_status_ref = {"text": ""}
                
                async def handle_training_upload(e):
                    try:
                        content = e.content.read().decode('utf-8')
                        lines = content.strip().split('\n')
                        
                        if len(lines) < 10:
                            ui.notify("Please upload at least 10 samples for training", type="warning")
                            return
                        
                        TRAINING_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
                        valid_samples = 0
                        
                        with open(TRAINING_DATA_PATH, "a", newline="", encoding="utf-8") as f:
                            writer = csv.writer(f)
                            for line in lines:
                                parts = line.strip().split(',', 1)
                                if len(parts) == 2:
                                    text, label = parts[0].strip(), parts[1].strip()
                                    if label in ['0', '1', '2']:
                                        writer.writerow([text, label])
                                        valid_samples += 1
                        
                        ui.notify(f"Added {valid_samples} training samples for {lang_select.value}", type="positive")
                        training_status_ref["text"] = f"✓ {valid_samples} samples added to {TRAINING_DATA_PATH}"
                        training_status.refresh()
                        
                    except Exception as ex:
                        print(f"Training upload error: {ex}")
                        ui.notify(f"Upload failed: {ex}", type="negative")
                
                ui.upload(on_upload=handle_training_upload, auto_upload=True).props("accept=.csv").classes("w-full")
                
                @ui.refreshable
                def training_status():
                    if training_status_ref["text"]:
                        ui.label(training_status_ref["text"]).classes("text-sm text-green-600 dark:text-green-400 font-semibold mt-2")
                
                training_status()
                
                with ui.row().classes("gap-4 mt-4"):
                    async def retrain_model():
                        try:
                            ui.notify("Starting model retraining... This may take a few minutes", type="info")
                            
                            proc = await asyncio.create_subprocess_shell(
                                "python -m src.data.load_data && python -m src.training.train_baseline",
                                stdout=asyncio.subprocess.PIPE,
                                stderr=asyncio.subprocess.PIPE
                            )
                            
                            stdout, stderr = await proc.communicate()
                            
                            if proc.returncode == 0:
                                ui.notify("Model retrained successfully! New model is now active.", type="positive")
                            else:
                                ui.notify(f"Retraining failed (exit {proc.returncode})", type="negative")
                                if stderr:
                                    print(f"Retrain error: {stderr.decode()}")
                        except Exception as ex:
                            print(f"Retrain exception: {ex}")
                            ui.notify(f"Retraining error: {ex}", type="negative")
                    
                    ui.button("🔄 Retrain Model", on_click=retrain_model).props("color=green")
                    ui.button("📈 View Training Stats", on_click=lambda: ui.navigate.to("/analytics")).props("outline")
                
                ui.separator().classes("my-4")
                
                ui.label("📚 Training Tips").classes("text-lg font-semibold mb-2 text-gray-800 dark:text-white")
                with ui.column().classes("gap-2 text-sm text-gray-600 dark:text-gray-400"):
                    ui.label("• Upload at least 50-100 samples per language for best results")
                    ui.label("• Balance your dataset: similar counts for Normal, Offensive, and Hate")
                    ui.label("• Include diverse examples: slang, formal, mixed scripts")
                    ui.label("• After uploading data, click 'Retrain Model' to update the classifier")
                    ui.label("• Monitor performance in Analytics page after retraining")