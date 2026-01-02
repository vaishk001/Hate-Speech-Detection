from nicegui import ui  # cSpell:ignore nicegui  # cSpell:ignore nicegui
import csv
from pathlib import Path
from src.api.predict import predict_text
from src.data.preprocess import preprocess_text
from src.utils.db import insert_prediction
from src.utils.word_categories import detect_words_with_categories, update_word_category, get_word_category, reload_categories
from src.ui.icons import get_icon
from src.ui.mobile_styles import MOBILE_CSS


LABEL_NAMES = {0: "Normal", 1: "Offensive", 2: "Hate"}
FEEDBACK_CSV = Path("data/feedback.csv")

# Color Constants
PRIMARY_BLUE = "#1E90FF"
WHITE = "#FFFFFF"
LIGHT_GRAY = "#F5F6F7"
ACCENT_GREEN = "#22C55E"
ACCENT_RED = "#EF4444"


def _ensure_feedback_csv():
    if not FEEDBACK_CSV.exists():
        FEEDBACK_CSV.parent.mkdir(parents=True, exist_ok=True)
        with open(FEEDBACK_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["text", "label", "source"])


def _write_feedback(text: str, label: int, source: str):
    _ensure_feedback_csv()
    with open(FEEDBACK_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([text, label, source])


def _extract_suspect_words(text: str) -> list:
    suspect_set = {
        "hate", "kill", "idiot", "stupid", "moron", "fool", "loser", "ugly", "trash", "scum",
        "racist", "sexist", "abuse", "bully", "attack", "damn", "hell", "crap", "suck", "worst"
    }
    words = text.lower().split()
    return [w for w in words if w in suspect_set]


def _get_top_features(text: str, label: int) -> list:
    try:
        from pathlib import Path
        import joblib
        model_path = Path("models/baseline/logreg.joblib")  # cSpell:ignore logreg
        if not model_path.exists():
            return []
        model = joblib.load(str(model_path))
        vectorizer = model.named_steps.get('tfidf') or model.named_steps.get('vect')  # cSpell:ignore vectorizer vect tfidf
        classifier = model.named_steps.get('clf') or model.named_steps.get('classifier')
        if not vectorizer or not classifier:
            return []
        vec = vectorizer.transform([text])
        feature_names = vectorizer.get_feature_names_out()
        if hasattr(classifier, 'coef_'):
            if label < len(classifier.coef_):
                coef = classifier.coef_[label]
                indices = vec.toarray()[0].nonzero()[0]  # cSpell:ignore toarray
                if len(indices) > 0:
                    scores = [(feature_names[i], coef[i] * vec[0, i]) for i in indices]
                    top = sorted(scores, key=lambda x: abs(x[1]), reverse=True)[:3]
                    return [f"{word} ({score:.2f})" for word, score in top]
        return []
    except Exception:
        return []


def register_home_page():
    @ui.page("/")
    async def home():
        input_text_ref = {"value": ""}
        result_data_ref = {"data": None}
        detect_btn_ref = {"btn": None}

        ui.add_head_html(f'''
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap" rel="stylesheet">
        <style>
            * {{  /* cSpell:ignore Segoe */
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            }}
            .q-icon {{ font-size: 0 !important; }}
            .q-icon::before {{ 
                content: '' !important;
                display: inline-block;
                width: 20px;
                height: 20px;
                background: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="%23666" stroke-width="2"><path d="m6 9 6 6 6-6"/></svg>') no-repeat center;
                background-size: contain;
            }}
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
                box-height: 73px;
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
            
            .hero-section {{
                background: linear-gradient(135deg, {PRIMARY_BLUE} 0%, #1C86EE 100%);
                padding: 4rem 2rem;
                border-radius: 1.5rem;
                margin-bottom: 2rem;
                box-shadow: 0 20px 40px rgba(30, 144, 255, 0.25);
                position: relative;
                overflow: hidden;
                    min-height: 470px;
                    height: 470px;
            }}
            
            .hero-section::before {{
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: url("data:image/svg+xml,%3Csvg width='120' height='120' viewBox='0 0 120 120' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.08'%3E%3Cpath d='M60 60v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");  # cSpell:disable-line
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
            
            .stat-number {{
                font-size: 2.5rem;
                font-weight: 800;
                color: var(--primary-blue);
            }}
            
            .stat-label {{
                color: #64748b;
                font-size: 0.875rem;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                font-weight: 600;
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
                    url("data:image/svg+xml,%3Csvg width='150' height='150' viewBox='0 0 150 150' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M30 30 L120 30 L120 120 L30 120 Z' fill='none' stroke='{ACCENT_RED.replace('#', '%23')}' stroke-width='0.3' opacity='0.02'/%3E%3C/svg%3E");  # cSpell:disable-line
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
                    url("data:image/svg+xml,%3Csvg width='200' height='200' viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M100 20 L180 60 L180 140 L100 180 L20 140 L20 60 Z' fill='none' stroke='{PRIMARY_BLUE.replace('#', '%23')}' stroke-width='0.5' opacity='0.05'/%3E%3C/svg%3E"),  # cSpell:disable-line
                    url("data:image/svg+xml,%3Csvg width='150' height='150' viewBox='0 0 150 150' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M30 30 L120 30 L120 120 L30 120 Z' fill='none' stroke='{ACCENT_RED.replace('#', '%23')}' stroke-width='0.3' opacity='0.03'/%3E%3C/svg%3E");  # cSpell:disable-line
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
            
            body.body--dark .stat-label {{
                color: #cbd5e0 !important;
            }}
            
            /* Enhanced gradient cards with new color scheme */
            .gradient-card-1 {{
                background: linear-gradient(135deg, rgba(30, 144, 255, 0.1) 0%, rgba(30, 144, 255, 0.05) 100%);
                border-left: 4px solid {PRIMARY_BLUE};
            }}
            
            .gradient-card-2 {{
                background: linear-gradient(135deg, rgba(34, 197, 94, 0.1) 0%, rgba(34, 197, 94, 0.05) 100%);
                border-left: 4px solid {ACCENT_GREEN};
            }}
            
            .gradient-card-3 {{
                background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(239, 68, 68, 0.05) 100%);
                border-left: 4px solid {ACCENT_RED};
            }}
            
            .gradient-card-4 {{
                background: linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(139, 92, 246, 0.05) 100%);
                border-left: 4px solid #8b5cf6;
            }}
            
            body.body--dark .gradient-card-1 {{
                background: linear-gradient(135deg, rgba(30, 144, 255, 0.2) 0%, rgba(30, 144, 255, 0.1) 100%);
            }}
            
            body.body--dark .gradient-card-2 {{
                background: linear-gradient(135deg, rgba(34, 197, 94, 0.2) 0%, rgba(34, 197, 94, 0.1) 100%);
            }}
            
            body.body--dark .gradient-card-3 {{
                background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(239, 68, 68, 0.1) 100%);
            }}
            
            body.body--dark .gradient-card-4 {{
                background: linear-gradient(135deg, rgba(139, 92, 246, 0.2) 0%, rgba(139, 92, 246, 0.1) 100%);
            }}
            
            .typing-animation {{
                overflow: hidden;
                border-right: 3px solid #fff;
                white-space: nowrap;
                animation: typing 7s steps(22) infinite, blink 0.75s step-end infinite;
                display: inline-block;
                max-width: fit-content;
            }}
            
            @keyframes typing {{
                0%, 10% {{ max-width: 0; }}
                40%, 60% {{ max-width: 100%; }}
                90%, 100% {{ max-width: 0; }}
            }}
            
            @keyframes blink {{
                50% {{ border-color: transparent; }}
            }}
            
            .floating-shapes {{
                position: absolute;
                width: 100%;
                height: 100%;
                top: 0;
                left: 0;
                overflow: hidden;
                z-index: 0;
            }}
            
            .floating-shapes div {{
                position: absolute;
                background: rgba(255, 255, 255, 0.15);
                border-radius: 50%;
                animation: float 15s infinite linear;
            }}
            
            .floating-shapes div:nth-child(1) {{
                width: 80px;
                height: 80px;
                top: 20%;
                left: 10%;
                animation-delay: 0s;
            }}
            
            .floating-shapes div:nth-child(2) {{
                width: 120px;
                height: 120px;
                top: 60%;
                left: 80%;
                animation-delay: -5s;
            }}
            
            .floating-shapes div:nth-child(3) {{
                width: 60px;
                height: 60px;
                top: 80%;
                left: 20%;
                animation-delay: -10s;
            }}
            
            @keyframes float {{
                0% {{ transform: translateY(0) rotate(0deg); }}
                50% {{ transform: translateY(-20px) rotate(180deg); }}
                100% {{ transform: translateY(0) rotate(360deg); }}
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
            
            {MOBILE_CSS}
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
                    ui.html(get_icon("shield", "28"), sanitize=False).classes("text-white")
                    ui.label("Hate Speech Detection").classes("text-2xl font-bold text-white")
                with ui.row().classes("gap-2 items-center"):
                    ui.link("HOME", "/").classes("nav-link text-white active")
                    ui.link("CLASSIFY", "/classify").classes("nav-link text-white")
                    ui.link("BATCH", "/batch").classes("nav-link text-white")
                    ui.link("LOGS", "/history").classes("nav-link text-white")
                    ui.link("ANALYTICS", "/analytics").classes("nav-link text-white")
                    ui.switch('', on_change=toggle_dark).props('color=white id=krx-dark-switch')

        # Enhanced Hero Section
        with ui.column().classes("w-[90%] mx-auto mt-8 px-4 mb-12"):
            with ui.card().classes("hero-section w-full items-center justify-center text-center relative overflow-hidden"):
                # Floating shapes background
                with ui.element('div').classes('floating-shapes'):
                    ui.element('div')
                    ui.element('div')
                    ui.element('div')
                
                with ui.column().classes("relative z-10 items-center justify-center gap-4"):
                    with ui.row().classes("items-center justify-center gap-3 mb-4"):
                        ui.html(get_icon("shield", "56"), sanitize=False).classes("text-white")
                    ui.label("Hate Speech Detection").classes("text-4xl font-black text-white typing-animation mb-2")
                    ui.label("Multilingual • Offline • Self-Learning").classes("text-xl font-semibold text-blue-100 mb-1")
                    ui.label("English • Hindi • Hinglish • More").classes("text-lg text-blue-200")  # cSpell:ignore Hinglish

        # About this project section
        with ui.column().classes("w-3/4 mx-auto my-3 gap-8"):
            ui.label("About this project").classes("text-3xl font-bold text-center text-gray-800 dark:text-white")
            
            with ui.grid(columns=2).classes("gap-6 w-full"):
                cards_data = [
                    ("info", "What is it?", "An offline, multilingual hate speech detection system combining classic ML, deep learning and ensemble methods for best accuracy.", "gradient-card-1"),
                    ("settings", "How it works", "Input → preprocess → TF-IDF or embeddings → selected model → explanation. Self-learning model adapts instantly from feedback.", "gradient-card-2"),
                    ("favorite", "Why it helps", "Reduces false positives/negatives, handles Hinglish, provides token highlights and runs fully offline for privacy.", "gradient-card-3"),
                    ("people", "Who benefits", "Moderation teams, community managers, researchers needing fast, transparent, privacy-first moderation.", "gradient-card-4")
                ]
                
                for icon_name, title, description, gradient_class in cards_data:
                    with ui.card().classes(f"card-modern {gradient_class} p-6 flex-1 min-h-40"):
                        with ui.row().classes("items-center gap-4 mb-4"):
                            ui.html(get_icon(icon_name, "32"), sanitize=False).classes("text-gray-700 dark:text-white")
                            ui.label(title).classes("font-bold text-xl text-gray-800 dark:text-white")
                        ui.label(description).classes("text-gray-600 dark:text-gray-300 leading-relaxed")

        # Stats Section
        from src.utils.db import fetch_history
        history_data = fetch_history(limit=1000)
        total_searches = len(history_data)
        total_words = sum(len(str(r.get("text", "")).split()) for r in history_data)
        
        with ui.row().classes("w-full max-w-6xl mx-auto gap-6 mb-12 justify-center"):
            stats = [
                (str(total_searches), "Total Searches"),
                (str(total_words), "Words Scanned"),
                ("9 Models", "Available"),
                ("3 Languages", "Supported")
            ]
            
            for value, label in stats:
                with ui.card().classes("card-modern text-center px-8 py-6 min-w-48"):
                    ui.label(value).classes("stat-number mb-2")
                    ui.label(label).classes("stat-label")

        # How it works section
        with ui.card().classes("w-5/6 mx-auto p-8 mb-12 card-modern"):
            ui.label("How it works").classes("text-3xl font-bold text-center mb-8 text-gray-800 dark:text-white")
            
            steps = [
                ("Ingest & Preprocess", "Language detection, normalization, transliteration, tokenization", "filter"),
                ("Feature/Embedding", "TF-IDF, transformer embeddings, paragraph density", "database"),
                ("Prediction", "Select Logistic Regression / SVM / Naive Bayes / Random Forest / CNN / BiLSTM / Ensemble", "brain"),
                ("Online Learning", "Correct mistakes; model updates instantly using SGD (HashingVectorizer)", "school"),  # cSpell:ignore HashingVectorizer
                ("Explainability", "Token highlights, top features, confidence scores", "visibility"),
            ]
            
            expanded = [False] * len(steps)
            
            def toggle_step(idx):
                expanded[idx] = not expanded[idx]
                step_cards.refresh()
                
            @ui.refreshable  # cSpell:ignore refreshable
            def step_cards():
                for i, (title, desc, icon_name) in enumerate(steps):
                    with ui.card().classes(f"w-full mb-4 p-5 transition-all duration-300 {'bg-blue-50 dark:bg-blue-900/20 border-l-4 border-blue-500' if expanded[i] else 'bg-gray-50 dark:bg-gray-700/50'}"):
                        with ui.row().classes("items-center justify-between"):
                            with ui.row().classes("items-center gap-4"):
                                ui.html(get_icon(icon_name, "28"), sanitize=False).classes("text-blue-600 dark:text-blue-400")
                                ui.label(f"{i+1}. {title}").classes("font-semibold text-lg text-gray-800 dark:text-white")
                            ui.button("Expand" if not expanded[i] else "Collapse", 
                                     on_click=lambda idx=i: toggle_step(idx)).props("size=sm outline color=primary")
                        if expanded[i]:
                            ui.label(desc).classes("text-gray-700 dark:text-gray-300 mt-4 text-md pl-12")

            step_cards()

        # Feature Grid Section
        with ui.column().classes("w-3/4 mx-auto mb-8 gap-9"):
            features = [
                ("star", "Key features", "Baselines: LR/SVM/NB/RF • Deep Learning: CNN, BiLSTM • Ensemble for best accuracy • Token highlights & paragraph density • CSV export & logs • Dashboard with live counters & charts", "blue"),
                ("security", "Data & privacy", "Runs 100% locally. No data leaves your machine. SQLite logs remain on-device.", "green"),
                ("check_circle", "How it prevents abusive language", "Real-time detection with clear labels and confidence • Token highlights reveal problematic words • Thresholds + online learning reduce false alarms over time • Feedback loop: corrections adapt the model instantly to your community", "red"),
                ("groups", "Where this model is useful", "Community platforms: flag toxic replies before publishing • Customer support: triage abusive tickets • Edu/corp tools: keep chats respectful • Research: label datasets and study toxicity trends", "purple")
            ]
            
            with ui.grid(columns=2).classes("gap-6 w-full"):
                for icon_name, title, description, color in features:
                    with ui.card().classes(f"card-modern p-6 flex-1 min-h-48 border-l-4 border-{color}-500"):
                        with ui.row().classes("items-center gap-4 mb-4"):
                            with ui.element("div").classes(f"p-3 bg-{color}-100 dark:bg-{color}-900/30 rounded-full"):
                                ui.html(get_icon(icon_name, "28"), sanitize=False).classes(f"text-{color}-600 dark:text-{color}-400")
                            ui.label(title).classes("font-bold text-xl text-gray-800 dark:text-white")
                        ui.label(description).classes("text-gray-600 dark:text-gray-300 leading-relaxed")

        # Try these samples section
        # cSpell:disable
        sample_texts = [
            ("Hey how are you?", "en"),
            ("You are an idiot", "en"),
            ("This is great news for everyone", "en"),
            ("Shut up and get lost", "en"),
            ("We support our team wholeheartedly", "en"),
            ("Go to hell", "en"),
            ("Let's meet tomorrow at 5pm", "en"),
            ("You are so stupid", "en"),
            ("Have a wonderful day", "en"),
            ("I hate you", "en"),
            ("तुम बहुत अच्छे हो", "hi"),
            ("तुम बेवकूफ हो", "hi"),
            ("आज का दिन बहुत सुंदर है", "hi"),
            ("तुम्हें शर्म आनी चाहिए", "hi"),
            ("Yaar tu bahut bura hai", "hi-en"),
            ("Bhai kya scene hai aaj", "hi-en"),
            ("Tu pagal hai kya", "hi-en"),
            ("Tere se baat nahi karni", "hi-en")
        ]
        # cSpell:enable
        
        with ui.card().classes("w-[97%] mx-auto p-8 mb-4 card-modern"):
            ui.label("Try these samples").classes("text-2xl font-bold text-center mb-6 text-gray-800 dark:text-white")
            with ui.grid(columns=2).classes("gap-4 w-full"):
                for text, lang in sample_texts:
                    lang_emoji = {"en": "🇬🇧", "hi": "🇮🇳", "hi-en": "🔀"}[lang]
                    with ui.card().classes("p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg border border-gray-200 dark:border-gray-600 hover:shadow-lg transition-shadow cursor-pointer").on("click", lambda t=text: input_text_ref["value"].set_value(t)):
                        with ui.row().classes("items-center justify-between"):
                            ui.button("COPY", on_click=lambda t=text: input_text_ref["value"].set_value(t)).props("size=sm flat color=primary")
                            ui.label(f"{lang_emoji} {text}").classes("text-sm text-gray-700 dark:text-gray-300 flex-1 text-center")

        # Detection Input Section
        with ui.column().classes("w-1/3 mx-auto mt-8 px-4 gap-6"):
            with ui.card().classes("w-full p-8 card-modern"):
                ui.label("Enter Text to Analyze").classes("text-2xl font-semibold mb-4 text-gray-800 dark:text-white")
                
                input_area = ui.textarea(placeholder="Type or paste text here (max 500 chars)...").props("maxlength=500 autogrow outlined").classes("w-full text-lg")  # cSpell:ignore autogrow
                input_text_ref["value"] = input_area

                spinner_ref = {"spinner": None}
                with ui.row().classes("gap-3 mt-6 justify-center"):
                    async def on_detect():
                        try:
                            raw = input_area.value or ""
                            if not raw.strip():
                                ui.notify("Please enter some text", type="warning")
                                return
                            detect_btn_ref["btn"].disable()
                            if spinner_ref["spinner"]:
                                spinner_ref["spinner"].set_visibility(True)
                            ui.notify("Analyzing...", type="info")
                            cleaned, lang = preprocess_text(raw)
                            if not cleaned.strip():
                                ui.notify("Text too short after cleaning", type="warning")
                                detect_btn_ref["btn"].enable()
                                if spinner_ref["spinner"]:
                                    spinner_ref["spinner"].set_visibility(False)
                                return
                            result = predict_text(cleaned)
                            result["text"] = cleaned
                            result["lang"] = lang
                            result_data_ref["data"] = result
                            result_container.refresh()
                            ui.notify("Analysis complete", type="positive")
                        except Exception as ex:
                            print(f"Detect error: {ex}")
                            ui.notify(f"Error: {ex}", type="negative")
                        finally:
                            detect_btn_ref["btn"].enable()
                            if spinner_ref["spinner"]:
                                spinner_ref["spinner"].set_visibility(False)

                    detect_btn = ui.button("Detect Hate Speech", on_click=on_detect).props("color=primary size=lg").classes("px-8")
                    detect_btn_ref["btn"] = detect_btn
                    ui.button("Clear Text", on_click=lambda: input_area.set_value("")).props("outline size=lg").classes("px-8")
                
                spinner_ref["spinner"] = ui.spinner(size="lg", color="primary").classes("mt-4 mx-auto")
                spinner_ref["spinner"].set_visibility(False)

            @ui.refreshable
            def result_container():
                data = result_data_ref["data"]
                if data is None:
                    return
                    
                with ui.card().classes("w-full p-8 card-modern mt-6"):
                    ui.label("Detection Result").classes("text-2xl font-semibold mb-6 text-gray-800 dark:text-white")
                    
                    label_int = data.get("label", 0)
                    label_name = LABEL_NAMES.get(label_int, "Unknown")
                    score_pct = data.get("score", 0.0) * 100
                    model = data.get("model_name", "N/A")
                    lang = data.get("lang", "N/A")
                    latency = data.get("latency_ms", 0)
                    
                    # Result badge with color coding
                    label_color = {
                        0: "positive", 
                        1: "warning", 
                        2: "negative"
                    }.get(label_int, "primary")
                    
                    with ui.row().classes("items-center justify-between mb-6 p-4 bg-gray-50 dark:bg-gray-700/50 rounded-lg"):
                        with ui.column().classes("items-start"):
                            ui.label("Classification").classes("text-sm text-gray-500")
                            ui.label(label_name).classes("text-2xl font-bold")
                        ui.label(f"{score_pct:.1f}%").classes("text-3xl font-black").props(f"color={label_color}")

                    with ui.grid(columns=3).classes("gap-4 mb-6 w-full"):
                        info_items = [
                            ("Model Used", model, "settings"),
                            ("Detected Language", lang, "language"),
                            ("Processing Time", f"{latency} ms", "schedule")
                        ]
                        
                        for title, value, icon_name in info_items:
                            with ui.card().classes("p-4 bg-gray-50 dark:bg-gray-700/50 text-center"):
                                ui.html(get_icon(icon_name, "24"), sanitize=False).classes("text-blue-500 mb-2")
                                ui.label(title).classes("text-sm text-gray-500 mb-1")
                                ui.label(value).classes("font-semibold text-gray-700 dark:text-gray-300")

                    # Additional insights with dynamic word categories
                    word_detection = detect_words_with_categories(data.get("text", ""))
                    if word_detection["hate_words"]:
                        with ui.card().classes("p-4 bg-red-50 dark:bg-red-900/20 border-l-4 border-red-500"):
                            ui.label("Hate Words").classes("text-sm font-semibold text-red-700 dark:text-red-300 mb-2")
                            ui.label(", ".join(word_detection["hate_words"])).classes("text-red-600 dark:text-red-400 font-medium")
                    if word_detection["offensive_words"]:
                        with ui.card().classes("p-4 bg-orange-50 dark:bg-orange-900/20 border-l-4 border-orange-500"):
                            ui.label("Offensive Words").classes("text-sm font-semibold text-orange-700 dark:text-orange-300 mb-2")
                            ui.label(", ".join(word_detection["offensive_words"])).classes("text-orange-600 dark:text-orange-400 font-medium")
                    
                    top_features = _get_top_features(data.get("text", ""), label_int)
                    if top_features:
                        with ui.card().classes("p-4 bg-blue-50 dark:bg-blue-900/20 border-l-4 border-blue-500"):
                            ui.label("Key Indicators").classes("text-sm font-semibold text-blue-700 dark:text-blue-300 mb-2")
                            ui.label(", ".join(top_features)).classes("text-blue-600 dark:text-blue-400 font-medium")

                    # Feedback buttons
                    with ui.row().classes("gap-3 mt-6 justify-center"):
                        async def on_incorrect():
                            try:
                                _write_feedback(data["text"], -1, "incorrect_flag")
                                ui.notify("Feedback recorded - thank you!", type="positive")
                            except Exception as ex:
                                print(f"Incorrect error: {ex}")
                                ui.notify(f"Error: {ex}", type="negative")
                                
                        def flag_click():
                            on_incorrect()
                        ui.button("Flag Incorrect", on_click=flag_click).props("outline color=red")
                        
                        async def on_fix():
                            try:
                                selected_label = {"value": label_int}
                                label_buttons = {}
                                
                                with ui.dialog() as fix_dialog, ui.card().classes("p-6 card-modern min-w-96"):
                                    ui.label("Correct the Classification").classes("text-xl font-semibold mb-4 text-gray-800 dark:text-white")
                                    ui.label("Select the correct label for this text").classes("text-sm text-gray-500 text-center mb-4")
                                    
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
                                            _write_feedback(data["text"], corrected_label, "user_correction")
                                            insert_prediction(
                                                data["text"],
                                                data.get("lang", "unknown"),
                                                corrected_label,
                                                1.0,
                                                "user_corrected",
                                                0
                                            )
                                            text_words = data["text"].lower().split()
                                            for word in text_words:
                                                clean_word = ''.join(c for c in word if c.isalnum())
                                                if clean_word:
                                                    if corrected_label == 0:
                                                        update_word_category(clean_word, "normal")
                                                    elif corrected_label == 1:
                                                        update_word_category(clean_word, "offensive")
                                                    elif corrected_label == 2:
                                                        update_word_category(clean_word, "hate")
                                            reload_categories()
                                            result_data_ref["data"]["label"] = corrected_label
                                            result_data_ref["data"]["score"] = 1.0
                                            ui.notify(f"Corrected to {LABEL_NAMES[corrected_label]} - thank you!", type="positive")
                                            result_container.refresh()
                                            fix_dialog.close()
                                        except Exception as ex:
                                            print(f"Fix submit error: {ex}")
                                            ui.notify(f"Error: {ex}", type="negative")
                                            
                                    with ui.row().classes("gap-3 mt-6 justify-center w-full"):
                                        ui.button("Submit Correction", on_click=on_submit).props("color=primary")
                                        ui.button("Cancel", on_click=fix_dialog.close).props("outline")
                                            
                                fix_dialog.open()
                            except Exception as ex:
                                print(f"Fix dialog error: {ex}")
                                ui.notify(f"Error: {ex}", type="negative")
                                
                        ui.button("Correct Label", on_click=on_fix).props("outline color=blue")
                        
            result_container()