from nicegui import ui
import subprocess
import asyncio
from pathlib import Path
from src.ui.icons import get_icon


BATCH_UPLOAD_PATH = Path("data/batch_upload.csv")
LOGS_FILE = Path("logs/app.log")

# Color Constants (matching other pages)
PRIMARY_BLUE = "#1E90FF"
WHITE = "#FFFFFF"
LIGHT_GRAY = "#F5F6F7"
ACCENT_GREEN = "#22C55E"
ACCENT_RED = "#EF4444"


def _tail_logs(path: Path, lines: int = 100) -> str:
    if not path.exists():
        return "No logs available"
    try:
        with open(path, "r", encoding="utf-8") as f:
            all_lines = f.readlines()
            return "".join(all_lines[-lines:])
    except Exception as ex:
        return f"Error reading logs: {ex}"


def register_admin_page():
    @ui.page("/admin")
    def admin():
        train_state = {"running": False}

        ui.add_head_html(f'''
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
        <style>
            * {{ font-family: 'Inter', sans-serif; }}
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
            
            .section-title {{
                font-size: 1.5rem;
                font-weight: 700;
                color: #1f2937;
                margin-bottom: 1.5rem;
                border-left: 4px solid {PRIMARY_BLUE};
                padding-left: 1rem;
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
            
            body.body--dark .gradient-bg {{
                background: linear-gradient(135deg, #1e293b 0%, #334155 100%) !important;
            }}
            
            body.body--dark .section-title {{
                color: #f8fafc !important;
                border-left-color: {PRIMARY_BLUE};
            }}
            
            /* Enhanced gradient cards */
            .gradient-card-blue {{
                background: linear-gradient(135deg, rgba(30, 144, 255, 0.1) 0%, rgba(30, 144, 255, 0.05) 100%);
                border-left: 4px solid {PRIMARY_BLUE};
            }}
            
            .gradient-card-green {{
                background: linear-gradient(135deg, rgba(34, 197, 94, 0.1) 0%, rgba(34, 197, 94, 0.05) 100%);
                border-left: 4px solid {ACCENT_GREEN};
            }}
            
            .gradient-card-red {{
                background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(239, 68, 68, 0.05) 100%);
                border-left: 4px solid {ACCENT_RED};
            }}
            
            body.body--dark .gradient-card-blue {{
                background: linear-gradient(135deg, rgba(30, 144, 255, 0.2) 0%, rgba(30, 144, 255, 0.1) 100%);
            }}
            
            body.body--dark .gradient-card-green {{
                background: linear-gradient(135deg, rgba(34, 197, 94, 0.2) 0%, rgba(34, 197, 94, 0.1) 100%);
            }}
            
            body.body--dark .gradient-card-red {{
                background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(239, 68, 68, 0.1) 100%);
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
            
            .logs-container {{
                background: #1a1a1a !important;
                color: #00ff00 !important;
                font-family: 'Courier New', monospace !important;
                border-radius: 0.5rem;
                border: 1px solid #333;
            }}
            
            body.body--dark .logs-container {{
                background: #0a0a0a !important;
                border-color: #444;
            }}
            
            @keyframes fadeInUp {{
                from {{
                    opacity: 0;
                    transform: translateY(20px);
                }}
                to {{
                    opacity: 1;
                    transform: translateY(0);
                }}
            }}
            
            .fade-in-up {{
                animation: fadeInUp 0.6s ease-out;
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
                    ui.html(get_icon("settings", "28"), sanitize=False).classes("text-white")
                    ui.label("Admin Panel").classes("text-2xl font-bold text-white")
                with ui.row().classes("gap-2 items-center"):
                    ui.link("HOME", "/").classes("nav-link text-white")
                    ui.link("CLASSIFY", "/classify").classes("nav-link text-white")
                    ui.link("BATCH", "/batch").classes("nav-link text-white")
                    ui.link("LOGS", "/history").classes("nav-link text-white")
                    ui.link("ANALYTICS", "/analytics").classes("nav-link text-white")
                    ui.link("ADMIN", "/admin").classes("nav-link active text-white")
                    ui.switch('', on_change=toggle_dark).props('color=white id=krx-dark-switch')

        with ui.column().classes("w-full max-w-5xl mx-auto mt-8 gap-8 px-6 fade-in-up"):
            
            # Upload Dataset Section
            with ui.card().classes("card-modern gradient-card-blue w-full p-6"):
                with ui.row().classes("items-center gap-3 mb-4"):
                    with ui.element('div').classes('flex items-center'):
                        ui.html(get_icon("upload", "24"), sanitize=False).classes("text-blue-600 dark:text-blue-400")
                    ui.label("Upload Dataset").classes("text-xl font-semibold text-gray-800 dark:text-white")
                ui.label("CSV format: text,label (0=Normal, 1=Offensive, 2=Hate)").classes("text-sm text-gray-600 dark:text-gray-400 mb-4")
                
                async def handle_upload(e):
                    try:
                        BATCH_UPLOAD_PATH.parent.mkdir(parents=True, exist_ok=True)
                        content = e.content.read()
                        with open(BATCH_UPLOAD_PATH, "wb") as f:
                            f.write(content)
                        ui.notify(f"✅ Dataset uploaded to {BATCH_UPLOAD_PATH}", type="positive")
                    except Exception as ex:
                        print(f"Upload error: {ex}")
                        ui.notify(f"❌ Upload failed: {ex}", type="negative")
                
                ui.upload(
                    on_upload=handle_upload, 
                    auto_upload=True
                ).props("accept=.csv label='Choose CSV file'").classes("w-full")
                
                with ui.row().classes("mt-4 gap-2"):
                    ui.button("View Uploaded Data", on_click=lambda: ui.notify("Feature coming soon!", type="info")).props("outline")
                    ui.button("Clear Dataset", on_click=lambda: ui.notify("Feature coming soon!", type="info")).props("outline color=red")

            # Training Section
            with ui.card().classes("card-modern gradient-card-green w-full p-6"):
                with ui.row().classes("items-center gap-3 mb-4"):
                    with ui.element('div').classes('flex items-center'):
                        ui.html(get_icon("training", "24"), sanitize=False).classes("text-green-600 dark:text-green-400")
                    ui.label("Model Training").classes("text-xl font-semibold text-gray-800 dark:text-white")
                ui.label("Train transformer embeddings model on the uploaded dataset").classes("text-sm text-gray-600 dark:text-gray-400 mb-4")
                
                with ui.column().classes("gap-4"):
                    with ui.row().classes("items-center gap-4"):
                        ui.html(get_icon("info", "20"), sanitize=False).classes("text-blue-500")
                        ui.label("This will train the model using the latest uploaded dataset").classes("text-sm text-gray-600 dark:text-gray-400")
                    
                    async def run_training():
                        if train_state["running"]:
                            return
                        
                        train_state["running"] = True
                        train_btn.disable()
                        training_status.set_visibility(True)
                        ui.notify("🚀 Starting training process...", type="info")
                        
                        try:
                            proc = await asyncio.create_subprocess_shell(
                                "python -m src.training.train_transformer_embeddings",
                                stdout=asyncio.subprocess.PIPE,
                                stderr=asyncio.subprocess.PIPE
                            )
                            
                            stdout, stderr = await proc.communicate()
                            
                            if proc.returncode == 0:
                                ui.notify("✅ Training completed successfully!", type="positive")
                                if stdout:
                                    print(f"Training output: {stdout.decode()}")
                            else:
                                ui.notify(f"❌ Training failed (exit code {proc.returncode})", type="negative")
                                if stderr:
                                    print(f"Training error: {stderr.decode()}")
                        except Exception as ex:
                            print(f"Training exception: {ex}")
                            ui.notify(f"❌ Training error: {ex}", type="negative")
                        finally:
                            train_state["running"] = False
                            train_btn.enable()
                            training_status.set_visibility(False)
                    
                    train_btn = ui.button("Start Training", on_click=run_training).props("color=positive size=lg").classes("px-8")
                    training_status = ui.spinner(size="lg", color="positive").classes("mx-auto")
                    training_status.set_visibility(False)

            # Logs Section
            with ui.card().classes("card-modern gradient-card-red w-full p-6"):
                with ui.row().classes("items-center gap-3 mb-4"):
                    with ui.element('div').classes('flex items-center'):
                        ui.html(get_icon("report", "24"), sanitize=False).classes("text-red-600 dark:text-red-400")
                    ui.label("System Logs").classes("text-xl font-semibold text-gray-800 dark:text-white")
                ui.label("Real-time application logs and debugging information").classes("text-sm text-gray-600 dark:text-gray-400 mb-4")
                
                @ui.refreshable
                def logs_display():
                    logs_content = _tail_logs(LOGS_FILE, lines=50)
                    with ui.scroll_area().classes("w-full h-80 logs-container p-4 font-mono text-xs"):
                        ui.label(logs_content).classes("whitespace-pre-wrap text-green-400")
                
                logs_display()
                
                with ui.row().classes("gap-3 mt-4"):
                    def refresh_logs():
                        logs_display.refresh()
                        ui.notify("🔄 Logs refreshed", type="info")
                    
                    ui.button("Refresh Logs", on_click=refresh_logs).props("color=primary")
                    ui.button("Clear Logs", on_click=lambda: ui.notify("Feature coming soon!", type="info")).props("outline")
                    ui.button("Download Logs", on_click=lambda: ui.notify("Feature coming soon!", type="info")).props("outline")

            # System Status Section
            with ui.card().classes("card-modern w-full p-6"):
                with ui.row().classes("items-center gap-3 mb-4"):
                    with ui.element('div').classes('flex items-center'):
                        ui.html(get_icon("settings", "24"), sanitize=False).classes("text-purple-600 dark:text-purple-400")
                    ui.label("System Status").classes("text-xl font-semibold text-gray-800 dark:text-white")
                
                with ui.grid(columns=2).classes("gap-4 w-full"):
                    status_items = [
                        ("Database", "🟢 Online", "green"),
                        ("Models", "🟢 Loaded", "green"), 
                        ("API", "🟢 Running", "green"),
                        ("Storage", "🟢 Available", "green")
                    ]
                    
                    for name, status, color in status_items:
                        with ui.card().classes("p-4 bg-gray-50 dark:bg-gray-700/50 text-center"):
                            ui.label(name).classes("text-sm font-semibold text-gray-600 dark:text-gray-400 mb-2")
                            ui.label(status).classes(f"font-bold text-{color}-600 dark:text-{color}-400")

    # Hidden admin page (keeping the original functionality)
    @ui.page("/krixion-admin-secure")
    def admin_hidden():
        # This page uses the same styling as the main admin page
        admin()