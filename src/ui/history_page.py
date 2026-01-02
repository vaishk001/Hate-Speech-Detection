from nicegui import ui
import csv
from pathlib import Path
from datetime import datetime, timedelta
from src.utils.db import fetch_history, delete_all_predictions
from src.ui.icons import get_icon


LABEL_NAMES = {0: "Normal", 1: "Offensive", 2: "Hate"}
EXPORT_PATH = Path("data/history_export.csv")

# Color Constants
PRIMARY_BLUE = "#1E90FF"
WHITE = "#FFFFFF"
LIGHT_GRAY = "#F5F6F7"
ACCENT_GREEN = "#22C55E"
ACCENT_RED = "#EF4444"


def get_latest_predictions(all_data):
    """Return only the latest prediction for each unique text."""
    seen = {}
    for item in all_data:
        text = item.get("text", "")
        if text not in seen:
            seen[text] = item
        else:
            # Keep the one with later created_at (more recent correction)
            existing_time = seen[text].get("created_at", "")
            new_time = item.get("created_at", "")
            if new_time > existing_time:
                seen[text] = item
    return list(seen.values())


def register_history_page():
    @ui.page("/history")
    def history():
        page_state = {
            "current": 0,
            "per_page": 10,
            "data": [],
            "search_query": "",
            "label_filter": None,
            "date_from": None,
            "date_to": None,
        }

        def load_data():
            all_data = fetch_history(limit=10000)
            # Get only latest prediction per text (shows corrections)
            all_data = get_latest_predictions(all_data)
            filtered_data = all_data
            
            # Apply search filter
            if page_state["search_query"]:
                query = page_state["search_query"].lower()
                filtered_data = [r for r in filtered_data if query in r.get("text", "").lower()]
            
            # Apply label filter
            if page_state["label_filter"] is not None:
                filtered_data = [r for r in filtered_data if r.get("predicted_label") == page_state["label_filter"]]
            
            # Apply date filters
            if page_state["date_from"]:
                filtered_data = [r for r in filtered_data if r.get("created_at", "")[:10] >= page_state["date_from"]]
            if page_state["date_to"]:
                filtered_data = [r for r in filtered_data if r.get("created_at", "")[:10] <= page_state["date_to"]]
            
            page_state["data"] = filtered_data
            page_state["current"] = 0

        load_data()

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
                    ui.label("History Logs Dashboard").classes("text-2xl font-bold text-white")
                with ui.row().classes("gap-2 items-center"):
                    ui.link("HOME", "/").classes("nav-link text-white")
                    ui.link("CLASSIFY", "/classify").classes("nav-link text-white")
                    ui.link("BATCH", "/batch").classes("nav-link text-white")
                    ui.link("LOGS", "/history").classes("nav-link active text-white")
                    ui.link("ANALYTICS", "/analytics").classes("nav-link text-white")
                    ui.switch('', on_change=toggle_dark).props('color=white id=krx-dark-switch')

        with ui.column().classes("w-full max-w-7xl mx-auto mt-8 gap-4"):
            # Search and Filter Controls
            with ui.card().classes("card-modern w-full p-4 mb-4"):
                with ui.row().classes("gap-4 w-full flex-wrap items-end"):
                    # Search bar
                    search_input = ui.input(label="Search", placeholder="Search text...").classes("flex-1 min-w-64")
                    search_input.bind_value(page_state, "search_query")
                    
                    def on_search_change():
                        load_data()
                        table_container.refresh()
                    
                    search_input.on_value_change(on_search_change)
                    
                    # Label filter
                    label_select = ui.select(
                        {None: "All Labels", 0: "Normal", 1: "Offensive", 2: "Hate"},
                        label="Label",
                        value=None
                    ).classes("min-w-40")
                    label_select.bind_value(page_state, "label_filter")
                    
                    def on_label_change():
                        load_data()
                        table_container.refresh()
                    
                    label_select.on_value_change(on_label_change)
                    
                    # Collapsible date filters
                    with ui.expansion("📅 Date Range").classes("w-full"):
                        with ui.row().classes("gap-4 w-full"):
                            # Date from
                            with ui.column().classes("items-center"):
                                ui.label("From").classes("text-sm font-semibold")
                                date_from = ui.date()
                                date_from.bind_value(page_state, "date_from")
                                
                                def on_date_from_change():
                                    load_data()
                                    table_container.refresh()
                                
                                date_from.on_value_change(on_date_from_change)
                            
                            # Date to
                            with ui.column().classes("items-center"):
                                ui.label("To").classes("text-sm font-semibold")
                                date_to = ui.date()
                                date_to.bind_value(page_state, "date_to")
                                
                                def on_date_to_change():
                                    load_data()
                                    table_container.refresh()
                                
                                date_to.on_value_change(on_date_to_change)
                    
                    # Clear filters button
                    def clear_filters():
                        page_state["search_query"] = ""
                        page_state["label_filter"] = None
                        page_state["date_from"] = None
                        page_state["date_to"] = None
                        search_input.value = ""
                        label_select.value = None
                        date_from.value = None
                        date_to.value = None
                        load_data()
                        table_container.refresh()
                    
                    ui.button("Clear Filters", on_click=clear_filters).props("outline")
            
            # Action buttons
            with ui.row().classes("gap-2 mb-4"):
                def export_csv():
                    try:
                        EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
                        with open(EXPORT_PATH, "w", newline="", encoding="utf-8") as f:
                            writer = csv.writer(f)
                            writer.writerow(["id", "text", "lang", "predicted_label", "score", "model_name", "latency_ms", "created_at"])
                            for row in page_state["data"]:
                                writer.writerow([
                                    row.get("id", ""),
                                    row.get("text", ""),
                                    row.get("lang", ""),
                                    row.get("predicted_label", ""),
                                    row.get("score", ""),
                                    row.get("model_name", ""),
                                    row.get("latency_ms", ""),
                                    row.get("created_at", "")
                                ])
                        ui.notify(f"Exported to {EXPORT_PATH}", type="positive")
                    except Exception as ex:
                        print(f"Export error: {ex}")
                        ui.notify(f"Export failed: {ex}", type="negative")

                def delete_history():
                    def confirm_delete():
                        delete_all_predictions()
                        page_state["data"] = []
                        page_state["current"] = 0
                        table_container.refresh()
                        ui.notify("History deleted", type="positive")
                        dialog.close()

                    with ui.dialog() as dialog, ui.card().classes("card-modern p-6"):
                        ui.label("Delete all history?").classes("text-lg font-bold text-gray-800 dark:text-white")
                        ui.label("This action cannot be undone.").classes("text-sm text-gray-500 dark:text-gray-400")
                        with ui.row().classes("gap-2 mt-4"):
                            ui.button("Cancel", on_click=dialog.close).props("outline")
                            ui.button("Delete", on_click=confirm_delete).props("color=red")
                    dialog.open()

                ui.button("Export CSV", on_click=export_csv).props("color=green outline")
                ui.button("Delete History", on_click=delete_history).props("color=red outline")

            @ui.refreshable
            def table_container():
                total = len(page_state["data"])
                start = page_state["current"] * page_state["per_page"]
                end = start + page_state["per_page"]
                page_data = page_state["data"][start:end]

                if not page_data:
                    ui.label("No predictions found").classes("text-gray-500 dark:text-gray-400 text-center mt-8")
                    return

                with ui.card().classes("card-modern w-full p-6"):
                    columns = [
                        {"name": "id", "label": "ID", "field": "id", "align": "left"},
                        {"name": "text", "label": "Text", "field": "text", "align": "left"},
                        {"name": "lang", "label": "Language", "field": "lang", "align": "left"},
                        {"name": "label", "label": "Label", "field": "label", "align": "left"},
                        {"name": "model", "label": "Model", "field": "model", "align": "left"},
                        {"name": "score", "label": "Confidence", "field": "score", "align": "left"},
                        {"name": "latency", "label": "Latency", "field": "latency", "align": "left"},
                        {"name": "timestamp", "label": "Timestamp", "field": "timestamp", "align": "left"},
                    ]

                    rows = []
                    for item in page_data:
                        text_raw = item.get("text", "")
                        text_display = (text_raw[:60] + "...") if len(text_raw) > 60 else text_raw
                        label_int = item.get("predicted_label", 0)
                        label_name = LABEL_NAMES.get(label_int, "Unknown")
                        score_pct = f"{item.get('score', 0.0) * 100:.1f}%"
                        created_at = item.get("created_at", "")
                        timestamp_display = created_at if created_at else "N/A"
                        rows.append({
                            "id": item.get("id", ""),
                            "text": text_display,
                            "label": label_name,
                            "score": score_pct,
                            "model": item.get("model_name", ""),
                            "lang": item.get("lang", ""),
                            "latency": f"{item.get('latency_ms', 0)} ms",
                            "timestamp": timestamp_display,
                        })

                    ui.table(columns=columns, rows=rows).classes("w-full")

                with ui.row().classes("gap-4 mt-4 items-center justify-between w-full"):
                    def prev_page():
                        if page_state["current"] > 0:
                            page_state["current"] -= 1
                            table_container.refresh()

                    def next_page():
                        max_page = (total - 1) // page_state["per_page"]
                        if page_state["current"] < max_page:
                            page_state["current"] += 1
                            table_container.refresh()

                    ui.button("Prev", on_click=prev_page).props("outline").bind_enabled_from(page_state, "current", lambda x: x > 0)
                    ui.label(f"Page {page_state['current'] + 1} of {max((total - 1) // page_state['per_page'] + 1, 1)}").classes("text-sm text-gray-700 dark:text-gray-300")
                    ui.button("Next", on_click=next_page).props("outline").bind_enabled_from(
                        page_state, "current", lambda x: x < (total - 1) // page_state["per_page"]
                    )

            table_container()
