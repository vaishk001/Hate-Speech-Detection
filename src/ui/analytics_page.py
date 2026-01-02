from nicegui import ui
import plotly.graph_objects as go
import json
from pathlib import Path
from collections import Counter
from datetime import datetime
from src.utils.db import fetch_history
from src.ui.icons import get_icon

LABEL_NAMES = {0: "Normal", 1: "Offensive", 2: "Hate"}
LABEL_COLORS = {0: "#22C55E", 1: "#f59e0b", 2: "#EF4444"}

PRIMARY_BLUE = "#1E90FF"
WHITE = "#FFFFFF"
LIGHT_GRAY = "#F5F6F7"
ACCENT_GREEN = "#22C55E"
ACCENT_RED = "#EF4444"

REPORTS_DIR = Path("reports")


def _create_bar_chart(label_counts: dict) -> go.Figure:
    labels = [LABEL_NAMES.get(k, f"Label {k}") for k in sorted(label_counts.keys())]
    values = [label_counts[k] for k in sorted(label_counts.keys())]
    colors = [LABEL_COLORS.get(k, PRIMARY_BLUE) for k in sorted(label_counts.keys())]
    
    fig = go.Figure(data=[
        go.Bar(x=labels, y=values, marker_color=colors, text=values, textposition='auto')
    ])
    fig.update_layout(
        title="Prediction Distribution",
        xaxis_title="Category",
        yaxis_title="Count",
        template="plotly_white",
        height=400,
        margin=dict(l=50, r=50, t=60, b=80),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#1f2937")
    )
    return fig


def _create_timeline_chart(records: list) -> go.Figure:
    dates = []
    labels = []
    for r in records:
        created = r.get("created_at", "")
        try:
            dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
            dates.append(dt)
            labels.append(r.get("predicted_label", 0))
        except:
            pass
    
    if not dates:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template="plotly_white", height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        return fig
    
    sorted_data = sorted(zip(dates, labels))
    dates_sorted = [d[0] for d in sorted_data]
    labels_sorted = [d[1] for d in sorted_data]
    counts = list(range(1, len(dates_sorted) + 1))
    colors = [LABEL_COLORS.get(lbl, PRIMARY_BLUE) for lbl in labels_sorted]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates_sorted, y=counts, mode='lines+markers',
        line=dict(color=PRIMARY_BLUE, width=3),
        marker=dict(size=8, color=colors, line=dict(width=2, color='white')),
        hovertemplate='<b>Time:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>'
    ))
    fig.update_layout(
        title="Predictions Timeline",
        xaxis_title="Time",
        yaxis_title="Cumulative Count",
        template="plotly_white",
        height=400,
        margin=dict(l=40, r=40, t=60, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#1f2937")
    )
    return fig

def _create_pie_chart(label_counts: dict) -> go.Figure:
    labels = [LABEL_NAMES.get(k, f"Label {k}") for k in sorted(label_counts.keys())]
    values = [label_counts[k] for k in sorted(label_counts.keys())]
    colors = [LABEL_COLORS.get(k, PRIMARY_BLUE) for k in sorted(label_counts.keys())]
    
    fig = go.Figure(data=[
        go.Pie(labels=labels, values=values, marker_colors=colors, hole=0.4, 
               textinfo='label+percent', hoverinfo='label+value+percent')
    ])
    fig.update_layout(
        title="Label Distribution",
        template="plotly_white",
        height=400,
        margin=dict(l=50, r=50, t=60, b=80),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#1f2937")
    )
    return fig


def register_analytics_page():
    @ui.page("/analytics")
    def analytics():
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
            
            .kpi-card {{
                background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
                border-radius: 1rem;
                border-left: 4px solid;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
                transition: all 0.3s ease;
                padding: 1.5rem;
            }}
            
            .kpi-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
            }}
            
            .chart-card {{
                background: white;
                border-radius: 1rem;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
                transition: all 0.3s ease;
                padding: 1.5rem;
                border: 1px solid #e5e7eb;
            }}
            
            .chart-card:hover {{
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
                transform: translateY(-2px);
            }}
            
            .section-title {{
                font-size: 1.5rem;
                font-weight: 700;
                color: #1f2937;
                margin-bottom: 1.5rem;
                border-left: 4px solid {PRIMARY_BLUE};
                padding-left: 1rem;
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
                    radial-gradient(circle at 40% 20%, rgba(239, 68, 68, 0.05) 0%, transparent 50%);
                pointer-events: none;
                z-index: -1;
            }}
            
            body.body--dark {{
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%) !important;
                background-attachment: fixed;
                color: #e2e8f0;
            }}
            
            body.body--dark .card-modern,
            body.body--dark .bg-white {{
                background: rgba(30, 41, 59, 0.95) !important;
                border-color: rgba(30, 144, 255, 0.2);
                color: #e2e8f0;
                backdrop-filter: blur(10px);
            }}
            
            body.body--dark .kpi-card,
            body.body--dark .chart-card {{
                background: rgba(30, 41, 59, 0.95) !important;
                border-color: rgba(30, 144, 255, 0.2);
                color: #e2e8f0;
            }}
            
            body.body--dark .section-title {{
                color: #f8fafc !important;
            }}
            
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
                    ui.html(get_icon("analytics", "28"), sanitize=False).classes("text-white")
                    ui.label("Analytics Dashboard").classes("text-2xl font-bold text-white")
                with ui.row().classes("gap-2 items-center"):
                    ui.link("HOME", "/").classes("nav-link text-white")
                    ui.link("CLASSIFY", "/classify").classes("nav-link text-white")
                    ui.link("BATCH", "/batch").classes("nav-link text-white")
                    ui.link("LOGS", "/history").classes("nav-link text-white")
                    ui.link("ANALYTICS", "/analytics").classes("nav-link active text-white")
                    ui.switch('', on_change=toggle_dark).props('color=white id=krx-dark-switch')

        with ui.column().classes("w-full max-w-7xl mx-auto mt-8 gap-8 px-6"):
            # KPI Cards
            with ui.row().classes('w-full gap-4'):
                with ui.card().classes('flex-1 kpi-card gradient-card-blue'):
                    with ui.column().classes('items-center text-center'):
                        ui.label('Total Predictions').classes('text-sm text-gray-500 uppercase tracking-wider font-semibold')
                        total_label = ui.label().classes('stat-number mt-2')
                
                with ui.card().classes('flex-1 kpi-card gradient-card-red'):
                    with ui.column().classes('items-center text-center'):
                        ui.label('Hate Speech').classes('text-sm text-gray-500 uppercase tracking-wider font-semibold')
                        hate_label = ui.label().classes('stat-number mt-2')
                
                with ui.card().classes('flex-1 kpi-card').style(f'border-left-color: #f59e0b'):
                    with ui.column().classes('items-center text-center'):
                        ui.label('Offensive Content').classes('text-sm text-gray-500 uppercase tracking-wider font-semibold')
                        offensive_label = ui.label().classes('stat-number mt-2')
                
                with ui.card().classes('flex-1 kpi-card gradient-card-green'):
                    with ui.column().classes('items-center text-center'):
                        ui.label('Avg Latency').classes('text-sm text-gray-500 uppercase tracking-wider font-semibold')
                        latency_label = ui.label().classes('stat-number mt-2')

            # Charts
            with ui.row().classes('w-full gap-6'):
                with ui.card().classes('flex-1 chart-card'):
                    ui.label('Prediction Distribution').classes('section-title')
                    bar_plot = ui.plotly(_create_bar_chart({})).classes('w-full h-80')
                
                with ui.card().classes('flex-1 chart-card'):
                    ui.label('Content Distribution').classes('section-title')
                    pie_plot = ui.plotly(_create_pie_chart({})).classes('w-full h-80')

            with ui.card().classes('chart-card w-full'):
                ui.label('Predictions Timeline').classes('section-title')
                timeline_plot = ui.plotly(_create_timeline_chart([])).classes('w-full h-96')

            # Model Performance
            with ui.card().classes('chart-card w-full'):
                ui.label('📊 Baseline Model Performance').classes('section-title')
                baseline_accuracy_label = ui.label().classes('text-xl font-bold text-green-600 text-center mb-4')
                baseline_metrics_container = ui.column().classes('gap-3 w-full')

            def update_data():
                data = fetch_history(limit=1000)
                total = len(data)
                label_counter = Counter(r.get("predicted_label", 0) for r in data)
                hate_count = label_counter.get(2, 0)
                offensive_count = label_counter.get(1, 0)
                avg_latency = sum(r.get("latency_ms", 0) for r in data) / total if total > 0 else 0
                
                total_label.text = str(total)
                hate_label.text = str(hate_count)
                offensive_label.text = str(offensive_count)
                latency_label.text = f"{avg_latency:.0f} ms"
                
                bar_plot.update_figure(_create_bar_chart(label_counter))
                timeline_plot.update_figure(_create_timeline_chart(data[:100]))
                pie_plot.update_figure(_create_pie_chart(label_counter))
                
                # Load baseline report
                baseline_report_path = REPORTS_DIR / 'classification_report_baseline.json'
                try:
                    if baseline_report_path.exists():
                        with open(baseline_report_path, 'r') as f:
                            report_data = json.load(f)
                        logreg_report = report_data.get('logreg', {}).get('report', {})
                        accuracy = logreg_report.get('accuracy', 0)
                        baseline_accuracy_label.text = f'Overall Accuracy: {accuracy:.2%}'
                        baseline_metrics_container.clear()
                        for key, metrics in logreg_report.items():
                            if isinstance(metrics, dict) and 'precision' in metrics:
                                with baseline_metrics_container:
                                    with ui.card().classes('p-4 gradient-card-blue rounded-lg'):
                                        with ui.row().classes('items-center justify-between'):
                                            ui.label(key).classes('font-semibold text-gray-800 dark:text-white')
                                            with ui.row().classes('gap-4'):
                                                ui.label(f'P: {metrics.get("precision", 0):.3f}').classes('text-sm bg-blue-100 dark:bg-blue-900/50 px-2 py-1 rounded')
                                                ui.label(f'R: {metrics.get("recall", 0):.3f}').classes('text-sm bg-green-100 dark:bg-green-900/50 px-2 py-1 rounded')
                                                ui.label(f'F1: {metrics.get("f1-score", 0):.3f}').classes('text-sm bg-purple-100 dark:bg-purple-900/50 px-2 py-1 rounded')
                    else:
                        baseline_accuracy_label.text = 'Overall Accuracy: No trained model'
                        baseline_metrics_container.clear()
                        with baseline_metrics_container:
                            ui.label('Train a model in Batch page to see metrics').classes('text-sm text-gray-500')
                except Exception as e:
                    baseline_accuracy_label.text = f'Error loading report'
                    baseline_metrics_container.clear()

            update_data()
            ui.timer(10.0, update_data)
