from nicegui import ui
from src.ui.main_page import register_home_page
from src.ui.history_page import register_history_page
from src.ui.analytics_page import register_analytics_page
from src.ui.admin_panel import register_admin_page
from src.ui.batch_page import register_batch_page
from src.ui.classify_page import *
from src.utils.db import init_db

init_db()


register_home_page()
register_history_page()
register_analytics_page()
register_admin_page()
register_batch_page()


if __name__ in {"__main__", "__mp_main__"}:
    import os
    ui.run(
        title='HateSpeechDetection',
        host=os.getenv('HOST', '0.0.0.0'),
        port=int(os.getenv('PORT', '10000')),
        reconnect_timeout=2.0,
        show=False,
        reload=False
    )
