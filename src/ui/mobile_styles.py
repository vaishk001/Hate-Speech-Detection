# Mobile responsive CSS that can be imported by all pages
MOBILE_CSS = """
/* Mobile Responsive Styles */
* {
    word-wrap: break-word;
    overflow-wrap: break-word;
}

@media (max-width: 768px) {
    .gradient-header {
        padding: 0.75rem 1rem !important;
    }
    
    .nav-link {
        padding: 0.4rem 0.6rem !important;
        font-size: 0.75rem !important;
    }
    
    .card-modern {
        padding: 1rem !important;
    }
    
    .card-modern:hover {
        transform: none !important;
    }
    
    body {
        font-size: 14px !important;
    }
    
    .text-2xl {
        font-size: 1.25rem !important;
    }
    
    .text-xl {
        font-size: 1.125rem !important;
    }
    
    .text-lg {
        font-size: 1rem !important;
    }
    
    .hero-section {
        padding: 2rem 1rem !important;
        min-height: auto !important;
        height: auto !important;
    }
    
    .stat-number {
        font-size: 1.75rem !important;
    }
    
    .typing-animation {
        font-size: 1.5rem !important;
    }
}

@media (max-width: 640px) {
    .gradient-header {
        padding: 0.5rem !important;
    }
    
    .nav-link {
        padding: 0.3rem 0.5rem !important;
        font-size: 0.7rem !important;
    }
    
    .card-modern {
        padding: 0.75rem !important;
        border-radius: 0.75rem !important;
    }
    
    .text-2xl {
        font-size: 1.125rem !important;
    }
    
    .text-xl {
        font-size: 1rem !important;
    }
    
    .stat-number {
        font-size: 1.5rem !important;
    }
    
    .hero-section {
        padding: 1.5rem 0.75rem !important;
    }
    
    /* Prevent text overflow */
    .q-table td, .q-table th {
        max-width: 150px !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        white-space: nowrap !important;
    }
    
    /* Make tables scrollable */
    .q-table {
        overflow-x: auto !important;
        display: block !important;
    }
}

/* Prevent horizontal scroll */
body, html {
    overflow-x: hidden !important;
    max-width: 100vw !important;
}

/* Ensure all containers respect viewport */
.w-full, .max-w-6xl, .max-w-7xl, .max-w-4xl {
    max-width: 100% !important;
    padding-left: 0.5rem !important;
    padding-right: 0.5rem !important;
}

/* Text truncation for long content */
.truncate-mobile {
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    white-space: nowrap !important;
}

@media (max-width: 640px) {
    .truncate-mobile {
        max-width: 200px !important;
    }
}
"""
