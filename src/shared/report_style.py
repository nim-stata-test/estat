"""
Shared CSS styles for HTML reports.

Based on the design system from statistik.bs.ch (Kanton Basel-Stadt)
"""

# Color palette
COLORS = {
    'primary_green': '#2a9749',
    'primary_dark_blue': '#1e4557',
    'accent_blue': '#079bca',
    'purple': '#9156b4',
    'light_green': '#ddecde',
    'dark_teal': '#146c8b',
    'gray_light': '#f8f8f8',
    'gray_border': '#e3e3e3',
    'gray_medium': '#bababa',
    'gray_dark': '#949494',
    'text': '#333',
    'red_alert': '#ff3a1f',
    'white': '#ffffff',
}

# CSS stylesheet matching statistik.bs.ch design
CSS = f"""
:root {{
    --primary-green: {COLORS['primary_green']};
    --primary-dark-blue: {COLORS['primary_dark_blue']};
    --accent-blue: {COLORS['accent_blue']};
    --purple: {COLORS['purple']};
    --light-green: {COLORS['light_green']};
    --dark-teal: {COLORS['dark_teal']};
    --gray-light: {COLORS['gray_light']};
    --gray-border: {COLORS['gray_border']};
    --gray-medium: {COLORS['gray_medium']};
    --gray-dark: {COLORS['gray_dark']};
    --text: {COLORS['text']};
    --red-alert: {COLORS['red_alert']};
}}

* {{
    box-sizing: border-box;
}}

body {{
    font-family: Inter, "Helvetica Neue", Helvetica, Arial, sans-serif;
    font-size: 16px;
    line-height: 1.6;
    color: var(--text);
    max-width: 1200px;
    margin: 0 auto;
    padding: 40px 20px;
    background-color: #fafafa;
}}

h1 {{
    font-size: 2.5rem;
    font-weight: 700;
    color: var(--primary-dark-blue);
    margin-top: 0;
    margin-bottom: 1rem;
    border-bottom: 3px solid var(--primary-green);
    padding-bottom: 0.5rem;
}}

h2 {{
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--primary-dark-blue);
    margin-top: 2.5rem;
    margin-bottom: 1rem;
    border-bottom: 2px solid var(--gray-border);
    padding-bottom: 0.25rem;
}}

h3 {{
    font-size: 1.25rem;
    font-weight: 600;
    color: var(--dark-teal);
    margin-top: 1.5rem;
    margin-bottom: 0.75rem;
}}

h4 {{
    font-size: 1rem;
    font-weight: 600;
    color: var(--primary-dark-blue);
    margin-top: 1.25rem;
    margin-bottom: 0.5rem;
}}

p {{
    margin-bottom: 1rem;
}}

a {{
    color: var(--primary-dark-blue);
    text-decoration: none;
    transition: color 0.15s ease;
}}

a:hover {{
    color: var(--accent-blue);
    text-decoration: underline;
}}

a:visited {{
    color: var(--purple);
}}

/* Tables */
table {{
    border-collapse: collapse;
    width: 100%;
    margin: 1.5rem 0;
    font-size: 0.9rem;
    background-color: white;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    border-radius: 4px;
    overflow: hidden;
}}

th {{
    background-color: var(--gray-light);
    color: var(--primary-green);
    font-weight: 600;
    text-align: left;
    padding: 12px 16px;
    border-bottom: 2px solid var(--text);
}}

td {{
    padding: 10px 16px;
    border-bottom: 1px solid var(--gray-border);
}}

tr:last-child td {{
    border-bottom: none;
}}

tr:hover {{
    background-color: var(--light-green);
}}

/* Code blocks */
pre, code {{
    font-family: "SF Mono", Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    font-size: 0.875rem;
}}

code {{
    background-color: var(--gray-light);
    padding: 2px 6px;
    border-radius: 4px;
    color: var(--dark-teal);
}}

pre {{
    background-color: var(--primary-dark-blue);
    color: #f0f0f0;
    padding: 16px 20px;
    border-radius: 4px;
    overflow-x: auto;
    margin: 1rem 0;
}}

pre code {{
    background: none;
    padding: 0;
    color: inherit;
}}

/* Alert boxes */
.info {{
    background-color: var(--light-green);
    border-left: 4px solid var(--primary-green);
    padding: 12px 16px;
    margin: 1rem 0;
    border-radius: 0 4px 4px 0;
}}

.warning {{
    background-color: #fff8e6;
    border-left: 4px solid #f0ad4e;
    padding: 12px 16px;
    margin: 1rem 0;
    border-radius: 0 4px 4px 0;
}}

.error {{
    background-color: #fdeaea;
    border-left: 4px solid var(--red-alert);
    padding: 12px 16px;
    margin: 1rem 0;
    border-radius: 0 4px 4px 0;
}}

.recommendation {{
    background-color: var(--light-green);
    border-left: 4px solid var(--primary-green);
    padding: 12px 16px;
    margin: 1rem 0;
    border-radius: 0 4px 4px 0;
}}

/* Summary boxes */
.summary {{
    background-color: white;
    border: 1px solid var(--gray-border);
    border-radius: 4px;
    padding: 20px;
    margin: 1.5rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}}

.summary h3 {{
    margin-top: 0;
    color: var(--primary-green);
}}

/* Metric cards */
.metrics {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin: 1.5rem 0;
}}

.metric {{
    background-color: white;
    border: 1px solid var(--gray-border);
    border-radius: 4px;
    padding: 16px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}}

.metric-value {{
    font-size: 2rem;
    font-weight: 700;
    color: var(--primary-green);
    line-height: 1.2;
}}

.metric-label {{
    font-size: 0.875rem;
    color: var(--gray-dark);
    margin-top: 4px;
}}

/* Figures */
figure {{
    margin: 1.5rem 0;
    text-align: center;
}}

figure img {{
    max-width: 100%;
    height: auto;
    border-radius: 4px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}}

figcaption {{
    font-size: 0.875rem;
    color: var(--gray-dark);
    margin-top: 8px;
    font-style: italic;
}}

/* Lists */
ul, ol {{
    margin-bottom: 1rem;
    padding-left: 1.5rem;
}}

li {{
    margin-bottom: 0.5rem;
}}

/* Footer */
.footer {{
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--gray-border);
    font-size: 0.875rem;
    color: var(--gray-dark);
    text-align: center;
}}

/* Print styles */
@media print {{
    body {{
        background-color: white;
        max-width: none;
        padding: 0;
    }}

    .no-print {{
        display: none;
    }}

    table {{
        box-shadow: none;
    }}

    pre {{
        white-space: pre-wrap;
    }}
}}

/* Responsive adjustments */
@media (max-width: 768px) {{
    body {{
        padding: 20px 16px;
    }}

    h1 {{
        font-size: 1.75rem;
    }}

    h2 {{
        font-size: 1.5rem;
    }}

    table {{
        font-size: 0.8rem;
    }}

    th, td {{
        padding: 8px 10px;
    }}
}}
"""


def get_html_head(title: str, extra_css: str = "") -> str:
    """Generate HTML head section with standard styling.

    Args:
        title: Page title
        extra_css: Additional CSS to include

    Returns:
        HTML string for the head section
    """
    return f"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
{CSS}
{extra_css}
</style>
</head>
"""


def get_html_footer(generated_by: str = "ESTAT Analysis Pipeline") -> str:
    """Generate HTML footer section.

    Args:
        generated_by: Attribution text

    Returns:
        HTML string for the footer
    """
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""
<div class="footer">
    <p>Generated by {generated_by} | {timestamp}</p>
    <p>Design based on <a href="https://statistik.bs.ch">Statistik Basel-Stadt</a></p>
</div>
</body>
</html>
"""


def wrap_html(title: str, body: str, extra_css: str = "") -> str:
    """Wrap body content in full HTML document with styling.

    Args:
        title: Page title
        body: HTML body content
        extra_css: Additional CSS to include

    Returns:
        Complete HTML document string
    """
    return f"""{get_html_head(title, extra_css)}
<body>
{body}
{get_html_footer()}
"""


def create_metric_card(value: str, label: str) -> str:
    """Create a metric card HTML element.

    Args:
        value: The metric value to display
        label: The label below the value

    Returns:
        HTML string for the metric card
    """
    return f"""<div class="metric">
    <div class="metric-value">{value}</div>
    <div class="metric-label">{label}</div>
</div>"""


def create_metrics_grid(metrics: list) -> str:
    """Create a grid of metric cards.

    Args:
        metrics: List of (value, label) tuples

    Returns:
        HTML string for the metrics grid
    """
    cards = "\n".join(create_metric_card(v, l) for v, l in metrics)
    return f'<div class="metrics">\n{cards}\n</div>'


def create_info_box(content: str, box_type: str = "info") -> str:
    """Create an info/warning/error box.

    Args:
        content: Box content
        box_type: One of 'info', 'warning', 'error', 'recommendation'

    Returns:
        HTML string for the box
    """
    return f'<div class="{box_type}">\n{content}\n</div>'


def create_figure(src: str, caption: str = "", alt: str = "") -> str:
    """Create a figure with caption.

    Args:
        src: Image source path
        caption: Figure caption
        alt: Alt text for accessibility

    Returns:
        HTML string for the figure
    """
    alt_text = alt or caption
    caption_html = f"<figcaption>{caption}</figcaption>" if caption else ""
    return f"""<figure>
    <img src="{src}" alt="{alt_text}">
    {caption_html}
</figure>"""
