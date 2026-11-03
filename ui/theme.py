"""Design tokens, CSS loader and the shared Plotly layout."""
from pathlib import Path

import streamlit as st

PRIMARY = "#4F46E5"
TEXT = "#0F172A"
MUTED = "#64748B"
BORDER = "#E2E8F0"
SUCCESS = "#16A34A"
WARNING = "#D97706"
CRITICAL = "#DC2626"
NEUTRAL_SERIES = "#94A3B8"

FONT_FAMILY = "Inter, -apple-system, 'Segoe UI', sans-serif"
_CSS_PATH = Path(__file__).parent / "styles.css"


def inject_css() -> None:
    st.markdown(f"<style>{_CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def style_figure(fig, *, title=None, x_title=None, y_title=None, height=380, legend=True):
    """Apply the shared chart look. Never touches the figure's data."""
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=15, color=TEXT)) if title else None,
        height=height,
        margin=dict(l=8, r=8, t=56 if title else 24, b=8),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family=FONT_FAMILY, size=12, color=MUTED),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=BORDER, font=dict(family=FONT_FAMILY, color=TEXT)),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=None),
    )
    fig.update_xaxes(title=x_title, showgrid=False, linecolor=BORDER, tickfont=dict(color=MUTED), automargin=True)
    fig.update_yaxes(title=y_title, gridcolor="#F1F5F9", zeroline=False, linecolor=BORDER,
                     tickfont=dict(color=MUTED), automargin=True)
    return fig
