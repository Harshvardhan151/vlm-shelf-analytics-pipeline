"""Reusable HTML/Streamlit presentation helpers. No analytics here."""
from html import escape

import streamlit as st

WORKFLOW_STEPS = [
    "Select the aisle you are auditing",
    "Upload a clear photo of the shelf",
    "Run the shelf analysis",
    "Review compliance, trends and forecast",
]


def _html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def page_header() -> None:
    _html(
        """
        <div class="siq-header">
          <div class="siq-eyebrow">Retail Vision-to-Sales Analytics</div>
          <div class="siq-title">ShelfIQ</div>
          <div class="siq-subtitle">Retail Vision-to-Sales Intelligence</div>
          <div class="siq-desc">
            Audit shelf photos with a vision language model, compare them to your planogram,
            quantify revenue at risk, and forecast demand to plan replenishment.
          </div>
        </div>
        """
    )


def sidebar_brand() -> None:
    _html(
        """
        <div class="siq-brand">
          <div class="siq-logo">S</div>
          <div>
            <div class="siq-brand-name">ShelfIQ</div>
            <div class="siq-brand-sub">Retail Vision-to-Sales Analytics</div>
          </div>
        </div>
        """
    )


def section_label(text: str) -> None:
    _html(f'<div class="siq-section-label">{escape(text)}</div>')


def sidebar_divider() -> None:
    _html('<hr class="siq-sidebar-divider">')


def workflow_guide() -> None:
    items = "".join(
        f'<li><span class="siq-step-num">{i}</span><span>{escape(s)}</span></li>'
        for i, s in enumerate(WORKFLOW_STEPS, 1)
    )
    _html(f'<ul class="siq-steps">{items}</ul>')


def aisle_summary(n_skus: int, items: list) -> None:
    chips = "".join(f'<span class="siq-chip">{escape(str(i))}</span>' for i in items)
    _html(
        f"""
        <div class="siq-stat-row">
          <div class="siq-stat"><div class="siq-stat-value">{n_skus}</div>
            <div class="siq-stat-label">Expected SKUs</div></div>
        </div>
        <div class="siq-chip-row">{chips}</div>
        """
    )


def heading(title: str, subtitle: str = "") -> None:
    sub = f'<div class="siq-h2-sub">{escape(subtitle)}</div>' if subtitle else ""
    _html(f'<div class="siq-h2">{escape(title)}</div>{sub}')


def spacer() -> None:
    _html('<div class="siq-spacer"></div>')


def kpi_card(title: str, value: str, delta: str = "", tone: str = "neutral") -> None:
    """tone: success | warning | critical | neutral"""
    delta_html = f'<div class="siq-kpi-delta">{escape(delta)}</div>' if delta else ""
    _html(
        f"""
        <div class="siq-kpi siq-kpi--{tone}">
          <div class="siq-kpi-title">{escape(title)}</div>
          <div class="siq-kpi-value">{escape(value)}</div>
          {delta_html}
        </div>
        """
    )


def alert(severity: str, title: str, meta: str, impact: str = "", impact_label: str = "") -> None:
    """severity: critical | warning | success"""
    badge = {"critical": "Critical", "warning": "Warning", "success": "Clear"}[severity]
    impact_html = (
        f'<div class="siq-alert-impact"><div class="siq-alert-impact-value">{escape(impact)}</div>'
        f'<div class="siq-alert-impact-label">{escape(impact_label)}</div></div>'
        if impact else ""
    )
    _html(
        f"""
        <div class="siq-alert siq-alert--{severity}">
          <div>
            <div class="siq-alert-head">
              <span class="siq-badge siq-badge--{severity}">{badge}</span>
              <span class="siq-alert-title">{escape(title)}</span>
            </div>
            <div class="siq-alert-meta">{meta}</div>
          </div>
          {impact_html}
        </div>
        """
    )


def callout(html_text: str) -> None:
    """html_text may contain <b>; escape dynamic values before passing."""
    _html(f'<div class="siq-callout">{html_text}</div>')


def empty_state(title: str, text: str) -> None:
    _html(
        f'<div class="siq-empty"><div class="siq-empty-title">{escape(title)}</div>'
        f'<div class="siq-empty-text">{escape(text)}</div></div>'
    )
