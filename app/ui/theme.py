"""
ui/theme.py
Helper that applies the dark “glass-morphism” look everywhere.
"""

from pathlib import Path
import streamlit as st

# expose colours so the main app can `from ui.theme import PRIMARY, SUCCESS`
PRIMARY = "#ff595e"   # critical / primary
SUCCESS = "#8ac926"   # safe

def apply_dark_glass() -> None:
    """Set page-config and inject the CSS defined in *ui/theme.css*."""
    st.set_page_config(
        page_title="SOTERIA – Crash Detector",
        page_icon="🚨",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    css = Path(__file__).with_suffix(".css").read_text(encoding="utf-8")
    st.markdown(css, unsafe_allow_html=True)
