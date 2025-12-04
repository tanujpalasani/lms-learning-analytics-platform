"""
Custom CSS styling for the LMS Analytics Dashboard.
"""
import streamlit as st


def apply_custom_css():
    """Apply custom CSS styling to the Streamlit app."""
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            color: #008080;
            margin-bottom: 1rem;
            text-align: center;
        }
        .sub-header {
            font-size: 1.5rem;
            font-weight: 600;
            color: #2F4F4F;
            margin-bottom: 0.5rem;
        }
        .metric-card {
            background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 4px solid #008080;
            margin-bottom: 1rem;
        }
        .cluster-card {
            background: #ffffff;
            padding: 1.5rem;
            border-radius: 10px;
            border: 1px solid #e0e0e0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }
        .stButton>button {
            background-color: #008080;
            color: white;
            border-radius: 5px;
            border: none;
            padding: 0.5rem 1rem;
        }
        .stButton>button:hover {
            background-color: #006666;
        }
        div[data-testid="stMetric"] {
            background-color: #f0f8f8;
            padding: 1rem;
            border-radius: 8px;
            border-left: 3px solid #008080;
        }
        .info-box {
            background-color: #e6f3f3;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #008080;
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)
