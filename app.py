"""
LMS Student Behavior Analytics Dashboard
Main application entry point with modular page structure.
"""
import streamlit as st
import matplotlib
matplotlib.use('Agg')

# Import utilities
from utils.helpers import init_session_state
from config.styles import apply_custom_css

# Import all pages
from views import (
    home,
    upload,
    dataset_overview,
    eda,
    cluster_count,
    model_training,
    cluster_interpretation,
    time_series,
    prediction,
    dashboard,
    vle_engagement,
    about
)

# Configure page
st.set_page_config(
    page_title="LMS Student Analytics Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state and apply styling
init_session_state()
apply_custom_css()


def main():
    """Main application function."""
    st.sidebar.markdown("## Navigation")
    
    # Define pages with their corresponding modules
    pages = {
        "Home": home,
        "Dataset & Preprocessing": dataset_overview,
        "Data Upload": upload,
        "Exploratory Data Analysis": eda,
        "Cluster Count Selection": cluster_count,
        "Model Training": model_training,
        "Cluster Interpretation": cluster_interpretation,
        "Time-Series Analysis": time_series,
        "New Student Prediction": prediction,
        "Dashboard and Exports": dashboard,
        "VLE Data (Phase 2)": vle_engagement,
        "About": about
    }
    
    # Navigation
    selection = st.sidebar.radio("Go to", list(pages.keys()))
    
    st.sidebar.markdown("---")
    
    # Current status sidebar display
    st.sidebar.markdown("### Current Status")
    
    if st.session_state.df is not None:
        st.sidebar.success("Dataset loaded")
    else:
        st.sidebar.warning("No dataset")
    
    if st.session_state.elbow_done:
        st.sidebar.success(f"K = {st.session_state.selected_k}")
    else:
        st.sidebar.warning("Cluster count not set")
    
    if st.session_state.models:
        st.sidebar.success(f"Models: {len(st.session_state.models)}")
        if st.session_state.active_model:
            st.sidebar.info(f"Active: {st.session_state.active_model}")
    else:
        st.sidebar.warning("No models trained")
    
    # Render selected page
    pages[selection].render()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        st.error("An error occurred during application execution:")
        st.code(traceback.format_exc())
