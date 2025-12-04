"""
Data upload page for CSV file upload and validation.
"""
import streamlit as st
import pandas as pd
import os
from utils.helpers import validate_dataset
from utils.constants import REQUIRED_COLUMNS
from utils.dataset_utils import PROCESSED_DATA_DIR


def load_and_validate_df(df, source_name="Uploaded File"):
    """
    Validate and load the dataframe into session state.
    """
    is_valid, message = validate_dataset(df)
    
    if is_valid:
        st.success(f"Successfully loaded data from: {source_name}")
        st.session_state.df = df
        
        # Reset state on new upload
        st.session_state.models = {}
        st.session_state.metrics = {}
        st.session_state.cluster_labels = {}
        st.session_state.cluster_centroids = {}
        st.session_state.active_model = None
        st.session_state.clustered_df = None
        st.session_state.elbow_done = False
        
        st.markdown("### Dataset Preview")
        st.dataframe(df.head(10))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Total Features", f"{len(df.columns)}")
        with col3:
            st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
        
        st.markdown("### Column Information")
        col_info = pd.DataFrame({
            'Column': df.columns.tolist(),
            'Data Type': [str(dt) for dt in df.dtypes.values],
            'Non-Null Count': df.count().values.tolist(),
            'Null Count': df.isnull().sum().values.tolist()
        })
        st.dataframe(col_info)
    else:
        st.error(message)


def render():
    """Render the data upload page."""
    st.markdown('<h1 class="main-header">Data Upload</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    Upload your cleaned LMS dataset in CSV format, or use the default dataset provided with the project.
    </div>
    """, unsafe_allow_html=True)
    
    # Create two tabs for the options
    tab1, tab2 = st.tabs(["📂 Upload Dataset", "💾 Use Default Dataset"])
    
    with tab1:
        st.markdown("### Upload your own CSV")
        st.markdown("""
        The dataset should contain columns like:
        - `id_student`, `final_result`
        - `total_clicks`, `active_days`, `avg_daily_clicks`
        - `quizzes_attempted`, `avg_quiz_score`
        """)
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                load_and_validate_df(df, source_name=uploaded_file.name)
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    with tab2:
        st.markdown("### Load Default Dataset")
        st.markdown("Use the pre-cleaned `lms_clean.csv` included in the project root.")
        
        if st.button("Load Default Dataset (lms_clean.csv)", type="primary"):
            default_path = os.path.join(PROCESSED_DATA_DIR, "lms_clean.csv")
            if os.path.exists(default_path):
                try:
                    df = pd.read_csv(default_path)
                    load_and_validate_df(df, source_name="Default Dataset (lms_clean.csv)")
                except Exception as e:
                    st.error(f"Error reading default file: {str(e)}")
            else:
                st.error(f"Default file not found at: {default_path}")
    
    # Show current status if data is loaded but we are not currently processing an action
    if st.session_state.df is not None and uploaded_file is None:
        st.markdown("---")
        st.info("✅ Dataset is currently loaded in memory. You can proceed to **EDA** or **Cluster Count**.")
