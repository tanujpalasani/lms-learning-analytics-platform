"""
Dataset & Preprocessing documentation page.
"""
import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go
from utils.dataset_utils import load_clean_dataset, get_raw_table_info, generate_preprocessing_report_pdf
from utils.constants import BACKGROUND_COLOR
from config.styles import apply_custom_css

def render():
    """Render the dataset overview page."""
    # Ensure CSS is applied
    apply_custom_css()
    
    st.markdown('<h1 class="main-header">Dataset & Preprocessing Documentation</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">From raw OULAD tables to ML-ready LMS behavior features</p>', unsafe_allow_html=True)
    
    # Load data
    clean_df = load_clean_dataset()
    raw_info = get_raw_table_info()
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview", 
        "Original Tables", 
        "Preprocessing Pipeline", 
        "Final Dataset", 
        "Downloads"
    ])
    
    # --- Tab 1: Overview ---
    with tab1:
        st.markdown("### 📚 The Open University Learning Analytics Dataset (OULAD)")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.info("""
            **Source:** The Open University  
            **Link:** [analyse.kmi.open.ac.uk/open-dataset](https://analyse.kmi.open.ac.uk/open-dataset)  
            **Citation:** Kuzilek J., Hlosta M., Zdrahal Z. Open University Learning Analytics dataset. Scientific Data 4, 170171 (2017).
            """)
            st.markdown("""
            The dataset contains data about courses, students, and their interactions with the Virtual Learning Environment (VLE) 
            for seven selected courses (modules). It is one of the most comprehensive public datasets for learning analytics.
            """)
            
        with col2:
            st.markdown("""
            <div style="background-color: {BACKGROUND_COLOR}; padding: 15px; border-radius: 5px;">
            <h4>Key Stats (Raw)</h4>
            <ul>
                <li><b>32,593</b> Students</li>
                <li><b>22</b> Course Presentations</li>
                <li><b>10.6M</b> VLE Interactions</li>
                <li><b>173k</b> Assessment Scores</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 🎯 Design Decisions & Feature Justification")
        st.markdown("""
        **Why were demographic features excluded?**
        
        The original dataset includes demographic attributes such as `gender`, `region`, `highest_education`, `imd_band` (deprivation index), `disability`, and `age_band`.
        
        We explicitly **excluded** these from the clustering model for the following reasons:
        1.  **Behavioral Focus**: The goal of this project is to identify *learner personas* based on **engagement and performance behaviors** (what they do), not who they are.
        2.  **Bias Mitigation**: Including demographics in unsupervised clustering can lead to clusters that simply proxy for socioeconomic status or other protected attributes, rather than revealing learning patterns.
        3.  **Actionability**: Interventions based on behavior (e.g., "increase login frequency") are more actionable for instructors than those based on demographics.
        
        *Note: These fields can be reintroduced in Phase 2 for fairness analysis to ensure the behavioral clusters do not disproportionately impact specific groups.*
        """)

    # --- Tab 2: Original Tables ---
    with tab2:
        st.markdown("### 🗃️ Raw Data Schema")
        st.markdown("The OULAD consists of 7 relational CSV tables connected by `id_student`, `code_module`, and `code_presentation`.")
        
        for name, info in raw_info.items():
            with st.expander(f"📄 {name} ({info['rows']} rows)"):
                st.markdown(f"**{info['description']}**")
                st.markdown("**Columns:**")
                st.markdown(", ".join([f"`{c}`" for c in info['columns']]))

    # --- Tab 3: Preprocessing Pipeline ---
    with tab3:
        st.markdown("### 🔄 Transformation Pipeline")
        st.markdown("How we transformed raw relational logs into a single student-level feature matrix.")
        
        st.markdown("""
        1.  **Load Raw Tables**: Import `studentVle`, `studentAssessment`, `studentRegistration`, and `studentInfo`.
        2.  **Engagement Engineering** (from `studentVle.csv`):
            *   `total_clicks`: Sum of `sum_click`.
            *   `active_days`: Count of unique dates with activity.
            *   `avg_daily_clicks`: `total_clicks` / `active_days`.
        3.  **Performance Engineering** (from `studentAssessment.csv`):
            *   Filtered for TMA (Tutor Marked Assessment) and CMA (Computer Marked Assessment).
            *   `quizzes_attempted`: Count of submissions.
            *   `avg_quiz_score`: Mean of `score`.
        4.  **Registration Engineering** (from `studentRegistration.csv`):
            *   `date_registration`: Days relative to course start.
            *   `unregistered_flag`: 1 if withdrawn, 0 otherwise.
        5.  **Merging**: Left-join all features onto the base `studentInfo` table using `id_student`.
        6.  **Cleaning**: Filled missing numeric values (clicks, scores) with `0` (implying no activity).
        7.  **Export**: Saved as `lms_clean.csv`.
        """)
        
        st.markdown("### 📉 Row/Column Changes Summary")
        
        # Static comparison table since we don't have raw files to count dynamically
        comp_data = {
            "Stage": [
                "Raw studentInfo.csv", 
                "After merging Engagement", 
                "After merging Assessments", 
                "Final Cleaned Dataset"
            ],
            "Rows": ["32,593", "32,593", "32,593", f"{len(clean_df):,}" if clean_df is not None else "32,593"],
            "Columns": ["12", "16", "18", f"{len(clean_df.columns)}" if clean_df is not None else "12"]
        }
        st.table(pd.DataFrame(comp_data))
        
        st.markdown("### ⚖️ Scaling & ML Preparation")
        st.info("""
        **Important:** `StandardScaler` was applied **during model training**, not in the saved dataset.
        """)
        st.markdown("""
        *   **Why?** Distance-based algorithms (KMeans, DBSCAN) are sensitive to scale. `total_clicks` (range 0-10,000) would dominate `avg_quiz_score` (range 0-100) without scaling.
        *   **Method:** We standardize features to have mean=0 and variance=1 immediately before fitting the models.
        """)

    # --- Tab 4: Final Dataset ---
    with tab4:
        st.markdown("### ✨ Final Cleaned Dataset (`lms_clean.csv`)")
        
        if clean_df is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rows (Students)", f"{len(clean_df):,}")
            with col2:
                st.metric("Columns (Features)", f"{len(clean_df.columns)}")
            
            st.markdown("#### Preview")
            st.dataframe(clean_df.head())
            
            with st.expander("View Statistical Summary (Describe)"):
                st.dataframe(clean_df.describe())
                
            st.markdown("#### Feature Dictionary")
            st.markdown("""
            - **id_student**: Unique identifier.
            - **final_result**: Target variable (Pass/Fail/etc).
            - **total_clicks**: Volume of interaction.
            - **active_days**: Consistency of interaction.
            - **avg_daily_clicks**: Intensity of interaction.
            - **quizzes_attempted**: Assessment participation.
            - **avg_quiz_score**: Assessment performance.
            - **date_registration**: Early/Late joiner status.
            """)
        else:
            st.warning("Dataset not loaded. Please upload a file or load the default dataset in the 'Upload' page.")

    # --- Tab 5: Downloads ---
    with tab5:
        st.markdown("### 📥 Export Documentation & Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Cleaned Dataset")
            if clean_df is not None:
                csv = clean_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download lms_clean.csv",
                    data=csv,
                    file_name='lms_clean.csv',
                    mime='text/csv',
                )
            else:
                st.button("Download lms_clean.csv", disabled=True)
        
        with col2:
            st.markdown("#### Preprocessing Report")
            if st.button("Generate PDF Report"):
                pdf_data = generate_preprocessing_report_pdf()
                st.download_button(
                    label="Download Report PDF",
                    data=pdf_data,
                    file_name='lms_preprocessing_report.pdf',
                    mime='application/pdf'
                )
            
            st.markdown("#### Preprocessing Notebook")
            # Path relative to project root
            notebook_rel_path = os.path.join("data", "notebooks", "dataSet_preprocessing.ipynb")
            notebook_abs_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), notebook_rel_path)
            
            if os.path.exists(notebook_abs_path):
                with open(notebook_abs_path, "rb") as f:
                    st.download_button(
                        label="Download Notebook (.ipynb)",
                        data=f,
                        file_name="dataSet_preprocessing.ipynb",
                        mime="application/x-ipynb+json"
                    )
            else:
                st.info(f"Notebook not found at {notebook_rel_path}")
        
        st.markdown("---")
        st.markdown("### 💻 Reproducibility")
        st.markdown("Use this snippet to load the data in your own Python script:")
        st.code("""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Load Data
df = pd.read_csv('lms_clean.csv')

# 2. Select Features
features = ['total_clicks', 'active_days', 'avg_daily_clicks', 'quizzes_attempted', 'avg_quiz_score']
X = df[features]

# 3. Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Train
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
        """, language="python")
