"""
VLE (Virtual Learning Environment) data integration page for Phase 2.
"""
import streamlit as st
import pandas as pd
import numpy as np


def render():
    """Render the VLE engagement page."""
    st.markdown('<h1 class="main-header">Phase 2: VLE Engagement Data</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <strong>Phase 2 Enhancement</strong><br><br>
    This page supports integration of additional Virtual Learning Environment (VLE) data 
    for enhanced student segmentation. Upload VLE interaction data to enrich the behavioral analysis.
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("Please upload a base LMS dataset first in the Data Upload page.")
        return
    
    st.markdown("### Upload VLE Engagement Data")
    st.markdown("""
    VLE data can include:
    - **Forum activity**: Posts, replies, views
    - **Resource access**: PDF downloads, video watches, page views
    - **Assignment submissions**: Timeliness, revision count
    - **Collaboration metrics**: Group activity, peer interactions
    """)
    
    vle_file = st.file_uploader("Upload VLE Data (CSV)", type=['csv'], key='vle_upload')
    
    if vle_file is not None:
        try:
            vle_df = pd.read_csv(vle_file)
            st.success("VLE data loaded successfully!")
            
            st.markdown("### VLE Data Preview")
            st.dataframe(vle_df.head(10))
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("VLE Records", f"{len(vle_df):,}")
            with col2:
                st.metric("VLE Features", f"{len(vle_df.columns)}")
            with col3:
                if 'id_student' in vle_df.columns:
                    matched = len(set(vle_df['id_student']) & set(st.session_state.df['id_student']))
                    st.metric("Matched Students", f"{matched:,}")
                else:
                    st.metric("Matched Students", "N/A")
            
            if 'id_student' in vle_df.columns:
                st.markdown("### Merge with LMS Data")
                
                vle_numeric_cols = vle_df.select_dtypes(include=[np.number]).columns.tolist()
                vle_numeric_cols = [c for c in vle_numeric_cols if c != 'id_student']
                
                if vle_numeric_cols:
                    selected_vle_features = st.multiselect(
                        "Select VLE features to include in analysis",
                        vle_numeric_cols,
                        default=vle_numeric_cols[:3] if len(vle_numeric_cols) >= 3 else vle_numeric_cols
                    )
                    
                    if st.button("Merge VLE Data", type="primary"):
                        vle_agg = vle_df.groupby('id_student')[selected_vle_features].sum().reset_index()
                        
                        merged_df = st.session_state.df.merge(vle_agg, on='id_student', how='left')
                        merged_df[selected_vle_features] = merged_df[selected_vle_features].fillna(0)
                        
                        st.session_state.df = merged_df
                        
                        st.success(f"Successfully merged {len(selected_vle_features)} VLE features!")
                        st.info("Go to Cluster Count Selection to re-analyze with enhanced features.")
                        
                        st.markdown("### Enhanced Dataset Preview")
                        st.dataframe(merged_df.head(10))
                else:
                    st.warning("No numeric columns found in VLE data for analysis.")
            else:
                st.warning("VLE data must contain 'id_student' column for merging.")
                
        except Exception as e:
            st.error(f"Error loading VLE data: {str(e)}")
    
    st.markdown("---")
    st.markdown("### VLE Feature Suggestions")
    
    st.markdown("""
    **Recommended VLE columns for enhanced segmentation:**
    
    | Feature | Description | Impact on Segmentation |
    |---------|-------------|------------------------|
    | forum_posts | Number of forum posts created | Measures active participation |
    | forum_replies | Number of replies to others | Measures collaboration |
    | resource_views | Total resource access count | Measures content engagement |
    | video_watched | Video content consumed | Measures multimedia engagement |
    | assignment_early | Assignments submitted early | Measures time management |
    | peer_feedback | Peer feedback given | Measures community involvement |
    
    Including these features can reveal hidden patterns in student engagement beyond basic LMS metrics.
    """)
