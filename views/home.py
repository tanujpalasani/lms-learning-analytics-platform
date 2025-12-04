"""
Home page for LMS Student Behavior Analytics Dashboard.
"""
import streamlit as st


def render():
    """Render the home page."""
    st.markdown('<h1 class="main-header">LMS Student Behavior Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <strong>Welcome to the AI-Powered Learning Management System Analytics Platform</strong><br><br>
    This dashboard helps educators and administrators understand student behavior patterns 
    through advanced machine learning techniques. Upload your LMS data to discover 
    distinct learner segments and gain actionable insights.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Data Analysis")
        st.write("Upload and explore your LMS dataset with comprehensive exploratory data analysis.")
    
    with col2:
        st.markdown("### ML Segmentation")
        st.write("Apply multiple clustering algorithms to identify distinct student behavior patterns.")
    
    with col3:
        st.markdown("### Actionable Insights")
        st.write("Get personalized intervention recommendations for each learner segment.")
    
    st.markdown("---")
    
    st.markdown("### Quick Start Guide")
    steps = [
        "Upload your cleaned LMS dataset (CSV format)",
        "Explore data through the EDA page",
        "Determine optimal cluster count using the Elbow method",
        "Train clustering models (KMeans, GMM, Agglomerative)",
        "Interpret clusters and view learner profiles",
        "Predict segments for new students",
        "Export results and generate reports"
    ]
    
    for i, step in enumerate(steps, 1):
        st.write(f"**Step {i}:** {step}")
