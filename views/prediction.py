"""
New student prediction page for classifying individual students.
"""
import streamlit as st
import numpy as np
from utils.helpers import get_cluster_description
from utils.clustering import predict_nearest_centroid
from utils.constants import PRIMARY_COLOR


def render():
    """Render the new student prediction page."""
    st.markdown('<h1 class="main-header">New Student Prediction</h1>', unsafe_allow_html=True)
    
    if not st.session_state.models:
        st.warning("Please train at least one model first.")
        return
    
    if st.session_state.active_model is None:
        st.warning("Please select an active model.")
        return
    
    active_model = st.session_state.active_model
    st.info(f"Using model: {active_model}")
    
    st.markdown("### Enter Student Behavior Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        total_clicks = st.number_input("Total Clicks", min_value=0, value=500, step=10)
        active_days = st.number_input("Active Days", min_value=0, value=30, step=1)
        avg_daily_clicks = st.number_input("Average Daily Clicks", min_value=0.0, value=15.0, step=0.5)
    
    with col2:
        quizzes_attempted = st.number_input("Quizzes Attempted", min_value=0, value=5, step=1)
        avg_quiz_score = st.number_input("Average Quiz Score", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
    
    if st.button("Predict Learner Type", type="primary"):
        input_data = np.array([[total_clicks, active_days, avg_daily_clicks, quizzes_attempted, avg_quiz_score]])
        
        scaler = st.session_state.scaler
        input_scaled = scaler.transform(input_data)
        
        model = st.session_state.models[active_model]
        
        # Predict cluster based on model type
        if active_model == 'KMeans':
            cluster = model.predict(input_scaled)[0]
        elif active_model == 'GMM':
            cluster = model.predict(input_scaled)[0]
        elif active_model == 'Agglomerative':
            centroids = st.session_state.cluster_centroids[active_model]
            cluster = predict_nearest_centroid(input_scaled[0], centroids)
        else:
            # For other models, use nearest centroid
            centroids = st.session_state.cluster_centroids.get(active_model)
            if centroids:
                cluster = predict_nearest_centroid(input_scaled[0], centroids)
            else:
                cluster = 0
        
        learner_type = st.session_state.learner_types.get(cluster, f"Group {cluster}")
        desc = get_cluster_description(learner_type, None)
        
        st.markdown("---")
        st.markdown("### Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Cluster ID", cluster)
        with col2:
            st.metric("Learner Type", learner_type)
        with col3:
            st.metric("Model Used", active_model)
        
        st.markdown(f"""
        <div class="cluster-card">
            <h3 style="color: {PRIMARY_COLOR};">Student Profile Analysis</h3>
            <p><strong>Assigned Cluster:</strong> {cluster}</p>
            <p><strong>Learner Type:</strong> {learner_type}</p>
            <p><strong>Behavior Pattern:</strong> {desc['behavior']}</p>
            <p><strong>Recommended Intervention:</strong> {desc['intervention']}</p>
        </div>
        """, unsafe_allow_html=True)
