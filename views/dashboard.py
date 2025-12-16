"""
Dashboard and exports page with visualizations and download options.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.constants import FEATURE_COLUMNS, SEQUENTIAL_PALETTE, CATEGORICAL_PALETTE, CLUSTER_COLORS
from utils.pdf_generator import generate_pdf_report


def render():
    """Render the dashboard and exports page."""
    st.markdown('<h1 class="main-header">Dashboard and Exports</h1>', unsafe_allow_html=True)
    
    if not st.session_state.models:
        st.warning("Please train at least one model first.")
        return
    
    if st.session_state.clustered_df is None:
        st.warning("Please select an active model first.")
        return
    
    active_model = st.session_state.active_model
    clustered_df = st.session_state.clustered_df
    learner_types = st.session_state.learner_types
    
    st.info(f"Dashboard for: {active_model}")
    
    st.markdown("### Cluster Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        cluster_counts = clustered_df['Cluster'].value_counts().sort_index()
        
        fig_dist = px.bar(
            x=[f"Cluster {c}" for c in cluster_counts.index],
            y=cluster_counts.values,
            title="Students per Cluster",
            labels={'x': 'Cluster', 'y': 'Count'},
            color=cluster_counts.index.astype(str),
            color_discrete_sequence=CATEGORICAL_PALETTE
        )
        st.plotly_chart(fig_dist)
    
    with col2:
        fig_pie = px.pie(
            values=cluster_counts.values,
            names=[f"Cluster {c}: {learner_types.get(c, 'Group')}" for c in cluster_counts.index],
            title="Cluster Proportions",
            color_discrete_sequence=CATEGORICAL_PALETTE
        )
        st.plotly_chart(fig_pie)
    
    st.markdown("---")
    
    st.markdown("### Cluster vs Outcome Analysis")
    
    crosstab = pd.crosstab(clustered_df['Cluster'], clustered_df['final_result'], normalize='index') * 100
    
    fig_stacked = px.bar(
        crosstab,
        barmode='stack',
        title="Final Result Distribution by Cluster (%)",
        color_discrete_sequence=CATEGORICAL_PALETTE
    )
    fig_stacked.update_layout(yaxis_title="Percentage", xaxis_title="Cluster")
    st.plotly_chart(fig_stacked)
    
    st.markdown("---")
    
    st.markdown("### Average Metrics by Cluster")
    
    cluster_means = clustered_df.groupby('Cluster')[FEATURE_COLUMNS].mean()
    
    fig_metrics = make_subplots(rows=2, cols=3, subplot_titles=FEATURE_COLUMNS + [''])
    
    positions = [(1,1), (1,2), (1,3), (2,1), (2,2)]
    
    for i, col in enumerate(FEATURE_COLUMNS):
        row, col_pos = positions[i]
        fig_metrics.add_trace(
            go.Bar(
                x=[f"Cluster {c}" for c in cluster_means.index],
                y=cluster_means[col].values,
                name=col,
                marker_color=CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)]
            ),
            row=row, col=col_pos
        )
    
    fig_metrics.update_layout(height=600, showlegend=False, title_text="Average Feature Values per Cluster")
    st.plotly_chart(fig_metrics)
    
    st.markdown("---")
    
    st.markdown("### Export Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Clustered Dataset")
        
        export_df = clustered_df.copy()
        export_df['Learner_Type'] = export_df['Cluster'].map(learner_types)
        
        csv_data = export_df.to_csv(index=False)
        
        st.download_button(
            label="Download Clustered Dataset (CSV)",
            data=csv_data,
            file_name="lms_clustered_data.csv",
            mime="text/csv"
        )
    
    with col2:
        st.markdown("#### Model Metrics")
        
        metrics_data = []
        for name, metrics in st.session_state.metrics.items():
            metrics_data.append({
                'Model': name,
                'Silhouette Score': metrics['Silhouette Score'],
                'Davies-Bouldin Index': metrics['Davies-Bouldin Index'],
                'Calinski-Harabasz Score': metrics['Calinski-Harabasz Score']
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_csv = metrics_df.to_csv(index=False)
        
        st.download_button(
            label="Download Model Metrics (CSV)",
            data=metrics_csv,
            file_name="model_metrics.csv",
            mime="text/csv"
        )
    
    with col3:
        st.markdown("#### PDF Report")
        
        if st.button("Generate PDF Report"):
            pdf_buffer = generate_pdf_report(
                clustered_df, 
                learner_types, 
                st.session_state.metrics, 
                active_model,
                st.session_state.selected_k
            )
            
            st.download_button(
                label="Download PDF Report",
                data=pdf_buffer,
                file_name="lms_cluster_report.pdf",
                mime="application/pdf"
            )
