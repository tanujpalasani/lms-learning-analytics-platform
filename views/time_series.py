"""
Time-series engagement analysis page.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.constants import FEATURE_COLUMNS, CLUSTER_COLORS


def render():
    """Render the time-series analysis page."""
    st.markdown('<h1 class="main-header">Time-Series Engagement Analysis</h1>', unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("Please upload a dataset first.")
        return
    
    df = st.session_state.df
    
    st.markdown("""
    <div class="info-box">
    Analyze student engagement patterns over time using registration data. This analysis helps identify 
    trends in student activity and the relationship between registration timing and outcomes.
    </div>
    """, unsafe_allow_html=True)
    
    has_date_reg = 'date_registration' in df.columns
    has_date_unreg = 'date_unregistration' in df.columns
    
    if not has_date_reg:
        st.warning("date_registration column not found. Limited time-series analysis available.")
        
        st.markdown("### Available Feature Trend Analysis")
        st.markdown("Simulating engagement timeline based on behavioral features.")
        
        if st.session_state.clustered_df is not None:
            clustered_df = st.session_state.clustered_df
            learner_types = st.session_state.learner_types
            
            st.markdown("### Engagement Patterns by Cluster")
            
            cluster_engagement = clustered_df.groupby('Cluster').agg({
                'total_clicks': 'mean',
                'active_days': 'mean',
                'avg_daily_clicks': 'mean',
                'quizzes_attempted': 'mean',
                'avg_quiz_score': 'mean'
            }).round(2)
            
            fig_engagement = go.Figure()
            
            for cluster in cluster_engagement.index:
                learner_type = learner_types.get(cluster, f"Cluster {cluster}")
                values = cluster_engagement.loc[cluster].values
                normalized_values = (values - values.min()) / (values.max() - values.min() + 0.001)
                
                fig_engagement.add_trace(go.Scatterpolar(
                    r=list(normalized_values) + [normalized_values[0]],
                    theta=FEATURE_COLUMNS + [FEATURE_COLUMNS[0]],
                    fill='toself',
                    name=f"Cluster {cluster}: {learner_type}"
                ))
            
            fig_engagement.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                title="Normalized Feature Profiles by Cluster (Radar Chart)"
            )
            st.plotly_chart(fig_engagement)
            
            st.markdown("### Engagement Intensity Distribution")
            
            clustered_df['engagement_score'] = (
                clustered_df['total_clicks'] / clustered_df['total_clicks'].max() * 0.25 +
                clustered_df['active_days'] / clustered_df['active_days'].max() * 0.25 +
                clustered_df['avg_daily_clicks'] / clustered_df['avg_daily_clicks'].max() * 0.25 +
                clustered_df['quizzes_attempted'] / clustered_df['quizzes_attempted'].max() * 0.25
            )
            
            fig_intensity = px.histogram(
                clustered_df,
                x='engagement_score',
                color='Cluster',
                nbins=30,
                title="Engagement Score Distribution by Cluster",
                color_discrete_sequence=CLUSTER_COLORS
            )
            st.plotly_chart(fig_intensity)
            
            st.markdown("### Activity Trends by Final Result")
            
            activity_by_result = df.groupby('final_result')[FEATURE_COLUMNS].mean()
            
            fig_result_trends = go.Figure()
            for result in activity_by_result.index:
                fig_result_trends.add_trace(go.Bar(
                    name=result,
                    x=FEATURE_COLUMNS,
                    y=activity_by_result.loc[result].values
                ))
            
            fig_result_trends.update_layout(
                barmode='group',
                title="Average Feature Values by Final Result",
                xaxis_title="Feature",
                yaxis_title="Average Value"
            )
            st.plotly_chart(fig_result_trends)
        else:
            st.info("Train a model first to see cluster-based engagement analysis.")
    else:
        st.markdown("### Registration Timeline Analysis")
        
        df_with_dates = df.copy()
        df_with_dates['date_registration'] = pd.to_numeric(df_with_dates['date_registration'], errors='coerce')
        
        if has_date_unreg:
            df_with_dates['date_unregistration'] = pd.to_numeric(df_with_dates['date_unregistration'], errors='coerce')
            df_with_dates['active_duration'] = df_with_dates['date_unregistration'] - df_with_dates['date_registration']
        
        reg_by_date = df_with_dates.groupby('date_registration').agg({
            'id_student': 'count',
            'total_clicks': 'mean',
            'avg_quiz_score': 'mean'
        }).reset_index()
        reg_by_date.columns = ['Registration Day', 'Student Count', 'Avg Clicks', 'Avg Quiz Score']
        
        fig_timeline = make_subplots(rows=2, cols=1, 
                                      subplot_titles=['Registration Volume Over Time', 'Average Engagement Over Time'],
                                      vertical_spacing=0.15)
        
        fig_timeline.add_trace(
            go.Scatter(x=reg_by_date['Registration Day'], y=reg_by_date['Student Count'],
                      mode='lines+markers', name='Registrations', line=dict(color='#008080')),
            row=1, col=1
        )
        
        fig_timeline.add_trace(
            go.Scatter(x=reg_by_date['Registration Day'], y=reg_by_date['Avg Clicks'],
                      mode='lines+markers', name='Avg Clicks', line=dict(color='#20B2AA')),
            row=2, col=1
        )
        
        fig_timeline.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig_timeline)
        
        st.markdown("### Early vs Late Registrants Analysis")
        
        median_reg = df_with_dates['date_registration'].median()
        df_with_dates['registration_timing'] = np.where(
            df_with_dates['date_registration'] <= median_reg, 'Early', 'Late'
        )
        
        timing_comparison = df_with_dates.groupby('registration_timing')[FEATURE_COLUMNS].mean()
        
        fig_timing = px.bar(
            timing_comparison.T,
            barmode='group',
            title="Feature Comparison: Early vs Late Registrants",
            color_discrete_sequence=['#008080', '#FF6B6B']
        )
        st.plotly_chart(fig_timing)
