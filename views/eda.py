"""
Exploratory Data Analysis page with statistics and visualizations.
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.constants import FEATURE_COLUMNS, SEQUENTIAL_PALETTE, CATEGORICAL_PALETTE


def render():
    """Render the EDA page."""
    st.markdown('<h1 class="main-header">Exploratory Data Analysis</h1>', unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("Please upload a dataset first.")
        return
    
    df = st.session_state.df
    
    st.markdown("### Summary Statistics")
    st.dataframe(df[FEATURE_COLUMNS].describe())
    
    st.markdown("---")
    
    st.markdown("### Feature Distributions")
    
    fig = make_subplots(rows=2, cols=3, subplot_titles=FEATURE_COLUMNS + [''])
    
    positions = [(1,1), (1,2), (1,3), (2,1), (2,2)]
    
    for i, col in enumerate(FEATURE_COLUMNS):
        row, col_pos = positions[i]
        fig.add_trace(
            go.Histogram(x=df[col], name=col, marker_color=SEQUENTIAL_PALETTE[i % len(SEQUENTIAL_PALETTE)],
                        opacity=0.7),
            row=row, col=col_pos
        )
    
    fig.update_layout(height=500, showlegend=False, title_text="Distribution of Behavioral Features")
    st.plotly_chart(fig)
    
    st.markdown("---")
    
    st.markdown("### Correlation Heatmap")
    
    corr_matrix = df[FEATURE_COLUMNS].corr()
    
    fig_corr = px.imshow(
        corr_matrix,
        labels=dict(color="Correlation"),
        x=FEATURE_COLUMNS,
        y=FEATURE_COLUMNS,
        color_continuous_scale='Blues',
        aspect='auto'
    )
    fig_corr.update_layout(height=500)
    st.plotly_chart(fig_corr)
    
    st.markdown("---")
    
    st.markdown("### Outcome Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        outcome_counts = df['final_result'].value_counts()
        fig_pie = px.pie(
            values=outcome_counts.values,
            names=outcome_counts.index,
            title="Final Result Distribution",
            color_discrete_sequence=CATEGORICAL_PALETTE
        )
        st.plotly_chart(fig_pie)
    
    with col2:
        fig_bar = px.bar(
            x=outcome_counts.index,
            y=outcome_counts.values,
            title="Final Result Counts",
            labels={'x': 'Final Result', 'y': 'Count'},
            color=outcome_counts.index,
            color_discrete_sequence=CATEGORICAL_PALETTE
        )
        st.plotly_chart(fig_bar)
    
    st.markdown("---")
    
    st.markdown("### Feature Relationships")
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_axis = st.selectbox("X-Axis", FEATURE_COLUMNS, index=0)
    with col2:
        y_axis = st.selectbox("Y-Axis", FEATURE_COLUMNS, index=4)
    
    fig_scatter = px.scatter(
        df, x=x_axis, y=y_axis, color='final_result',
        title=f"{x_axis} vs {y_axis}",
        color_discrete_sequence=CATEGORICAL_PALETTE,
        opacity=0.6
    )
    st.plotly_chart(fig_scatter)
    
    st.markdown("---")
    
    st.markdown("### Box Plots by Final Result")
    
    selected_feature = st.selectbox("Select Feature", FEATURE_COLUMNS, key="boxplot_feature")
    
    fig_box = px.box(
        df, x='final_result', y=selected_feature,
        title=f"{selected_feature} by Final Result",
        color='final_result',
        color_discrete_sequence=CATEGORICAL_PALETTE
    )
    st.plotly_chart(fig_box)
