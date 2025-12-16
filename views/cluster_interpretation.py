"""
Cluster interpretation page with visualizations and learner profiles.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from utils.constants import FEATURE_COLUMNS, PRIMARY_COLOR, CATEGORICAL_PALETTE
from utils.helpers import get_cluster_description


from sklearn.decomposition import PCA

def render():
    """Render the cluster interpretation page."""
    st.markdown('<h1 class="main-header">Cluster Interpretation</h1>', unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("Please upload a dataset first.")
        return
    
    if not st.session_state.models:
        st.warning("Please train at least one model first.")
        return
    
    if st.session_state.active_model is None:
        st.warning("Please select an active model.")
        return
    
    active_model = st.session_state.active_model
    clustered_df = st.session_state.clustered_df
    learner_types = st.session_state.learner_types
    
    st.info(f"Analyzing clusters from: {active_model}")
    
    st.markdown("### Cluster Statistics")
    
    cluster_stats = clustered_df.groupby('Cluster')[FEATURE_COLUMNS].mean()
    st.dataframe(cluster_stats.round(2))
    
    st.markdown("---")
    
    st.markdown("### Cluster vs Final Result")
    
    crosstab = pd.crosstab(clustered_df['Cluster'], clustered_df['final_result'])
    
    fig_crosstab = px.bar(
        crosstab,
        barmode='group',
        title="Final Result Distribution by Cluster",
        color_discrete_sequence=CATEGORICAL_PALETTE
    )
    fig_crosstab.update_layout(xaxis_title="Cluster", yaxis_title="Count")
    st.plotly_chart(fig_crosstab)
    
    st.markdown("---")
    
    st.markdown("### Learner Profile Cards")
    
    n_clusters = len(clustered_df['Cluster'].unique())
    cols = st.columns(min(n_clusters, 4))
    
    for i, cluster in enumerate(sorted(clustered_df['Cluster'].unique())):
        col_idx = i % len(cols)
        
        with cols[col_idx]:
            learner_type = learner_types.get(cluster, f"Group {cluster}")
            stats = cluster_stats.loc[cluster]
            cluster_size = (clustered_df['Cluster'] == cluster).sum()
            
            desc = get_cluster_description(learner_type, stats)
            
            result_dist = clustered_df[clustered_df['Cluster'] == cluster]['final_result'].value_counts()
            top_result = result_dist.index[0] if len(result_dist) > 0 else "N/A"
            
            st.markdown(f"""
            <div class="cluster-card">
                <h3 style="color: {PRIMARY_COLOR};">Cluster {cluster}</h3>
                <h4>{learner_type}</h4>
                <p><strong>Size:</strong> {cluster_size} students</p>
                <p><strong>Behavior:</strong> {desc['behavior']}</p>
                <p><strong>Typical Outcome:</strong> {top_result}</p>
                <p><strong>Recommendation:</strong> {desc['intervention']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### PCA Visualizations")
    
    # Ensure PCA data exists (might be missing if Manual Selection was used)
    if st.session_state.pca_2d_data is None or st.session_state.pca_3d_data is None:
        if st.session_state.scaled_features is not None:
            with st.spinner("Computing PCA for visualization..."):
                pca_2d = PCA(n_components=2)
                pca_3d = PCA(n_components=3)
                st.session_state.pca_2d_data = pca_2d.fit_transform(st.session_state.scaled_features)
                st.session_state.pca_3d_data = pca_3d.fit_transform(st.session_state.scaled_features)
        else:
            st.error("Missing scaled features. Please re-run the analysis from the beginning.")
            return

    pca_2d_data = st.session_state.pca_2d_data
    pca_3d_data = st.session_state.pca_3d_data
    labels = st.session_state.cluster_labels[active_model]
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 2D PCA Visualization")
        
        color_by = st.radio("Color by", ["Cluster", "Final Result"], key="pca_2d_color", horizontal=True)
        
        pca_df = pd.DataFrame({
            'PC1': pca_2d_data[:, 0],
            'PC2': pca_2d_data[:, 1],
            'Cluster': [str(l) for l in labels],
            'Final Result': clustered_df['final_result'].values
        })
        
        if color_by == "Cluster":
            fig_2d = px.scatter(
                pca_df, x='PC1', y='PC2', color='Cluster',
                title="2D PCA - Colored by Cluster",
                color_discrete_sequence=CATEGORICAL_PALETTE
            )
        else:
            fig_2d = px.scatter(
                pca_df, x='PC1', y='PC2', color='Final Result',
                title="2D PCA - Colored by Final Result",
                color_discrete_sequence=CATEGORICAL_PALETTE
            )
        
        fig_2d.update_layout(height=500)
        st.plotly_chart(fig_2d)
    
    with col2:
        st.markdown("#### 3D PCA Visualization")
        
        color_by_3d = st.radio("Color by", ["Cluster", "Final Result"], key="pca_3d_color", horizontal=True)
        
        pca_3d_df = pd.DataFrame({
            'PC1': pca_3d_data[:, 0],
            'PC2': pca_3d_data[:, 1],
            'PC3': pca_3d_data[:, 2],
            'Cluster': [str(l) for l in labels],
            'Final Result': clustered_df['final_result'].values
        })
        
        if color_by_3d == "Cluster":
            fig_3d = px.scatter_3d(
                pca_3d_df, x='PC1', y='PC2', z='PC3', color='Cluster',
                title="3D PCA - Colored by Cluster",
                color_discrete_sequence=CATEGORICAL_PALETTE
            )
        else:
            fig_3d = px.scatter_3d(
                pca_3d_df, x='PC1', y='PC2', z='PC3', color='Final Result',
                title="3D PCA - Colored by Final Result",
                color_discrete_sequence=CATEGORICAL_PALETTE
            )
        
        fig_3d.update_layout(height=500)
        st.plotly_chart(fig_3d)
    
    st.markdown("---")
    
    st.markdown("### Parallel Coordinates Plot")
    
    parallel_df = clustered_df[FEATURE_COLUMNS + ['Cluster']].copy()
    parallel_df['Cluster'] = parallel_df['Cluster'].astype(str)
    
    fig_parallel = px.parallel_coordinates(
        parallel_df,
        dimensions=FEATURE_COLUMNS,
        color=clustered_df['Cluster'],
        color_continuous_scale='Blues',
        title="Feature Comparison Across Clusters"
    )
    fig_parallel.update_layout(height=500)
    st.plotly_chart(fig_parallel)
    
    st.markdown("---")
    
    st.markdown("### Feature Distributions by Cluster")
    
    selected_feature = st.selectbox("Select Feature", FEATURE_COLUMNS, key="violin_feature")
    
    fig_violin = px.violin(
        clustered_df,
        x='Cluster',
        y=selected_feature,
        color='Cluster',
        box=True,
        title=f"{selected_feature} Distribution by Cluster",
        color_discrete_sequence=CATEGORICAL_PALETTE
    )
    st.plotly_chart(fig_violin)
