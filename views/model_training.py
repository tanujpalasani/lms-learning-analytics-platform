"""
Model training page with support for multiple clustering algorithms.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
try:
    from sklearn.cluster import HDBSCAN
except ImportError:
    HDBSCAN = None
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from utils.constants import FEATURE_COLUMNS, PRIMARY_COLOR, CATEGORICAL_PALETTE
from utils.clustering import compute_cluster_centroids, compute_optimal_eps
from utils.helpers import assign_learner_types


def render():
    """Render the model training page."""
    st.markdown('<h1 class="main-header">Model Training</h1>', unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("Please upload a dataset first.")
        return
    
    if not st.session_state.elbow_done:
        st.warning("Please run cluster analysis first to determine the optimal number of clusters.")
        return
    
    df = st.session_state.df
    k = st.session_state.selected_k
    X_scaled = st.session_state.scaled_features
    
    st.info(f"Training models with K = {k} clusters")
    
    st.markdown("### Select Models to Train")
    st.markdown("**Standard Clustering Algorithms** (use selected K)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        train_kmeans = st.checkbox("KMeans", value=True)
    with col2:
        train_gmm = st.checkbox("Gaussian Mixture Model (GMM)", value=True)
    with col3:
        if len(df) > 5000:
            st.warning("⚠️ Agglomerative disabled (Dataset > 5k rows)")
            train_agg = False
            st.caption("Agglomerative Clustering requires O(N²) memory. For 32k rows, it needs ~8GB RAM, which exceeds cloud limits.")
        else:
            train_agg = st.checkbox("Agglomerative Clustering", value=True)
    
    st.markdown("**Advanced Clustering Algorithms**")
    col4, col5, col6 = st.columns(3)
    
    with col4:
        train_spectral = st.checkbox("Spectral Clustering", value=False)
    with col5:
        train_dbscan = st.checkbox("DBSCAN (auto clusters)", value=False, 
                                   help="DBSCAN automatically determines cluster count based on density")
    with col6:
        train_hdbscan = st.checkbox("HDBSCAN (auto clusters)", value=False,
                                    help="HDBSCAN automatically determines cluster count based on hierarchical density") if HDBSCAN else False
    
    auto_eps_value = None
    k_distances = None
    
    if train_dbscan or train_hdbscan:
        st.markdown("**Density-Based Parameters**")
        
        db_col1, db_col2 = st.columns(2)
        with db_col1:
            min_samples = st.slider("Min samples per cluster", 2, 20, 5,
                                    help="Minimum points required to form a dense region")
        with db_col2:
            use_auto_eps = st.checkbox("Auto-detect eps (recommended)", value=True,
                                       help="Automatically compute optimal eps using k-NN distance analysis")
        
        if train_dbscan:
            if use_auto_eps:
                with st.spinner("Computing optimal eps using k-NN analysis..."):
                    auto_eps_value, k_distances = compute_optimal_eps(X_scaled, min_samples)
                
                st.markdown(f"**Detected optimal eps: {auto_eps_value:.3f}**")
                
                with st.expander("View k-Distance Plot (eps detection visualization)"):
                    fig_kdist = go.Figure()
                    fig_kdist.add_trace(go.Scatter(
                        y=k_distances,
                        mode='lines',
                        name='k-distances',
                        line=dict(color=PRIMARY_COLOR)
                    ))
                    
                    elbow_idx = np.searchsorted(k_distances, auto_eps_value)
                    fig_kdist.add_hline(y=auto_eps_value, line_dash="dash", 
                                        annotation_text=f"Optimal eps = {auto_eps_value:.3f}",
                                        line_color=CATEGORICAL_PALETTE[6])
                    
                    fig_kdist.update_layout(
                        title="k-Nearest Neighbor Distance Plot",
                        xaxis_title="Points (sorted by distance)",
                        yaxis_title=f"{min_samples}-NN Distance",
                        height=300
                    )
                    st.plotly_chart(fig_kdist)
                
                allow_override = st.checkbox("Override auto-detected eps", value=False)
                if allow_override:
                    eps_value = st.slider("Manual eps override", 0.1, 3.0, float(auto_eps_value), 0.05,
                                          help="Override the automatically detected eps value")
                else:
                    eps_value = auto_eps_value
            else:
                eps_value = st.slider("DBSCAN eps (neighborhood size)", 0.1, 3.0, 0.5, 0.05,
                                      help="Maximum distance between points in a neighborhood")
        else:
            eps_value = 0.5
    else:
        eps_value = 0.5
        min_samples = 5
    
    col_btn1, col_btn2 = st.columns(2)
    
    with col_btn1:
        train_selected = st.button("Train Selected Models", type="primary")
    with col_btn2:
        train_all = st.button("Train All Models")
    
    if train_all:
        train_kmeans = train_gmm = train_agg = train_spectral = train_dbscan = True
        if HDBSCAN:
            train_hdbscan = True
    
    if train_selected or train_all:
        models_to_train = []
        if train_kmeans:
            models_to_train.append(('KMeans', KMeans(n_clusters=k, random_state=42, n_init=10), 'standard'))
        if train_gmm:
            models_to_train.append(('GMM', GaussianMixture(n_components=k, random_state=42), 'gmm'))
        if train_agg:
            models_to_train.append(('Agglomerative', AgglomerativeClustering(n_clusters=k), 'standard'))
        if train_spectral:
            models_to_train.append(('Spectral', SpectralClustering(n_clusters=k, random_state=42, affinity='nearest_neighbors'), 'standard'))
        if train_dbscan:
            models_to_train.append(('DBSCAN', DBSCAN(eps=eps_value, min_samples=min_samples), 'density'))
        if train_hdbscan and HDBSCAN:
            models_to_train.append(('HDBSCAN', HDBSCAN(min_cluster_size=min_samples, min_samples=min_samples), 'density'))
        
        if not models_to_train:
            st.warning("Please select at least one model to train.")
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, (name, model, model_type) in enumerate(models_to_train):
            try:
                status_text.text(f"Training {name}...")
                
                if model_type == 'gmm':
                    model.fit(X_scaled)
                    labels = model.predict(X_scaled)
                else:
                    labels = model.fit_predict(X_scaled)
                
                labels = np.asarray(labels, dtype=int)
                
                unique_labels = set(labels)
                if -1 in unique_labels:
                    n_clusters = len(unique_labels) - 1
                    noise_count = int(list(labels).count(-1))
                    st.info(f"{name}: Found {n_clusters} clusters with {noise_count} noise points")
                
                valid_mask = labels != -1
                valid_labels = labels[valid_mask]
                X_valid = X_scaled[valid_mask]
                
                sil_score = 0.0
                db_score = float('inf')
                ch_score = 0.0
                
                if X_valid.shape[0] > 1 and len(np.unique(valid_labels)) > 1:
                    try:
                        sil_score = float(silhouette_score(X_valid, valid_labels))
                    except:
                        sil_score = 0.0
                    try:
                        db_score = float(davies_bouldin_score(X_valid, valid_labels))
                    except:
                        db_score = float('inf')
                    try:
                        ch_score = float(calinski_harabasz_score(X_valid, valid_labels))
                    except:
                        ch_score = 0.0
                
                centroids = compute_cluster_centroids(X_scaled, labels)
                if not centroids:
                    centroids = {0: X_scaled.mean(axis=0)}
                
                st.session_state.models[name] = model
                st.session_state.cluster_labels[name] = labels
                st.session_state.cluster_centroids[name] = centroids
                st.session_state.metrics[name] = {
                    'Silhouette Score': sil_score,
                    'Davies-Bouldin Index': db_score,
                    'Calinski-Harabasz Score': ch_score
                }
                
                progress_bar.progress((i + 1) / len(models_to_train))
                
            except Exception as e:
                st.error(f"Error training {name}: {str(e)}")
                status_text.text(f"Error training {name}")
                continue
        
        status_text.text("Training complete!")
        
        if st.session_state.active_model is None or st.session_state.active_model not in st.session_state.models:
            st.session_state.active_model = list(st.session_state.models.keys())[0]
        
        st.success(f"Successfully trained {len(models_to_train)} model(s)!")
    
    if st.session_state.models:
        st.markdown("---")
        st.markdown("### Model Performance Comparison")
        
        metrics_data = []
        for name, metrics in st.session_state.metrics.items():
            metrics_data.append({
                'Model': name,
                'Silhouette Score': round(metrics['Silhouette Score'], 4),
                'Davies-Bouldin Index': round(metrics['Davies-Bouldin Index'], 4),
                'Calinski-Harabasz Score': round(metrics['Calinski-Harabasz Score'], 2)
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df = metrics_df.sort_values('Silhouette Score', ascending=False)
        metrics_df['Rank'] = range(1, len(metrics_df) + 1)
        
        st.dataframe(metrics_df[['Rank', 'Model', 'Silhouette Score', 'Davies-Bouldin Index', 'Calinski-Harabasz Score']])
        
        st.markdown("""
        **Metric Interpretation:**
        - **Silhouette Score**: Higher is better (range: -1 to 1)
        - **Davies-Bouldin Index**: Lower is better
        - **Calinski-Harabasz Score**: Higher is better
        """)
        
        st.markdown("---")
        st.markdown("### Select Active Model")
        
        active_model = st.selectbox(
            "Choose the model to use for analysis and predictions",
            options=list(st.session_state.models.keys()),
            index=list(st.session_state.models.keys()).index(st.session_state.active_model) 
                  if st.session_state.active_model in st.session_state.models else 0
        )
        
        st.session_state.active_model = active_model
        
        try:
            labels = st.session_state.cluster_labels[active_model]
            clustered_df = df.copy()
            clustered_df['Cluster'] = labels
            st.session_state.clustered_df = clustered_df
            
            cluster_stats = clustered_df.groupby('Cluster')[FEATURE_COLUMNS].mean()
            
            if cluster_stats.empty:
                st.session_state.learner_types = {0: "Group 1"}
            else:
                cluster_min = cluster_stats.min()
                cluster_max = cluster_stats.max()
                
                range_vals = cluster_max - cluster_min
                range_vals = range_vals.replace(0, 1)
                
                cluster_stats_normalized = (cluster_stats - cluster_min) / range_vals
                st.session_state.learner_types = assign_learner_types(cluster_stats_normalized)
            
            st.success(f"Active model set to: {active_model}")
        except Exception as e:
            st.error(f"Error setting active model: {str(e)}")
            st.session_state.learner_types = {}
