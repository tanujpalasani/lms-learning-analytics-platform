"""
Cluster count selection page with elbow method, silhouette analysis, and gap statistic.
"""
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from utils.constants import FEATURE_COLUMNS
from utils.clustering import compute_gap_statistic


def calculate_elbow_point(inertias):
    """
    Calculate the elbow point using the maximum distance from the line connecting the first and last points.
    """
    n_points = len(inertias)
    all_coords = np.vstack((range(n_points), inertias)).T
    first_point = all_coords[0]
    last_point = all_coords[-1]
    line_vec = last_point - first_point
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
    vec_from_first = all_coords - first_point
    scalar_product = np.sum(vec_from_first * line_vec_norm, axis=1)
    vec_from_first_parallel = np.outer(scalar_product, line_vec_norm)
    vec_to_line = vec_from_first - vec_from_first_parallel
    dist_to_line = np.sqrt(np.sum(vec_to_line ** 2, axis=1))
    idx_of_best_point = np.argmax(dist_to_line)
    return idx_of_best_point + 2  # +2 because range starts at 2


def render():
    """Render the cluster count selection page."""
    st.markdown('<h1 class="main-header">Cluster Count Selection</h1>', unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("Please upload a dataset first.")
        return
    
    df = st.session_state.df
    
    st.markdown("""
    <div class="info-box">
    Use the Elbow Method, Silhouette Analysis, and Gap Statistic to determine the optimal number of clusters 
    for your data. Multiple metrics provide more robust cluster count recommendations.
    </div>
    """, unsafe_allow_html=True)
    
    include_gap = st.checkbox("Include Gap Statistic Analysis (slower but more robust)", value=False,
                              help="Gap statistic compares within-cluster dispersion to a null reference distribution")
    
    if st.button("Run Cluster Analysis", type="primary"):
        with st.spinner("Analyzing optimal cluster count..."):
            X = df[FEATURE_COLUMNS].values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            st.session_state.scaler = scaler
            st.session_state.scaled_features = X_scaled
            
            pca_2d = PCA(n_components=2)
            pca_3d = PCA(n_components=3)
            
            st.session_state.pca_2d = pca_2d
            st.session_state.pca_3d = pca_3d
            st.session_state.pca_2d_data = pca_2d.fit_transform(X_scaled)
            st.session_state.pca_3d_data = pca_3d.fit_transform(X_scaled)
            
            k_range = range(2, 11)
            inertias = []
            silhouettes = []
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(X_scaled)
                inertias.append(kmeans.inertia_)
                silhouettes.append(silhouette_score(X_scaled, kmeans.labels_))
            
            # --- Recommendations Logic ---
            recommendations = []
            
            # 1. Silhouette Score Recommendation
            best_sil_idx = silhouettes.index(max(silhouettes))
            optimal_k_sil = best_sil_idx + 2
            recommendations.append({
                "k": optimal_k_sil,
                "method": "Silhouette Score",
                "reason": f"Highest silhouette score ({max(silhouettes):.3f}), indicating well-separated clusters."
            })
            
            # 2. Elbow Method Recommendation
            optimal_k_elbow = calculate_elbow_point(inertias)
            recommendations.append({
                "k": optimal_k_elbow,
                "method": "Elbow Method",
                "reason": "Point of maximum curvature (diminishing returns) in inertia plot."
            })
            
            # 3. Gap Statistic Recommendation (if enabled)
            optimal_k_gap = None
            if include_gap:
                gaps, gap_errors = compute_gap_statistic(X_scaled, k_range, n_refs=5)
                
                optimal_k_gap = 2
                for i in range(len(gaps) - 1):
                    if gaps[i] >= gaps[i+1] - gap_errors[i+1]:
                        optimal_k_gap = i + 2
                        break
                else:
                    optimal_k_gap = gaps.index(max(gaps)) + 2
                
                recommendations.append({
                    "k": optimal_k_gap,
                    "method": "Gap Statistic",
                    "reason": "Statistically significant gap between data and random distribution."
                })
            
            # Save results to session state
            st.session_state.cluster_results = {
                "k_range": list(k_range),
                "inertias": inertias,
                "silhouettes": silhouettes,
                "gaps": gaps if include_gap else None,
                "gap_errors": gap_errors if include_gap else None,
                "recommendations": recommendations
            }
            
            # Default to Silhouette recommendation
            st.session_state.selected_k = optimal_k_sil
            st.session_state.elbow_done = True
    
    # --- Display Results (if analysis is done) ---
    if st.session_state.elbow_done and hasattr(st.session_state, 'cluster_results'):
        results = st.session_state.cluster_results
        k_range = results["k_range"]
        recommendations = results["recommendations"]
        
        # Plotting
        if results["gaps"]:
            col1, col2, col3 = st.columns(3)
        else:
            col1, col2 = st.columns(2)
        
        with col1:
            fig_elbow = go.Figure()
            fig_elbow.add_trace(go.Scatter(
                x=k_range, y=results["inertias"],
                mode='lines+markers', name='Inertia',
                line=dict(color='#008080', width=3), marker=dict(size=10)
            ))
            fig_elbow.update_layout(title="Elbow Method (Inertia)", xaxis_title="K", yaxis_title="Inertia", height=350)
            st.plotly_chart(fig_elbow, width="stretch")
        
        with col2:
            fig_sil = go.Figure()
            fig_sil.add_trace(go.Scatter(
                x=k_range, y=results["silhouettes"],
                mode='lines+markers', name='Silhouette',
                line=dict(color='#20B2AA', width=3), marker=dict(size=10)
            ))
            fig_sil.update_layout(title="Silhouette Score", xaxis_title="K", yaxis_title="Score", height=350)
            st.plotly_chart(fig_sil, width="stretch")
            
        if results["gaps"]:
            with col3:
                fig_gap = go.Figure()
                fig_gap.add_trace(go.Scatter(
                    x=k_range, y=results["gaps"],
                    mode='lines+markers', name='Gap',
                    line=dict(color='#FF6B6B', width=3), marker=dict(size=10),
                    error_y=dict(type='data', array=results["gap_errors"], visible=True)
                ))
                fig_gap.update_layout(title="Gap Statistic", xaxis_title="K", yaxis_title="Gap", height=350)
                st.plotly_chart(fig_gap, width="stretch")
        
        st.markdown("---")
        st.markdown("### 💡 Recommended Cluster Counts")
        
        # Display recommendations in columns
        rec_cols = st.columns(len(recommendations))
        for i, rec in enumerate(recommendations):
            with rec_cols[i]:
                st.info(f"**Option {i+1}: K = {rec['k']}**\n\n"
                        f"*{rec['method']}*\n\n"
                        f"{rec['reason']}")
        
        st.markdown("---")
        st.markdown("### ✅ Final Selection")
        
        # Create a list of recommended K values for quick selection
        recommended_k_values = sorted(list(set(r['k'] for r in recommendations)))
        
        # Radio button for quick selection from recommendations
        selection_mode = st.radio(
            "How would you like to select K?",
            ["Choose from Recommendations", "Manual Selection (Slider)"],
            horizontal=True
        )
        
        if selection_mode == "Choose from Recommendations":
            selected_k = st.radio(
                "Select a recommended K:",
                recommended_k_values,
                index=recommended_k_values.index(st.session_state.selected_k) if st.session_state.selected_k in recommended_k_values else 0,
                horizontal=True
            )
        else:
            selected_k = st.slider(
                "Manually select K:",
                min_value=2,
                max_value=10,
                value=st.session_state.selected_k
            )
        
        st.session_state.selected_k = selected_k
        
        st.success(f"**Selected K = {selected_k}**. You can now proceed to **Model Training**.")
    
    # --- Manual Selection (Skip Analysis) ---
    # Only show if analysis hasn't been run (because analysis view has its own selection logic)
    if "cluster_results" not in st.session_state:
        st.markdown("---")
        st.markdown("### ⏭️ Manual Selection (Skip Analysis)")
        st.markdown("Already know the number of clusters you want? Select it below to skip the analysis step.")
        
        col_man1, col_man2 = st.columns([2, 1])
        
        with col_man1:
            manual_k = st.slider("Select Number of Clusters (K)", 2, 10, 4, key="manual_k_slider")
            
        with col_man2:
            st.markdown("<br>", unsafe_allow_html=True) # Spacer
            if st.button("Confirm Manual K", type="primary"):
                # Perform necessary scaling even if skipping analysis
                X = df[FEATURE_COLUMNS].values
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                st.session_state.scaler = scaler
                st.session_state.scaled_features = X_scaled
                
                # Set state
                st.session_state.selected_k = manual_k
                st.session_state.elbow_done = True
                
                st.success(f"Set K = {manual_k}. Ready for training!")
                st.rerun()
