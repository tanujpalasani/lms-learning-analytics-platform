import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
import hdbscan
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import io
import base64

st.set_page_config(
    page_title="LMS Student Analytics Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

TEAL_PALETTE = ['#008080', '#20B2AA', '#40E0D0', '#48D1CC', '#00CED1', '#5F9EA0', '#2F4F4F', '#006666']
CLUSTER_COLORS = ['#008080', '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']

REQUIRED_COLUMNS = ['id_student', 'final_result', 'total_clicks', 'active_days', 
                    'avg_daily_clicks', 'quizzes_attempted', 'avg_quiz_score']
FEATURE_COLUMNS = ['total_clicks', 'active_days', 'avg_daily_clicks', 'quizzes_attempted', 'avg_quiz_score']

def init_session_state():
    defaults = {
        'df': None,
        'scaler': None,
        'scaled_features': None,
        'pca_2d': None,
        'pca_3d': None,
        'pca_2d_data': None,
        'pca_3d_data': None,
        'selected_k': 4,
        'elbow_done': False,
        'models': {},
        'metrics': {},
        'cluster_labels': {},
        'cluster_centroids': {},
        'active_model': None,
        'clustered_df': None,
        'learner_types': {}
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def compute_cluster_centroids(X_scaled, labels):
    unique_labels = np.unique(labels)
    centroids = {}
    for label in unique_labels:
        mask = labels == label
        centroids[label] = X_scaled[mask].mean(axis=0)
    return centroids

def predict_nearest_centroid(input_scaled, centroids):
    min_dist = float('inf')
    predicted_cluster = 0
    for label, centroid in centroids.items():
        dist = np.linalg.norm(input_scaled - centroid)
        if dist < min_dist:
            min_dist = dist
            predicted_cluster = label
    return predicted_cluster

def compute_optimal_eps(X_scaled, min_samples=5):
    """
    Compute optimal eps for DBSCAN using k-nearest neighbor distance analysis.
    Uses the 'elbow' method on the k-distance graph where k = min_samples.
    """
    n_neighbors = min(min_samples, len(X_scaled) - 1)
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(X_scaled)
    distances, _ = nn.kneighbors(X_scaled)
    k_distances = np.sort(distances[:, -1])
    
    n_points = len(k_distances)
    line_start = np.array([0, k_distances[0]])
    line_end = np.array([n_points - 1, k_distances[-1]])
    line_vec = line_end - line_start
    line_len = np.linalg.norm(line_vec)
    line_unit = line_vec / line_len
    
    max_distance = 0
    best_idx = n_points // 2
    
    for i in range(n_points):
        point = np.array([i, k_distances[i]])
        vec_to_point = point - line_start
        proj_length = np.dot(vec_to_point, line_unit)
        proj_point = line_start + proj_length * line_unit
        distance = np.linalg.norm(point - proj_point)
        
        if distance > max_distance:
            max_distance = distance
            best_idx = i
    
    optimal_eps = k_distances[best_idx]
    optimal_eps = max(0.1, min(optimal_eps, 3.0))
    
    return optimal_eps, k_distances

init_session_state()

def apply_custom_css():
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            color: #008080;
            margin-bottom: 1rem;
            text-align: center;
        }
        .sub-header {
            font-size: 1.5rem;
            font-weight: 600;
            color: #2F4F4F;
            margin-bottom: 0.5rem;
        }
        .metric-card {
            background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 4px solid #008080;
            margin-bottom: 1rem;
        }
        .cluster-card {
            background: #ffffff;
            padding: 1.5rem;
            border-radius: 10px;
            border: 1px solid #e0e0e0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }
        .stButton>button {
            background-color: #008080;
            color: white;
            border-radius: 5px;
            border: none;
            padding: 0.5rem 1rem;
        }
        .stButton>button:hover {
            background-color: #006666;
        }
        div[data-testid="stMetric"] {
            background-color: #f0f8f8;
            padding: 1rem;
            border-radius: 8px;
            border-left: 3px solid #008080;
        }
        .info-box {
            background-color: #e6f3f3;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #008080;
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)

apply_custom_css()

def validate_dataset(df):
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        return False, f"Missing required columns: {', '.join(missing)}"
    for col in FEATURE_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            return False, f"Column '{col}' must be numeric"
    return True, "Dataset validation successful"

def assign_learner_types(cluster_stats):
    learner_types = {}
    clusters = cluster_stats.index.tolist()
    
    scores = {}
    for cluster in clusters:
        row = cluster_stats.loc[cluster]
        score = (
            row['total_clicks'] * 0.2 +
            row['active_days'] * 0.2 +
            row['avg_daily_clicks'] * 0.2 +
            row['quizzes_attempted'] * 0.2 +
            row['avg_quiz_score'] * 0.2
        )
        scores[cluster] = score
    
    sorted_clusters = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    
    type_labels = ["Top Performer", "Consistent Learner", "Casual Learner", "At-Risk Learner"]
    
    for i, cluster in enumerate(sorted_clusters):
        if i < len(type_labels):
            learner_types[cluster] = type_labels[i]
        else:
            learner_types[cluster] = f"Learner Group {i+1}"
    
    return learner_types

def get_cluster_description(learner_type, stats):
    descriptions = {
        "Top Performer": {
            "behavior": "High engagement with frequent clicks, many active days, and excellent quiz performance.",
            "intervention": "Provide advanced content, leadership opportunities, and peer mentoring roles."
        },
        "Consistent Learner": {
            "behavior": "Regular engagement patterns with steady activity and good quiz scores.",
            "intervention": "Maintain engagement with challenging content and recognition programs."
        },
        "Casual Learner": {
            "behavior": "Moderate engagement with occasional activity and average quiz attempts.",
            "intervention": "Encourage more frequent participation with reminders and gamification."
        },
        "At-Risk Learner": {
            "behavior": "Low engagement with minimal clicks, few active days, and low quiz participation.",
            "intervention": "Provide immediate support, personalized outreach, and simplified learning paths."
        }
    }
    
    if learner_type in descriptions:
        return descriptions[learner_type]
    else:
        return {
            "behavior": "Mixed engagement patterns requiring further analysis.",
            "intervention": "Monitor closely and provide personalized support as needed."
        }

def page_home():
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

def page_upload():
    st.markdown('<h1 class="main-header">Data Upload</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Upload your cleaned LMS dataset in CSV format. The dataset should contain the following columns:
    - **id_student**: Unique student identifier
    - **final_result**: Student outcome (Pass, Fail, Withdrawn, Distinction)
    - **total_clicks**: Total number of clicks in the LMS
    - **active_days**: Number of days the student was active
    - **avg_daily_clicks**: Average clicks per active day
    - **quizzes_attempted**: Number of quizzes attempted
    - **avg_quiz_score**: Average score across all quizzes
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            is_valid, message = validate_dataset(df)
            
            if is_valid:
                st.success(message)
                st.session_state.df = df
                
                st.session_state.models = {}
                st.session_state.metrics = {}
                st.session_state.cluster_labels = {}
                st.session_state.cluster_centroids = {}
                st.session_state.active_model = None
                st.session_state.clustered_df = None
                st.session_state.elbow_done = False
                
                st.markdown("### Dataset Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Records", f"{len(df):,}")
                with col2:
                    st.metric("Total Features", f"{len(df.columns)}")
                with col3:
                    st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
                
                st.markdown("### Column Information")
                col_info = pd.DataFrame({
                    'Column': df.columns.tolist(),
                    'Data Type': [str(dt) for dt in df.dtypes.values],
                    'Non-Null Count': df.count().values.tolist(),
                    'Null Count': df.isnull().sum().values.tolist()
                })
                st.dataframe(col_info, use_container_width=True)
            else:
                st.error(message)
                
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    
    elif st.session_state.df is not None:
        st.info("Dataset already loaded. You can upload a new file to replace it.")
        st.markdown("### Current Dataset Preview")
        st.dataframe(st.session_state.df.head(10), use_container_width=True)

def page_eda():
    st.markdown('<h1 class="main-header">Exploratory Data Analysis</h1>', unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("Please upload a dataset first.")
        return
    
    df = st.session_state.df
    
    st.markdown("### Summary Statistics")
    st.dataframe(df[FEATURE_COLUMNS].describe(), use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("### Feature Distributions")
    
    fig = make_subplots(rows=2, cols=3, subplot_titles=FEATURE_COLUMNS + [''])
    
    positions = [(1,1), (1,2), (1,3), (2,1), (2,2)]
    
    for i, col in enumerate(FEATURE_COLUMNS):
        row, col_pos = positions[i]
        fig.add_trace(
            go.Histogram(x=df[col], name=col, marker_color=TEAL_PALETTE[i % len(TEAL_PALETTE)],
                        opacity=0.7),
            row=row, col=col_pos
        )
    
    fig.update_layout(height=500, showlegend=False, title_text="Distribution of Behavioral Features")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("### Correlation Heatmap")
    
    corr_matrix = df[FEATURE_COLUMNS].corr()
    
    fig_corr = px.imshow(
        corr_matrix,
        labels=dict(color="Correlation"),
        x=FEATURE_COLUMNS,
        y=FEATURE_COLUMNS,
        color_continuous_scale='Teal',
        aspect='auto'
    )
    fig_corr.update_layout(height=500)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("### Outcome Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        outcome_counts = df['final_result'].value_counts()
        fig_pie = px.pie(
            values=outcome_counts.values,
            names=outcome_counts.index,
            title="Final Result Distribution",
            color_discrete_sequence=TEAL_PALETTE
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        fig_bar = px.bar(
            x=outcome_counts.index,
            y=outcome_counts.values,
            title="Final Result Counts",
            labels={'x': 'Final Result', 'y': 'Count'},
            color=outcome_counts.index,
            color_discrete_sequence=TEAL_PALETTE
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
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
        color_discrete_sequence=TEAL_PALETTE,
        opacity=0.6
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("### Box Plots by Final Result")
    
    selected_feature = st.selectbox("Select Feature", FEATURE_COLUMNS, key="boxplot_feature")
    
    fig_box = px.box(
        df, x='final_result', y=selected_feature,
        title=f"{selected_feature} by Final Result",
        color='final_result',
        color_discrete_sequence=TEAL_PALETTE
    )
    st.plotly_chart(fig_box, use_container_width=True)

def compute_gap_statistic(X, k_range, n_refs=10):
    gaps = []
    gap_errors = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        
        log_wk = np.log(kmeans.inertia_)
        
        ref_inertias = []
        for _ in range(n_refs):
            random_ref = np.random.uniform(X.min(axis=0), X.max(axis=0), X.shape)
            kmeans_ref = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans_ref.fit(random_ref)
            ref_inertias.append(np.log(kmeans_ref.inertia_))
        
        gap = np.mean(ref_inertias) - log_wk
        gap_error = np.std(ref_inertias) * np.sqrt(1 + 1/n_refs)
        
        gaps.append(gap)
        gap_errors.append(gap_error)
    
    return gaps, gap_errors

def page_cluster_count():
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
            
            if include_gap:
                gaps, gap_errors = compute_gap_statistic(X_scaled, k_range, n_refs=5)
            
            if include_gap:
                col1, col2, col3 = st.columns(3)
            else:
                col1, col2 = st.columns(2)
            
            with col1:
                fig_elbow = go.Figure()
                fig_elbow.add_trace(go.Scatter(
                    x=list(k_range), y=inertias,
                    mode='lines+markers',
                    name='Inertia',
                    line=dict(color='#008080', width=3),
                    marker=dict(size=10)
                ))
                fig_elbow.update_layout(
                    title="Elbow Method - Inertia vs K",
                    xaxis_title="Number of Clusters (K)",
                    yaxis_title="Inertia",
                    height=400
                )
                st.plotly_chart(fig_elbow, use_container_width=True)
            
            with col2:
                fig_sil = go.Figure()
                fig_sil.add_trace(go.Scatter(
                    x=list(k_range), y=silhouettes,
                    mode='lines+markers',
                    name='Silhouette Score',
                    line=dict(color='#20B2AA', width=3),
                    marker=dict(size=10)
                ))
                fig_sil.update_layout(
                    title="Silhouette Score vs K",
                    xaxis_title="Number of Clusters (K)",
                    yaxis_title="Silhouette Score",
                    height=400
                )
                st.plotly_chart(fig_sil, use_container_width=True)
            
            if include_gap:
                with col3:
                    fig_gap = go.Figure()
                    fig_gap.add_trace(go.Scatter(
                        x=list(k_range), y=gaps,
                        mode='lines+markers',
                        name='Gap Statistic',
                        line=dict(color='#FF6B6B', width=3),
                        marker=dict(size=10),
                        error_y=dict(type='data', array=gap_errors, visible=True)
                    ))
                    fig_gap.update_layout(
                        title="Gap Statistic vs K",
                        xaxis_title="Number of Clusters (K)",
                        yaxis_title="Gap Statistic",
                        height=400
                    )
                    st.plotly_chart(fig_gap, use_container_width=True)
            
            optimal_k_silhouette = silhouettes.index(max(silhouettes)) + 2
            
            if include_gap:
                optimal_k_gap = 2
                for i in range(len(gaps) - 1):
                    if gaps[i] >= gaps[i+1] - gap_errors[i+1]:
                        optimal_k_gap = i + 2
                        break
                else:
                    optimal_k_gap = gaps.index(max(gaps)) + 2
                
                optimal_k = optimal_k_silhouette
                st.session_state.selected_k = optimal_k
                
                st.markdown("### Optimization Summary")
                opt_col1, opt_col2 = st.columns(2)
                with opt_col1:
                    st.metric("Optimal K (Silhouette)", optimal_k_silhouette)
                with opt_col2:
                    st.metric("Optimal K (Gap Statistic)", optimal_k_gap)
            else:
                optimal_k = optimal_k_silhouette
                st.session_state.selected_k = optimal_k
            
            st.session_state.elbow_done = True
            
            st.success(f"Analysis complete! Suggested optimal K: {optimal_k} (based on highest silhouette score)")
    
    if st.session_state.elbow_done:
        st.markdown("---")
        st.markdown("### Select Number of Clusters")
        
        selected_k = st.slider(
            "Choose the number of clusters (K)",
            min_value=2,
            max_value=10,
            value=st.session_state.selected_k,
            help="Adjust based on the elbow point and silhouette analysis above"
        )
        
        st.session_state.selected_k = selected_k
        
        st.info(f"Selected K = {selected_k}. Proceed to Model Training to train clustering models.")

def page_model_training():
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
                                    help="HDBSCAN automatically determines cluster count based on hierarchical density")
    
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
                        line=dict(color='#008080')
                    ))
                    
                    elbow_idx = np.searchsorted(k_distances, auto_eps_value)
                    fig_kdist.add_hline(y=auto_eps_value, line_dash="dash", 
                                        annotation_text=f"Optimal eps = {auto_eps_value:.3f}",
                                        line_color="red")
                    
                    fig_kdist.update_layout(
                        title="k-Nearest Neighbor Distance Plot",
                        xaxis_title="Points (sorted by distance)",
                        yaxis_title=f"{min_samples}-NN Distance",
                        height=300
                    )
                    st.plotly_chart(fig_kdist, use_container_width=True)
                
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
        train_kmeans = train_gmm = train_agg = train_spectral = train_dbscan = train_hdbscan = True
    
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
        if train_hdbscan:
            models_to_train.append(('HDBSCAN', hdbscan.HDBSCAN(min_cluster_size=min_samples, min_samples=min_samples), 'density'))
        
        if not models_to_train:
            st.warning("Please select at least one model to train.")
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, (name, model, model_type) in enumerate(models_to_train):
            status_text.text(f"Training {name}...")
            
            if model_type == 'gmm':
                model.fit(X_scaled)
                labels = model.predict(X_scaled)
            else:
                labels = model.fit_predict(X_scaled)
            
            unique_labels = set(labels)
            if -1 in unique_labels:
                n_clusters = len(unique_labels) - 1
                noise_count = list(labels).count(-1)
                st.info(f"{name}: Found {n_clusters} clusters with {noise_count} noise points")
            
            valid_mask = labels != -1
            if valid_mask.sum() > 1 and len(set(labels[valid_mask])) > 1:
                sil_score = silhouette_score(X_scaled[valid_mask], labels[valid_mask])
                db_score = davies_bouldin_score(X_scaled[valid_mask], labels[valid_mask])
                ch_score = calinski_harabasz_score(X_scaled[valid_mask], labels[valid_mask])
            else:
                sil_score = 0.0
                db_score = float('inf')
                ch_score = 0.0
            
            centroids = compute_cluster_centroids(X_scaled, labels)
            
            st.session_state.models[name] = model
            st.session_state.cluster_labels[name] = labels
            st.session_state.cluster_centroids[name] = centroids
            st.session_state.metrics[name] = {
                'Silhouette Score': sil_score,
                'Davies-Bouldin Index': db_score,
                'Calinski-Harabasz Score': ch_score
            }
            
            progress_bar.progress((i + 1) / len(models_to_train))
        
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
        
        st.dataframe(metrics_df[['Rank', 'Model', 'Silhouette Score', 'Davies-Bouldin Index', 'Calinski-Harabasz Score']], 
                     use_container_width=True)
        
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
        
        labels = st.session_state.cluster_labels[active_model]
        clustered_df = df.copy()
        clustered_df['Cluster'] = labels
        st.session_state.clustered_df = clustered_df
        
        cluster_stats = clustered_df.groupby('Cluster')[FEATURE_COLUMNS].mean()
        cluster_stats_normalized = (cluster_stats - cluster_stats.min()) / (cluster_stats.max() - cluster_stats.min())
        st.session_state.learner_types = assign_learner_types(cluster_stats_normalized)
        
        st.success(f"Active model set to: {active_model}")

def page_cluster_interpretation():
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
    st.dataframe(cluster_stats.round(2), use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("### Cluster vs Final Result")
    
    crosstab = pd.crosstab(clustered_df['Cluster'], clustered_df['final_result'])
    
    fig_crosstab = px.bar(
        crosstab,
        barmode='group',
        title="Final Result Distribution by Cluster",
        color_discrete_sequence=TEAL_PALETTE
    )
    fig_crosstab.update_layout(xaxis_title="Cluster", yaxis_title="Count")
    st.plotly_chart(fig_crosstab, use_container_width=True)
    
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
                <h3 style="color: #008080;">Cluster {cluster}</h3>
                <h4>{learner_type}</h4>
                <p><strong>Size:</strong> {cluster_size} students</p>
                <p><strong>Behavior:</strong> {desc['behavior']}</p>
                <p><strong>Typical Outcome:</strong> {top_result}</p>
                <p><strong>Recommendation:</strong> {desc['intervention']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### PCA Visualizations")
    
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
                color_discrete_sequence=CLUSTER_COLORS
            )
        else:
            fig_2d = px.scatter(
                pca_df, x='PC1', y='PC2', color='Final Result',
                title="2D PCA - Colored by Final Result",
                color_discrete_sequence=TEAL_PALETTE
            )
        
        fig_2d.update_layout(height=500)
        st.plotly_chart(fig_2d, use_container_width=True)
    
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
                color_discrete_sequence=CLUSTER_COLORS
            )
        else:
            fig_3d = px.scatter_3d(
                pca_3d_df, x='PC1', y='PC2', z='PC3', color='Final Result',
                title="3D PCA - Colored by Final Result",
                color_discrete_sequence=TEAL_PALETTE
            )
        
        fig_3d.update_layout(height=500)
        st.plotly_chart(fig_3d, use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("### Parallel Coordinates Plot")
    
    parallel_df = clustered_df[FEATURE_COLUMNS + ['Cluster']].copy()
    parallel_df['Cluster'] = parallel_df['Cluster'].astype(str)
    
    fig_parallel = px.parallel_coordinates(
        parallel_df,
        dimensions=FEATURE_COLUMNS,
        color=clustered_df['Cluster'],
        color_continuous_scale='Teal',
        title="Feature Comparison Across Clusters"
    )
    fig_parallel.update_layout(height=500)
    st.plotly_chart(fig_parallel, use_container_width=True)
    
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
        color_discrete_sequence=CLUSTER_COLORS
    )
    st.plotly_chart(fig_violin, use_container_width=True)

def page_prediction():
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
        
        if active_model == 'KMeans':
            cluster = model.predict(input_scaled)[0]
        elif active_model == 'GMM':
            cluster = model.predict(input_scaled)[0]
        elif active_model == 'Agglomerative':
            centroids = st.session_state.cluster_centroids[active_model]
            cluster = predict_nearest_centroid(input_scaled[0], centroids)
        else:
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
            <h3 style="color: #008080;">Student Profile Analysis</h3>
            <p><strong>Assigned Cluster:</strong> {cluster}</p>
            <p><strong>Learner Type:</strong> {learner_type}</p>
            <p><strong>Behavior Pattern:</strong> {desc['behavior']}</p>
            <p><strong>Recommended Intervention:</strong> {desc['intervention']}</p>
        </div>
        """, unsafe_allow_html=True)

def page_dashboard():
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
            color_discrete_sequence=CLUSTER_COLORS
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        fig_pie = px.pie(
            values=cluster_counts.values,
            names=[f"Cluster {c}: {learner_types.get(c, 'Group')}" for c in cluster_counts.index],
            title="Cluster Proportions",
            color_discrete_sequence=CLUSTER_COLORS
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("### Cluster vs Outcome Analysis")
    
    crosstab = pd.crosstab(clustered_df['Cluster'], clustered_df['final_result'], normalize='index') * 100
    
    fig_stacked = px.bar(
        crosstab,
        barmode='stack',
        title="Final Result Distribution by Cluster (%)",
        color_discrete_sequence=TEAL_PALETTE
    )
    fig_stacked.update_layout(yaxis_title="Percentage", xaxis_title="Cluster")
    st.plotly_chart(fig_stacked, use_container_width=True)
    
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
                marker_color=TEAL_PALETTE[i % len(TEAL_PALETTE)]
            ),
            row=row, col=col_pos
        )
    
    fig_metrics.update_layout(height=600, showlegend=False, title_text="Average Feature Values per Cluster")
    st.plotly_chart(fig_metrics, use_container_width=True)
    
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

def generate_pdf_report(clustered_df, learner_types, metrics, active_model, k):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#008080'),
        spaceAfter=20,
        alignment=1
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#2F4F4F'),
        spaceBefore=15,
        spaceAfter=10
    )
    normal_style = styles['Normal']
    
    elements = []
    
    elements.append(Paragraph("LMS Student Behavior Analytics Report", title_style))
    elements.append(Spacer(1, 20))
    
    elements.append(Paragraph("Executive Summary", heading_style))
    summary_text = f"""
    This report presents the results of student behavior segmentation analysis using machine learning.
    The analysis identified {k} distinct learner segments based on behavioral patterns including 
    total clicks, active days, quiz attempts, and quiz scores. The active model used for this 
    analysis is {active_model}.
    """
    elements.append(Paragraph(summary_text, normal_style))
    elements.append(Spacer(1, 15))
    
    elements.append(Paragraph("Model Performance Metrics", heading_style))
    
    metrics_table_data = [['Model', 'Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz']]
    for name, m in metrics.items():
        metrics_table_data.append([
            name,
            f"{m['Silhouette Score']:.4f}",
            f"{m['Davies-Bouldin Index']:.4f}",
            f"{m['Calinski-Harabasz Score']:.2f}"
        ])
    
    metrics_table = Table(metrics_table_data, colWidths=[1.5*inch, 1.2*inch, 1.3*inch, 1.5*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#008080')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f0f8f8')),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#008080')),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
    ]))
    elements.append(metrics_table)
    elements.append(Spacer(1, 20))
    
    elements.append(Paragraph("Cluster Profiles", heading_style))
    
    cluster_stats = clustered_df.groupby('Cluster')[FEATURE_COLUMNS].mean()
    cluster_counts = clustered_df['Cluster'].value_counts().sort_index()
    
    for cluster in sorted(clustered_df['Cluster'].unique()):
        if cluster == -1:
            continue
            
        learner_type = learner_types.get(cluster, f"Group {cluster}")
        count = cluster_counts.get(cluster, 0)
        stats = cluster_stats.loc[cluster]
        desc = get_cluster_description(learner_type, stats)
        
        cluster_title = f"Cluster {cluster}: {learner_type} ({count} students)"
        elements.append(Paragraph(cluster_title, ParagraphStyle(
            'ClusterTitle',
            parent=styles['Heading3'],
            fontSize=12,
            textColor=colors.HexColor('#008080'),
            spaceBefore=10,
            spaceAfter=5
        )))
        
        elements.append(Paragraph(f"<b>Behavior:</b> {desc['behavior']}", normal_style))
        elements.append(Paragraph(f"<b>Recommendation:</b> {desc['intervention']}", normal_style))
        
        stats_text = f"Avg Clicks: {stats['total_clicks']:.1f} | Active Days: {stats['active_days']:.1f} | Quiz Score: {stats['avg_quiz_score']:.1f}"
        elements.append(Paragraph(stats_text, normal_style))
        elements.append(Spacer(1, 10))
    
    elements.append(Spacer(1, 20))
    elements.append(Paragraph("Methodology", heading_style))
    methodology_text = """
    The analysis uses StandardScaler for feature normalization and PCA for dimensionality reduction.
    Multiple clustering algorithms were evaluated including KMeans, GMM, and Agglomerative clustering.
    Cluster quality was assessed using Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Score.
    Learner types are assigned based on relative feature values within each cluster.
    """
    elements.append(Paragraph(methodology_text, normal_style))
    
    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()

def page_time_series():
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
            st.plotly_chart(fig_engagement, use_container_width=True)
            
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
            st.plotly_chart(fig_intensity, use_container_width=True)
            
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
            st.plotly_chart(fig_result_trends, use_container_width=True)
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
        st.plotly_chart(fig_timeline, use_container_width=True)
        
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
        st.plotly_chart(fig_timing, use_container_width=True)

def page_vle_engagement():
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
            st.dataframe(vle_df.head(10), use_container_width=True)
            
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
                        
                        global FEATURE_COLUMNS
                        new_features = FEATURE_COLUMNS + selected_vle_features
                        
                        st.success(f"Successfully merged {len(selected_vle_features)} VLE features!")
                        st.info("Go to Cluster Count Selection to re-analyze with enhanced features.")
                        
                        st.markdown("### Enhanced Dataset Preview")
                        st.dataframe(merged_df.head(10), use_container_width=True)
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

def page_about():
    st.markdown('<h1 class="main-header">About This Application</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Problem Statement
    
    Educational institutions generate vast amounts of data through Learning Management Systems (LMS). 
    Understanding student behavior patterns is crucial for:
    
    - Identifying at-risk students early
    - Personalizing learning experiences
    - Optimizing resource allocation
    - Improving overall educational outcomes
    
    This dashboard provides an AI-powered solution for analyzing LMS behavioral data and 
    segmenting students into meaningful learner types.
    
    ---
    
    ## Dataset Description
    
    The application expects a cleaned LMS dataset with the following columns:
    
    | Column | Description |
    |--------|-------------|
    | id_student | Unique student identifier |
    | final_result | Course outcome (Pass, Fail, Withdrawn, Distinction) |
    | total_clicks | Total interactions in the LMS |
    | active_days | Number of days with activity |
    | avg_daily_clicks | Average clicks per active day |
    | quizzes_attempted | Number of quizzes taken |
    | avg_quiz_score | Average score across quizzes |
    
    ---
    
    ## Machine Learning Approach
    
    ### Data Preprocessing
    
    **StandardScaler**: All behavioral features are standardized to have zero mean and unit variance. 
    This ensures that features with larger scales don't dominate the clustering process.
    
    ### Dimensionality Reduction
    
    **Principal Component Analysis (PCA)**: Used to reduce the feature space for visualization while 
    preserving maximum variance. We generate both 2D and 3D representations for interactive exploration.
    
    ### Clustering Algorithms
    
    1. **KMeans**: Partitions data into K clusters by minimizing within-cluster variance. 
       Fast and effective for spherical clusters.
    
    2. **Gaussian Mixture Model (GMM)**: A probabilistic model that assumes data comes from a 
       mixture of Gaussian distributions. More flexible than KMeans for non-spherical clusters.
    
    3. **Agglomerative Clustering**: A hierarchical approach that builds clusters by progressively 
       merging smaller clusters. Good for discovering nested cluster structures.
    
    ### Cluster Count Selection
    
    **Elbow Method**: Plots inertia (within-cluster sum of squares) against K. The "elbow" point 
    where the curve bends indicates a good cluster count.
    
    **Silhouette Analysis**: Measures how similar objects are to their own cluster compared to 
    other clusters. Higher scores indicate better-defined clusters.
    
    ### Evaluation Metrics
    
    - **Silhouette Score** (-1 to 1): Higher values indicate well-separated clusters
    - **Davies-Bouldin Index**: Lower values indicate better clustering
    - **Calinski-Harabasz Score**: Higher values indicate denser, well-separated clusters
    
    ---
    
    ## Phase 1 Scope
    
    This Phase 1 release focuses on LMS behavioral analytics:
    
    - Static behavioral features from LMS data
    - Multiple clustering algorithm support
    - Interactive visualizations
    - New student prediction
    - Export capabilities
    
    ### Future Phase 2 Enhancements
    
    Potential additions for Phase 2:
    
    - Real-time engagement tracking
    - Forum participation analysis
    - Video lecture interaction patterns
    - Predictive early warning systems
    - Integration with live LMS feeds
    
    ---
    
    ## Technical Stack
    
    - **Frontend**: Streamlit
    - **Data Processing**: Pandas, NumPy
    - **Machine Learning**: Scikit-learn
    - **Visualization**: Plotly, Matplotlib, Seaborn
    - **Model Persistence**: Joblib
    
    ---
    
    ## Usage Instructions
    
    1. **Upload Data**: Navigate to the Data Upload page and upload your cleaned LMS CSV file
    2. **Explore Data**: Use the EDA page to understand your dataset
    3. **Determine Clusters**: Run the Elbow analysis to find the optimal number of clusters
    4. **Train Models**: Select and train clustering models
    5. **Interpret Results**: Explore cluster profiles and visualizations
    6. **Make Predictions**: Predict learner types for new students
    7. **Export**: Download clustered data and metrics
    
    ---
    
    *Developed for educational analytics and student success initiatives.*
    """)

def main():
    st.sidebar.markdown("## Navigation")
    
    pages = {
        "Home": page_home,
        "Data Upload": page_upload,
        "Exploratory Data Analysis": page_eda,
        "Cluster Count Selection": page_cluster_count,
        "Model Training": page_model_training,
        "Cluster Interpretation": page_cluster_interpretation,
        "Time-Series Analysis": page_time_series,
        "New Student Prediction": page_prediction,
        "Dashboard and Exports": page_dashboard,
        "VLE Data (Phase 2)": page_vle_engagement,
        "About": page_about
    }
    
    selection = st.sidebar.radio("Go to", list(pages.keys()))
    
    st.sidebar.markdown("---")
    
    st.sidebar.markdown("### Current Status")
    
    if st.session_state.df is not None:
        st.sidebar.success("Dataset loaded")
    else:
        st.sidebar.warning("No dataset")
    
    if st.session_state.elbow_done:
        st.sidebar.success(f"K = {st.session_state.selected_k}")
    else:
        st.sidebar.warning("Cluster count not set")
    
    if st.session_state.models:
        st.sidebar.success(f"Models: {len(st.session_state.models)}")
        if st.session_state.active_model:
            st.sidebar.info(f"Active: {st.session_state.active_model}")
    else:
        st.sidebar.warning("No models trained")
    
    pages[selection]()

if __name__ == "__main__":
    main()
