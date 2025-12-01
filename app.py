import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import io

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
        'active_model': None,
        'clustered_df': None,
        'learner_types': {}
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

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
                    'Column': df.columns,
                    'Data Type': df.dtypes.values,
                    'Non-Null Count': df.count().values,
                    'Null Count': df.isnull().sum().values
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

def page_cluster_count():
    st.markdown('<h1 class="main-header">Cluster Count Selection</h1>', unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("Please upload a dataset first.")
        return
    
    df = st.session_state.df
    
    st.markdown("""
    <div class="info-box">
    Use the Elbow Method and Silhouette Analysis to determine the optimal number of clusters 
    for your data. The elbow point indicates where adding more clusters provides diminishing returns.
    </div>
    """, unsafe_allow_html=True)
    
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
            
            optimal_k = silhouettes.index(max(silhouettes)) + 2
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
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        train_kmeans = st.checkbox("KMeans", value=True)
    with col2:
        train_gmm = st.checkbox("Gaussian Mixture Model (GMM)", value=True)
    with col3:
        train_agg = st.checkbox("Agglomerative Clustering", value=True)
    
    col_btn1, col_btn2 = st.columns(2)
    
    with col_btn1:
        train_selected = st.button("Train Selected Models", type="primary")
    with col_btn2:
        train_all = st.button("Train All Models")
    
    if train_all:
        train_kmeans = train_gmm = train_agg = True
    
    if train_selected or train_all:
        models_to_train = []
        if train_kmeans:
            models_to_train.append(('KMeans', KMeans(n_clusters=k, random_state=42, n_init=10)))
        if train_gmm:
            models_to_train.append(('GMM', GaussianMixture(n_components=k, random_state=42)))
        if train_agg:
            models_to_train.append(('Agglomerative', AgglomerativeClustering(n_clusters=k)))
        
        if not models_to_train:
            st.warning("Please select at least one model to train.")
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, (name, model) in enumerate(models_to_train):
            status_text.text(f"Training {name}...")
            
            if name == 'GMM':
                model.fit(X_scaled)
                labels = model.predict(X_scaled)
            else:
                labels = model.fit_predict(X_scaled)
            
            sil_score = silhouette_score(X_scaled, labels)
            db_score = davies_bouldin_score(X_scaled, labels)
            ch_score = calinski_harabasz_score(X_scaled, labels)
            
            st.session_state.models[name] = model
            st.session_state.cluster_labels[name] = labels
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
        
        if active_model == 'GMM':
            cluster = model.predict(input_scaled)[0]
        else:
            cluster = model.predict(input_scaled)[0] if hasattr(model, 'predict') else model.fit_predict(input_scaled)[0]
        
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
    
    col1, col2 = st.columns(2)
    
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
        "New Student Prediction": page_prediction,
        "Dashboard and Exports": page_dashboard,
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
