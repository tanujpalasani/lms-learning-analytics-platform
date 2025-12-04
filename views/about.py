"""
About page with technical documentation and methodology.
"""
import streamlit as st


def render():
    """Render the about page."""
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
    | final_result | Course outcome (Pass, Fail, With drawn, Distinction) |
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
