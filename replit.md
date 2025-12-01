# LMS Student Behavior Analytics Dashboard

## Overview
An AI-powered Learning Management System (LMS) analytics platform that uses machine learning to segment students into distinct learner types based on their behavioral patterns.

## Current State
Phase 2 implementation complete with:
- Multi-page Streamlit dashboard (11 pages)
- 6 clustering algorithms (KMeans, GMM, Agglomerative, DBSCAN, HDBSCAN, Spectral)
- Interactive 2D/3D PCA visualizations
- New student prediction capability
- CSV and PDF export functionality
- Time-series engagement analysis
- VLE data integration support

## Project Architecture

### Main Files
- `app.py` - Main Streamlit application with all pages and logic
- `README.md` - Project documentation
- `.streamlit/config.toml` - Streamlit server configuration

### Application Pages
1. **Home** - Welcome page with overview and quick start guide
2. **Data Upload** - CSV upload with validation for required columns
3. **Exploratory Data Analysis** - Distributions, correlations, outcome analysis
4. **Cluster Count Selection** - Elbow method, silhouette analysis, and Gap statistic
5. **Model Training** - Train 6 clustering algorithms with metrics comparison
6. **Cluster Interpretation** - Profiles, PCA visualization, parallel coordinates
7. **Time-Series Analysis** - Engagement patterns and trends over time
8. **New Student Prediction** - Predict learner type for new students
9. **Dashboard & Exports** - Summary charts, CSV download, PDF report generation
10. **VLE Data (Phase 2)** - Upload and merge additional VLE engagement data
11. **About** - Documentation and methodology explanation

### Key Features
- StandardScaler for feature normalization
- PCA for dimensionality reduction (2D and 3D)
- Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Score for model evaluation
- Gap statistic for automated optimal cluster count detection
- Nearest-centroid prediction for clustering models without predict method
- Automatic eps parameter detection for DBSCAN using k-NN
- PDF report generation with cluster insights using ReportLab
- Session state management for data persistence

### Clustering Algorithms
1. **KMeans** - Centroid-based clustering
2. **GMM (Gaussian Mixture Models)** - Probabilistic soft clustering
3. **Agglomerative** - Hierarchical bottom-up clustering
4. **DBSCAN** - Density-based with automatic eps detection
5. **HDBSCAN** - Hierarchical density-based clustering
6. **Spectral** - Graph-based clustering using eigenvectors

## Technical Stack
- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, HDBSCAN
- **Visualization**: Plotly, Matplotlib, Seaborn
- **PDF Generation**: ReportLab

## Dataset Requirements
Required columns:
- id_student
- final_result (Pass, Fail, Withdrawn, Distinction)
- total_clicks
- active_days
- avg_daily_clicks
- quizzes_attempted
- avg_quiz_score

Optional columns for enhanced analysis:
- date_registration (for time-series analysis)
- date_unregistration

## Recent Changes
- Added DBSCAN, HDBSCAN, and Spectral Clustering algorithms
- Implemented Gap statistic for automated cluster count optimization
- Created Time-Series Analysis page with engagement patterns
- Added PDF report generation with cluster insights
- Created VLE Data page for Phase 2 data integration
- Updated navigation to 11 pages

## UI Design
- Professional teal/blue color palette (#008080, #20B2AA, #2F4F4F)
- No emojis in UI elements
- Clean card-based layout with consistent styling

## Running the Application
```bash
streamlit run app.py --server.port 5000
```
