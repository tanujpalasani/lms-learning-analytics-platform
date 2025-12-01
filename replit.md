# LMS Student Behavior Analytics Dashboard

## Overview
An AI-powered Learning Management System (LMS) analytics platform that uses machine learning to segment students into distinct learner types based on their behavioral patterns.

## Current State
Phase 1 implementation complete with:
- Multi-page Streamlit dashboard
- Multiple clustering algorithms (KMeans, GMM, Agglomerative)
- Interactive 2D/3D PCA visualizations
- New student prediction capability
- CSV export functionality

## Project Architecture

### Main Files
- `app.py` - Main Streamlit application with all pages and logic
- `README.md` - Project documentation
- `.streamlit/config.toml` - Streamlit server configuration

### Application Pages
1. **Home** - Welcome page with overview and quick start guide
2. **Data Upload** - CSV upload with validation for required columns
3. **Exploratory Data Analysis** - Distributions, correlations, outcome analysis
4. **Cluster Count Selection** - Elbow method and silhouette analysis
5. **Model Training** - Train KMeans, GMM, Agglomerative with metrics comparison
6. **Cluster Interpretation** - Profiles, PCA visualization, parallel coordinates
7. **New Student Prediction** - Predict learner type for new students
8. **Dashboard & Exports** - Summary charts and CSV download
9. **About** - Documentation and methodology explanation

### Key Features
- StandardScaler for feature normalization
- PCA for dimensionality reduction (2D and 3D)
- Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Score for model evaluation
- Nearest-centroid prediction for Agglomerative clustering
- Session state management for data persistence

## Technical Stack
- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn
- **Visualization**: Plotly, Matplotlib, Seaborn

## Dataset Requirements
Required columns:
- id_student
- final_result (Pass, Fail, Withdrawn, Distinction)
- total_clicks
- active_days
- avg_daily_clicks
- quizzes_attempted
- avg_quiz_score

## Recent Changes
- Initial implementation of complete Phase 1 dashboard
- Added multi-model clustering support
- Implemented nearest-centroid prediction for Agglomerative clustering
- Created professional UI with teal/blue color palette

## Running the Application
```bash
streamlit run app.py --server.port 5000
```
