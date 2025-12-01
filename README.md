# LMS Student Behavior Analytics Dashboard

An AI-powered Learning Management System (LMS) analytics platform that uses machine learning to segment students into distinct learner types based on their behavioral patterns.

## Overview

This dashboard helps educators and administrators understand student behavior patterns through advanced clustering techniques. By analyzing LMS interaction data, the application identifies distinct learner segments and provides actionable insights for personalized interventions.

## Features

- **Data Upload**: Upload and validate cleaned LMS datasets in CSV format
- **Exploratory Data Analysis**: Comprehensive statistical summaries, distributions, and correlation analysis
- **Cluster Count Selection**: Elbow method and silhouette analysis to determine optimal cluster count
- **Multiple Clustering Models**: Support for KMeans, Gaussian Mixture Model (GMM), and Agglomerative Clustering
- **Model Comparison**: Side-by-side evaluation using Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Score
- **Advanced Visualizations**: Interactive 2D/3D PCA plots, parallel coordinates, and violin plots
- **Cluster Interpretation**: Human-readable learner type labels with intervention recommendations
- **New Student Prediction**: Predict cluster and learner type for new students
- **Export Capabilities**: Download clustered datasets and model metrics as CSV

## Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Dataset Requirements

The application expects a CSV file with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| id_student | Any | Unique student identifier |
| final_result | String | Course outcome (Pass, Fail, Withdrawn, Distinction) |
| total_clicks | Numeric | Total number of LMS interactions |
| active_days | Numeric | Number of days with activity |
| avg_daily_clicks | Numeric | Average clicks per active day |
| quizzes_attempted | Numeric | Number of quizzes taken |
| avg_quiz_score | Numeric | Average score across quizzes |

## Usage Guide

### Step 1: Upload Data
Navigate to the "Data Upload" page and upload your cleaned LMS CSV file. The system will validate required columns and display a preview.

### Step 2: Explore Data
Use the "Exploratory Data Analysis" page to understand your dataset through:
- Summary statistics
- Feature distributions
- Correlation heatmaps
- Outcome analysis

### Step 3: Determine Cluster Count
Run the Elbow analysis on the "Cluster Count Selection" page to find the optimal number of clusters (K). Adjust the slider based on the elbow point and silhouette scores.

### Step 4: Train Models
On the "Model Training" page:
- Select which models to train (KMeans, GMM, Agglomerative)
- Click "Train Selected Models" or "Train All Models"
- Compare model performance metrics
- Select the active model for analysis

### Step 5: Interpret Clusters
The "Cluster Interpretation" page provides:
- Cluster statistics and profiles
- Learner type labels (Top Performer, Consistent Learner, etc.)
- 2D and 3D PCA visualizations
- Parallel coordinates plots
- Feature distribution by cluster

### Step 6: Predict New Students
Enter behavioral metrics for a new student on the "New Student Prediction" page to predict their cluster and learner type.

### Step 7: Export Results
Use the "Dashboard and Exports" page to:
- View summary dashboards
- Download clustered data as CSV
- Export model metrics

## Clustering Algorithms

### KMeans
Partitions data into K clusters by minimizing within-cluster variance. Fast and effective for spherical clusters.

### Gaussian Mixture Model (GMM)
A probabilistic model assuming data comes from a mixture of Gaussian distributions. More flexible for non-spherical clusters.

### Agglomerative Clustering
A hierarchical approach that builds clusters by progressively merging smaller clusters. Good for discovering nested structures.

## Evaluation Metrics

- **Silhouette Score**: Measures cluster cohesion and separation (-1 to 1, higher is better)
- **Davies-Bouldin Index**: Measures cluster similarity (lower is better)
- **Calinski-Harabasz Score**: Ratio of between-cluster to within-cluster variance (higher is better)

## Learner Types

The system automatically assigns human-readable labels based on cluster profiles:

- **Top Performer**: High engagement with excellent quiz performance
- **Consistent Learner**: Regular activity with good scores
- **Casual Learner**: Moderate engagement with occasional activity
- **At-Risk Learner**: Low engagement requiring intervention

## Technical Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Model Persistence**: Joblib

## Phase 1 Scope

This Phase 1 release focuses on LMS behavioral analytics with static features. Future Phase 2 enhancements may include:
- Real-time engagement tracking
- Forum participation analysis
- Video lecture interaction patterns
- Predictive early warning systems

## License

This project is developed for educational analytics and student success initiatives.
