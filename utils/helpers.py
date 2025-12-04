"""
Helper functions for data validation, learner type assignment, and general utilities.
"""
import pandas as pd
from .constants import REQUIRED_COLUMNS, FEATURE_COLUMNS


def init_session_state():
    """Initialize session state variables."""
    import streamlit as st
    
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


def validate_dataset(df):
    """
    Validate that the dataset contains all required columns.
    
    Args:
        df: Pandas DataFrame to validate
        
    Returns:
        tuple: (is_valid: bool, message: str)
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        return False, f"Missing required columns: {', '.join(missing)}"
    
    for col in FEATURE_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            return False, f"Column '{col}' must be numeric"
    
    return True, "Dataset validation successful"


def assign_learner_types(cluster_stats):
    """
    Assign human-readable learner type labels to clusters based on their statistics.
    
    Args:
        cluster_stats: DataFrame with cluster statistics (normalized)
        
    Returns:
        dict: Mapping of cluster ID to learner type label
    """
    learner_types = {}
    clusters = cluster_stats.index.tolist()
    
    # Calculate composite score for each cluster
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
    
    # Sort clusters by score
    sorted_clusters = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    
    # Assign labels
    type_labels = ["Top Performer", "Consistent Learner", "Casual Learner", "At-Risk Learner"]
    
    for i, cluster in enumerate(sorted_clusters):
        if i < len(type_labels):
            learner_types[cluster] = type_labels[i]
        else:
            learner_types[cluster] = f"Learner Group {i+1}"
    
    return learner_types


def get_cluster_description(learner_type, stats):
    """
    Get behavior description and intervention recommendation for a learner type.
    
    Args:
        learner_type: String learner type label
        stats: Cluster statistics (not used currently, for future enhancement)
        
    Returns:
        dict: {'behavior': str, 'intervention': str}
    """
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
