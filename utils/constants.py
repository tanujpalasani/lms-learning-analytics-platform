"""
Constants and configuration values for the LMS Analytics Dashboard.
"""

# Color palettes
TEAL_PALETTE = ['#008080', '#20B2AA', '#40E0D0', '#48D1CC', '#00CED1', '#5F9EA0', '#2F4F4F', '#006666']
CLUSTER_COLORS = ['#008080', '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']

# Required dataset columns
REQUIRED_COLUMNS = ['id_student', 'final_result', 'total_clicks', 'active_days', 
                    'avg_daily_clicks', 'quizzes_attempted', 'avg_quiz_score']

# Feature columns for clustering
FEATURE_COLUMNS = ['total_clicks', 'active_days', 'avg_daily_clicks', 'quizzes_attempted', 'avg_quiz_score']
