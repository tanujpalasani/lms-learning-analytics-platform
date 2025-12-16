"""
Constants and configuration values for the LMS Analytics Dashboard.
"""

# Premium Color System - Modern Indigo & Emerald Theme
PRIMARY_COLOR = '#6366f1'  # Indigo 500
SECONDARY_COLOR = '#4f46e5'  # Indigo 600
BACKGROUND_COLOR = '#f8fafc'  # Slate 50

# Sequential Palette (Indigo -> Light Blue) - Replaces TEAL_PALETTE
SEQUENTIAL_PALETTE = ['#312e81', '#3730a3', '#4338ca', '#4f46e5', '#6366f1', '#818cf8', '#a5b4fc', '#c7d2fe']

# Categorical Palette (Distinct & Vibrant) - Replaces CLUSTER_COLORS
# Indices: 0:Indigo, 1:Emerald, 2:Amber, 3:Pink, 4:Violet, 5:Blue, 6:Red, 7:Teal
CATEGORICAL_PALETTE = ['#6366f1', '#10b981', '#f59e0b', '#ec4899', '#8b5cf6', '#3b82f6', '#ef4444', '#14b8a6']

# Backward Compatibility Aliases
TEAL_PALETTE = SEQUENTIAL_PALETTE
CLUSTER_COLORS = CATEGORICAL_PALETTE

# Required dataset columns
REQUIRED_COLUMNS = ['id_student', 'final_result', 'total_clicks', 'active_days', 
                    'avg_daily_clicks', 'quizzes_attempted', 'avg_quiz_score']

# Feature columns for clustering
FEATURE_COLUMNS = ['total_clicks', 'active_days', 'avg_daily_clicks', 'quizzes_attempted', 'avg_quiz_score']
