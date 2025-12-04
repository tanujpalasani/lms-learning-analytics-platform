"""
Utility functions for the Dataset & Preprocessing overview page.
"""
import pandas as pd
import streamlit as st
import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import io

# Path to the dataset folder (relative to this file)
# utils/ is one level deep, so we go up one level to root, then into data/
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

def load_clean_dataset(file_path=None):
    """
    Load the cleaned dataset from session state or file.
    """
    if 'df' in st.session_state and st.session_state.df is not None:
        return st.session_state.df
    
    if file_path is None:
        file_path = os.path.join(PROCESSED_DATA_DIR, 'lms_clean.csv')
    
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    
    return None

@st.cache_data
def get_raw_table_info():
    """
    Returns information about the raw OULAD tables.
    Attempts to read from the raw data folder if it exists.
    """
    
    # Default descriptions
    descriptions = {
        "courses.csv": "Information about the module presentations offered.",
        "assessments.csv": "Information about assessments in module presentations.",
        "vle.csv": "Information about the available materials in the VLE.",
        "studentInfo.csv": "Demographic and registration information about students.",
        "studentRegistration.csv": "Information about student registration dates.",
        "studentAssessment.csv": "Results of students' assessments.",
        "studentVle.csv": "Logs of students' interactions with the VLE."
    }
    
    info_dict = {}
    
    if os.path.exists(RAW_DATA_DIR):
        for filename, desc in descriptions.items():
            file_path = os.path.join(RAW_DATA_DIR, filename)
            if os.path.exists(file_path):
                try:
                    # Read header for columns
                    df_head = pd.read_csv(file_path, nrows=5)
                    columns = df_head.columns.tolist()
                    
                    # Get row count efficiently
                    if filename == "studentVle.csv":
                         # Reading just one column is faster
                         df_len = pd.read_csv(file_path, usecols=[0]).shape[0]
                    else:
                        df_len = pd.read_csv(file_path, usecols=[0]).shape[0]
                        
                    info_dict[filename] = {
                        "description": desc,
                        "columns": columns,
                        "rows": f"{df_len:,}"
                    }
                except Exception as e:
                    # Fallback if read fails
                    info_dict[filename] = {
                        "description": f"{desc} (Error reading file: {str(e)})",
                        "columns": ["Error reading columns"],
                        "rows": "Unknown"
                    }
            else:
                 info_dict[filename] = {
                    "description": f"{desc} (File not found)",
                    "columns": [],
                    "rows": "N/A"
                }
    else:
        # Fallback to hardcoded if directory doesn't exist
        return {
            "courses.csv": {
                "description": "Information about the module presentations offered.",
                "columns": ["code_module", "code_presentation", "length"],
                "rows": "22"
            },
            "assessments.csv": {
                "description": "Information about assessments in module presentations.",
                "columns": ["code_module", "code_presentation", "id_assessment", "assessment_type", "date", "weight"],
                "rows": "206"
            },
            "vle.csv": {
                "description": "Information about the available materials in the VLE.",
                "columns": ["id_site", "code_module", "code_presentation", "activity_type", "week_from", "week_to"],
                "rows": "6,364"
            },
            "studentInfo.csv": {
                "description": "Demographic and registration information about students.",
                "columns": ["code_module", "code_presentation", "id_student", "gender", "region", "highest_education", "imd_band", "age_band", "num_of_prev_attempts", "studied_credits", "disability", "final_result"],
                "rows": "32,593"
            },
            "studentRegistration.csv": {
                "description": "Information about student registration dates.",
                "columns": ["code_module", "code_presentation", "id_student", "date_registration", "date_unregistration"],
                "rows": "32,593"
            },
            "studentAssessment.csv": {
                "description": "Results of students' assessments.",
                "columns": ["id_assessment", "id_student", "date_submitted", "is_banked", "score"],
                "rows": "173,912"
            },
            "studentVle.csv": {
                "description": "Logs of students' interactions with the VLE.",
                "columns": ["code_module", "code_presentation", "id_student", "id_site", "date", "sum_click"],
                "rows": "10,655,280"
            }
        }
        
    return info_dict

def generate_preprocessing_report_pdf():
    """
    Generate a PDF report summarizing the preprocessing pipeline.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], alignment=1, spaceAfter=20)
    story.append(Paragraph("LMS Dataset & Preprocessing Report", title_style))
    
    # Overview
    story.append(Paragraph("1. Dataset Overview", styles['Heading2']))
    text = """
    The dataset used in this analysis is derived from the Open University Learning Analytics Dataset (OULAD). 
    It contains data about courses, students, and their interactions with the Virtual Learning Environment (VLE) 
    for seven selected courses (modules).
    """
    story.append(Paragraph(text, styles['Normal']))
    story.append(Spacer(1, 12))

    # Pipeline
    story.append(Paragraph("2. Preprocessing Pipeline", styles['Heading2']))
    steps = [
        "1. Aggregated VLE engagement data (clicks, active days) from studentVle.csv.",
        "2. Calculated assessment performance (quiz attempts, average scores) from studentAssessment.csv.",
        "3. Extracted registration timelines (date_registration, date_unregistration) from studentRegistration.csv.",
        "4. Merged all features onto the base studentInfo.csv table.",
        "5. Handled missing values (filled numeric NaNs with 0).",
        "6. Filtered for valid final results."
    ]
    for step in steps:
        story.append(Paragraph(step, styles['Normal']))
        story.append(Spacer(1, 6))

    # Features
    story.append(Paragraph("3. Final Feature Set", styles['Heading2']))
    features = [
        "total_clicks: Sum of clicks across all materials",
        "active_days: Count of unique days with activity",
        "avg_daily_clicks: Intensity of engagement",
        "quizzes_attempted: Number of assessments submitted",
        "avg_quiz_score: Mean score of assessments",
        "date_registration: Days relative to course start",
        "unregistered_flag: Binary indicator of withdrawal"
    ]
    for feat in features:
        story.append(Paragraph(f"- {feat}", styles['Normal']))
        story.append(Spacer(1, 6))

    doc.build(story)
    buffer.seek(0)
    return buffer
