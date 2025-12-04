"""
PDF report generation utilities.
"""
import io
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

from .constants import FEATURE_COLUMNS
from .helpers import get_cluster_description


def generate_pdf_report(clustered_df, learner_types, metrics, active_model, k):
    """
    Generate a comprehensive PDF report with cluster analysis results.
    
    Args:
        clustered_df: DataFrame with cluster assignments
        learner_types: Dictionary mapping cluster IDs to learner type labels
        metrics: Dictionary of model performance metrics
        active_model: Name of the active model
        k: Number of clusters
        
    Returns:
        bytes: PDF file content
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#008080'),
        spaceAfter=20,
        alignment=1  # Center alignment
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
    
    # Title
    elements.append(Paragraph("LMS Student Behavior Analytics Report", title_style))
    elements.append(Spacer(1, 20))
    
    # Executive Summary
    elements.append(Paragraph("Executive Summary", heading_style))
    summary_text = f"""
    This report presents the results of student behavior segmentation analysis using machine learning.
    The analysis identified {k} distinct learner segments based on behavioral patterns including 
    total clicks, active days, quiz attempts, and quiz scores. The active model used for this 
    analysis is {active_model}.
    """
    elements.append(Paragraph(summary_text, normal_style))
    elements.append(Spacer(1, 15))
    
    # Model Performance Metrics
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
    
    # Cluster Profiles
    elements.append(Paragraph("Cluster Profiles", heading_style))
    
    cluster_stats = clustered_df.groupby('Cluster')[FEATURE_COLUMNS].mean()
    cluster_counts = clustered_df['Cluster'].value_counts().sort_index()
    
    for cluster in sorted(clustered_df['Cluster'].unique()):
        if cluster == -1:  # Skip noise points
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
    
    # Methodology
    elements.append(Spacer(1, 20))
    elements.append(Paragraph("Methodology", heading_style))
    methodology_text = """
    The analysis uses StandardScaler for feature normalization and PCA for dimensionality reduction.
    Multiple clustering algorithms were evaluated including KMeans, GMM, and Agglomerative clustering.
    Cluster quality was assessed using Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Score.
    Learner types are assigned based on relative feature values within each cluster.
    """
    elements.append(Paragraph(methodology_text, normal_style))
    
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()
