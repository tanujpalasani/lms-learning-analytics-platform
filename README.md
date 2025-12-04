# LMS Student Behavior Analytics Dashboard

A comprehensive Streamlit application for analyzing student behavior in Learning Management Systems (LMS) using the Open University Learning Analytics Dataset (OULAD). This tool helps educators and administrators identify learner personas and predict student outcomes based on engagement and performance metrics.

## 🚀 Features

- **Dataset Overview**: Explore the OULAD dataset structure and preprocessing pipeline.
- **Data Upload**: Upload your own student data or use the default processed dataset.
- **Exploratory Data Analysis (EDA)**: Visualize distributions, correlations, and key metrics.
- **Cluster Analysis**: Determine optimal learner groups using Elbow Method, Silhouette Score, and Gap Statistic.
- **Manual Cluster Selection**: Option to manually set the number of clusters (K) without running full analysis.
- **Model Training**: Train unsupervised learning models (KMeans, GMM, Agglomerative Clustering, DBSCAN, Spectral Clustering).
- **Cluster Interpretation**: Analyze and name the resulting learner personas (e.g., "High Achievers", "At-Risk").
- **Prediction**: Classify new students into identified personas based on their behavior.
- **Dashboard**: High-level executive summary of student population health.

## 📂 Project Structure

```
LMS/
├── app.py                  # Main application entry point
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── config/                 # Configuration files (styles, settings)
├── data/                   # Data storage
│   ├── raw/                # Raw OULAD CSV files (optional)
│   ├── processed/          # Cleaned dataset (lms_clean.csv)
│   └── notebooks/          # Preprocessing notebooks
├── models/                 # Model storage
│   └── pretrained/         # Pretrained models (K=4) and scaler
├── scripts/                # Utility scripts (e.g., model generation)
├── utils/                  # Helper functions (data loading, clustering, plotting)
└── views/                  # UI pages for the Streamlit app
```

## 🛠️ Installation

1.  **Clone the repository** (if applicable) or download the source code.
2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```
3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## ▶️ Usage

1.  **Run the application**:
    ```bash
    streamlit run app.py
    ```
    *Optionally specify a port:*
    ```bash
    streamlit run app.py --server.port 8501
    ```

2.  **Access the dashboard**:
    Open your browser and navigate to `http://localhost:8501`.

## 📊 Workflow

1.  **Upload Data**: Start by loading the default dataset or uploading your own `lms_clean.csv`.
2.  **Select Clusters**: Go to "Cluster Count Selection" to determine the optimal K (or use Manual Selection).
3.  **Train Models**: Train clustering algorithms in the "Model Training" section.
4.  **Analyze**: Use "Cluster Interpretation" to understand the groups.
5.  **Predict**: Use "New Student Prediction" to classify individual students.

## ☁️ Deployment

This application is ready for deployment on platforms like **Streamlit Community Cloud**, **Heroku**, or **AWS**.

-   **Streamlit Cloud**: Connect your GitHub repository and point the main file to `app.py`.
-   **Requirements**: Ensure `requirements.txt` is present in the root directory.
-   **Data**: If using the default dataset, ensure `data/processed/lms_clean.csv` is committed to the repository.

## 📝 License

This project uses the Open University Learning Analytics Dataset (OULAD) available under CC-BY 4.0.
