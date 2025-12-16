# LMS Learning Analytics Platform - Project Documentation

## 1️⃣ Project Summary
The **LMS Learning Analytics Platform** is a Streamlit-based dashboard designed to analyze student behavior data from Learning Management Systems (LMS). 

**Goal:** To identify distinct learner personas and behavior patterns using unsupervised machine learning techniques.

**Target Audience:** Educators, Administrators, and Learning Analysts.

**Problem Solved:** Raw LMS logs are difficult to interpret. This tool transforms raw interaction data (clicks, assessment scores, active days) into actionable insights by grouping students into meaningful clusters (e.g., "Top Performers", "At-Risk Learners").

**How it Works:**
The system takes a cleaned dataset of student behaviors, applies dimensionality reduction and clustering algorithms (like KMeans and GMM), and visualizes the results to help users understand student engagement and performance.

---

## 2️⃣ Features Overview
The application is modular, with each feature encapsulated in a specific view:

- **Home**: Landing page with project overview and quick start guide. (`views/home.py`)
- **Dataset & Preprocessing**: Documentation of the OULAD dataset, raw tables, and the cleaning pipeline. (`views/dataset_overview.py`)
- **Data Upload**: Interface to upload custom CSV datasets or load the default processed dataset. (`views/upload.py`)
- **Exploratory Data Analysis (EDA)**: Statistical summaries, correlation heatmaps, and distribution plots of behavioral features. (`views/eda.py`)
- **Cluster Count Selection**: Tools to determine the optimal number of clusters ($K$) using Elbow Method, Silhouette Analysis, and Gap Statistic. (`views/cluster_count.py`)
- **Model Training**: Interface to train multiple clustering algorithms (KMeans, GMM, Spectral, DBSCAN, etc.) and compare them using validity metrics. (`views/model_training.py`)
- **Cluster Interpretation**: Visualizes clusters using PCA (2D/3D), parallel coordinates, and generates human-readable learner profiles. (`views/cluster_interpretation.py`)
- **Time-Series Analysis**: Analyzes engagement trends over time and registration timing effects. (`views/time_series.py`)
- **New Student Prediction**: Predicts the learner persona for a new student based on their behavioral metrics. (`views/prediction.py`)
- **Dashboard and Exports**: High-level summary dashboard and options to export results. (`views/dashboard.py`)
- **VLE Data (Phase 2)**: Placeholder for future analysis of specific Virtual Learning Environment resources. (`views/vle_engagement.py`)
- **About**: Project background and team information. (`views/about.py`)

---

## 3️⃣ Folder and Code Structure

### Root Directory
- **`app.py`**: Main entry point. Handles navigation, session state initialization, and global configuration.
- **`requirements.txt`**: List of Python dependencies.
- **`packages.txt`**: System-level dependencies (e.g., for Streamlit Cloud).

### `/views/`
Contains the logic for each page in the application.
- **`home.py`**: Renders the welcome page.
- **`upload.py`**: Handles file uploads and dataframe storage in session state.
- **`dataset_overview.py`**: Displays documentation about the data source.
- **`eda.py`**: Generates Plotly charts for data exploration.
- **`cluster_count.py`**: Runs algorithms to suggest the best $K$.
- **`model_training.py`**: Manages model training, storage, and metric calculation.
- **`cluster_interpretation.py`**: Handles PCA projection and cluster profiling.
- **`prediction.py`**: Simple interface for single-point inference.
- **`time_series.py`**: Visualizes temporal patterns.
- **`dashboard.py`**: Summary metrics and export functionality.

### `/utils/`
Helper functions and shared logic.
- **`helpers.py`**: `init_session_state()` (manages global variables), `assign_learner_types()` (labels clusters), `get_cluster_description()`.
- **`constants.py`**: Defines `REQUIRED_COLUMNS`, `FEATURE_COLUMNS`, and color palettes.
- **`dataset_utils.py`**: Functions to load data and get raw table info.
- **`clustering.py`**: Core math for `compute_gap_statistic`, `compute_optimal_eps`, and centroid calculations.

### `/config/`
- **`styles.py`**: Contains custom CSS for styling the Streamlit app.

### `/data/`
- **`processed/`**: Stores the cleaned `lms_clean.csv`.
- **`raw/`**: (Optional) Storage for raw OULAD tables.

---

## 4️⃣ Dataset Explanation
The project is built around the **Open University Learning Analytics Dataset (OULAD)**.

**Required Columns:**
The application expects a CSV with the following columns (`utils/constants.py`):
- **`id_student`**: Unique identifier.
- **`final_result`**: Target label (Pass, Fail, Distinction, Withdrawn).
- **`total_clicks`**: Total VLE interactions.
- **`active_days`**: Number of days the student accessed the system.
- **`avg_daily_clicks`**: Intensity of engagement.
- **`quizzes_attempted`**: Number of assessments submitted.
- **`avg_quiz_score`**: Average score across assessments.
- **`date_registration`**: Days relative to course start (optional, for time-series).

**Data Flow:**
Data is loaded via `views/upload.py` or `utils/dataset_utils.py`. It is stored in `st.session_state.df` and accessible globally across all pages.

---

## 5️⃣ Machine Learning Workflow
The ML pipeline is implemented using `scikit-learn`.

1.  **Preprocessing**:
    - Feature selection: `['total_clicks', 'active_days', 'avg_daily_clicks', 'quizzes_attempted', 'avg_quiz_score']`.
    - Scaling: `StandardScaler` is applied immediately before clustering in `views/cluster_count.py` and `views/model_training.py`.

2.  **Cluster Count Selection**:
    - **Elbow Method**: Plots Inertia vs $K$.
    - **Silhouette Analysis**: Plots Silhouette Score vs $K$.
    - **Gap Statistic**: Compares within-cluster dispersion to a null reference.

3.  **Algorithms**:
    - **KMeans**: Standard centroid-based clustering.
    - **Gaussian Mixture (GMM)**: Probabilistic clustering.
    - **Agglomerative**: Hierarchical clustering (Disabled for datasets > 5,000 rows due to memory constraints).
    - **Spectral Clustering**: Graph-based clustering.
    - **DBSCAN**: Density-based clustering (auto-detects noise).
    - **HDBSCAN**: Hierarchical density-based clustering (if installed).

4.  **Evaluation Metrics**:
    - Silhouette Score
    - Davies-Bouldin Index
    - Calinski-Harabasz Score

---

## 6️⃣ Streamlit Application Flow
The app uses a **Sidebar Navigation** model managed in `app.py`.

1.  **Initialization**: `init_session_state()` sets up empty variables for models, data, and metrics.
2.  **Navigation**: `st.sidebar.radio` selects the active view.
3.  **Rendering**: `pages[selection].render()` calls the `render()` function of the selected module.
4.  **State Management**: Data persists in `st.session_state` as the user moves between pages.

---

## 7️⃣ Data Visualization & Output
The project uses **Plotly** for interactive visualizations.

- **EDA**: Histograms, Correlation Heatmaps, Box Plots (`views/eda.py`).
- **Clustering**:
    - Elbow/Silhouette Line Charts (`views/cluster_count.py`).
    - 2D & 3D PCA Scatter Plots (`views/cluster_interpretation.py`).
    - Parallel Coordinates Plot (`views/cluster_interpretation.py`).
    - Radar Charts for Cluster Profiles (`views/time_series.py`).
    - K-Distance Plot for DBSCAN (`views/model_training.py`).

---

## 8️⃣ How the System Works End-to-End

**Step 1: Data Loading**
User goes to **Data Upload**, uploads a CSV, or clicks "Load Default Dataset". The data is validated and stored in session state.

**Step 2: Exploration**
User visits **Exploratory Data Analysis** to view distributions and correlations to understand the data structure.

**Step 3: Hyperparameter Tuning**
User navigates to **Cluster Count Selection**, runs the analysis, and selects an optimal $K$ (e.g., $K=4$) based on the Elbow/Silhouette plots.

**Step 4: Model Training**
User goes to **Model Training**, selects algorithms (e.g., KMeans, GMM), and clicks "Train". The app scales the data, fits the models, and calculates performance metrics.

**Step 5: Interpretation**
User selects an "Active Model". The app assigns learner types (e.g., "Top Performer") based on centroids. User views **Cluster Interpretation** to see PCA plots and learner profiles.

**Step 6: Prediction**
User goes to **New Student Prediction**, inputs values for a hypothetical student, and gets a predicted cluster assignment.

---

## 9️⃣ Limitations Based Only on Current Code
- **Memory Constraint**: Agglomerative Clustering is explicitly disabled in `views/model_training.py` for datasets larger than 5,000 rows to prevent OOM errors.
- **Matplotlib Backend**: `app.py` forces `matplotlib.use('Agg')`, meaning interactive Matplotlib charts are not supported (only static).
- **HDBSCAN Dependency**: The import is wrapped in a `try-except` block; if the package is missing, the feature is hidden.
- **Hardcoded Columns**: The system relies on specific column names defined in `utils/constants.py`.

---

## 🔟 Developer Notes
- **Session State**: The app relies heavily on `st.session_state` to pass data between pages. See `utils/helpers.py` for the full schema.
- **Error Handling**: The `main()` function in `app.py` is wrapped in a global `try-except` block to catch and display runtime errors gracefully.
- **Styling**: Custom CSS is injected via `config/styles.py` to enforce a Teal color theme (`#008080`).
- **PCA Caching**: PCA coordinates are computed on-demand in `views/cluster_interpretation.py` if missing, to support workflows where the user skips the explicit analysis step.
