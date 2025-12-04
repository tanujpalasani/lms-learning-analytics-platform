import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# Add project root to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.constants import FEATURE_COLUMNS

def generate_models():
    print("Starting model generation...")
    
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'processed', 'lms_clean.csv')
    models_dir = os.path.join(base_dir, 'models', 'pretrained')
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Created directory: {models_dir}")
        
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        return

    # Load Data
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    X = df[FEATURE_COLUMNS].values
    
    # Scale Data
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save Scaler
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler to {scaler_path}")
    
    # Find Optimal K (Silhouette)
    # print("Determining optimal K (this may take a moment)...")
    # silhouettes = []
    # k_range = range(2, 11)
    
    # for k in k_range:
    #     kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    #     labels = kmeans.fit_predict(X_scaled)
    #     score = silhouette_score(X_scaled, labels)
    #     silhouettes.append(score)
    #     print(f"K={k}: Silhouette Score = {score:.4f}")
        
    # best_k_idx = silhouettes.index(max(silhouettes))
    # optimal_k = k_range[best_k_idx]
    # print(f"Optimal K determined: {optimal_k}")
    
    optimal_k = 4
    print(f"Using fixed K={optimal_k} as requested.")
    
    # Train and Save Models
    print(f"Training models with K={optimal_k}...")
    
    # 1. KMeans
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    joblib.dump(kmeans, os.path.join(models_dir, f'kmeans_k{optimal_k}.pkl'))
    print("Saved KMeans model")
    
    # 2. GMM
    gmm = GaussianMixture(n_components=optimal_k, random_state=42)
    gmm.fit(X_scaled)
    joblib.dump(gmm, os.path.join(models_dir, f'gmm_k{optimal_k}.pkl'))
    print("Saved GMM model")
    
    # 3. Agglomerative
    # Note: AgglomerativeClustering doesn't have a predict method in the same way, 
    # but we can save the fitted object. However, for new predictions it's limited.
    # We'll save it anyway as per request.
    agg = AgglomerativeClustering(n_clusters=optimal_k)
    agg.fit(X_scaled)
    joblib.dump(agg, os.path.join(models_dir, f'agglomerative_k{optimal_k}.pkl'))
    print("Saved Agglomerative model")
    
    print("All models generated and saved successfully!")

if __name__ == "__main__":
    generate_models()
