"""
Clustering-related utility functions including centroid computation,
optimal eps detection, and gap statistic.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


def compute_cluster_centroids(X_scaled, labels):
    """
    Compute centroids for each cluster.
    
    Args:
        X_scaled: Scaled feature matrix
        labels: Cluster labels
        
    Returns:
        dict: Mapping of cluster label to centroid coordinates
    """
    unique_labels = np.unique(labels)
    centroids = {}
    for label in unique_labels:
        if label == -1:  # Skip noise points
            continue
        mask = labels == label
        if mask.sum() > 0:
            centroids[label] = X_scaled[mask].mean(axis=0)
    return centroids


def predict_nearest_centroid(input_scaled, centroids):
    """
    Predict cluster by finding nearest centroid.
    
    Args:
        input_scaled: Scaled input features (1D array)
        centroids: Dictionary of cluster centroids
        
    Returns:
        int: Predicted cluster label
    """
    min_dist = float('inf')
    predicted_cluster = 0
    for label, centroid in centroids.items():
        dist = np.linalg.norm(input_scaled - centroid)
        if dist < min_dist:
            min_dist = dist
            predicted_cluster = label
    return predicted_cluster


def compute_optimal_eps(X_scaled, min_samples=5):
    """
    Compute optimal eps for DBSCAN using k-nearest neighbor distance analysis.
    Uses the 'elbow' method on the k-distance graph where k = min_samples.
    
    Args:
        X_scaled: Scaled feature matrix
        min_samples: Minimum samples for DBSCAN
        
    Returns:
        tuple: (optimal_eps: float, k_distances: array)
    """
    n_neighbors = min(min_samples, len(X_scaled) - 1)
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(X_scaled)
    distances, _ = nn.kneighbors(X_scaled)
    k_distances = np.sort(distances[:, -1])
    
    # Find elbow point using perpendicular distance
    n_points = len(k_distances)
    line_start = np.array([0, k_distances[0]])
    line_end = np.array([n_points - 1, k_distances[-1]])
    line_vec = line_end - line_start
    line_len = np.linalg.norm(line_vec)
    line_unit = line_vec / line_len
    
    max_distance = 0
    best_idx = n_points // 2
    
    for i in range(n_points):
        point = np.array([i, k_distances[i]])
        vec_to_point = point - line_start
        proj_length = np.dot(vec_to_point, line_unit)
        proj_point = line_start + proj_length * line_unit
        distance = np.linalg.norm(point - proj_point)
        
        if distance > max_distance:
            max_distance = distance
            best_idx = i
    
    optimal_eps = k_distances[best_idx]
    # Clamp to reasonable range
    optimal_eps = max(0.1, min(optimal_eps, 3.0))
    
    return optimal_eps, k_distances


def compute_gap_statistic(X, k_range, n_refs=10):
    """
    Compute gap statistic for determining optimal cluster count.
    
    Args:
        X: Feature matrix
        k_range: Range of K values to test
        n_refs: Number of reference datasets
        
    Returns:
        tuple: (gaps: list, gap_errors: list)
    """
    gaps = []
    gap_errors = []
    
    for k in k_range:
        # Fit KMeans on actual data
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        log_wk = np.log(kmeans.inertia_)
        
        # Generate reference datasets and compute inertias
        ref_inertias = []
        for _ in range(n_refs):
            random_ref = np.random.uniform(X.min(axis=0), X.max(axis=0), X.shape)
            kmeans_ref = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans_ref.fit(random_ref)
            ref_inertias.append(np.log(kmeans_ref.inertia_))
        
        # Compute gap and error
        gap = np.mean(ref_inertias) - log_wk
        gap_error = np.std(ref_inertias) * np.sqrt(1 + 1/n_refs)
        
        gaps.append(gap)
        gap_errors.append(gap_error)
    
    return gaps, gap_errors
