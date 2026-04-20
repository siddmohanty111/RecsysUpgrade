import argparse
import os
import csv
import pickle
import numpy as np
from skfuzzy import cmeans, cmeans_predict
from tqdm import tqdm

try:
    from cuml.cluster import SpectralClustering, DBSCAN
    from cuml.mixture import GaussianMixture
    _CUML = True
except ImportError:
    from sklearn.cluster import SpectralClustering, DBSCAN
    from sklearn.mixture import GaussianMixture
    _CUML = False

def fkmeans(playlist_embeddings, num_clusters,
                      playlist_titles, playlist_tracks, output_file, c_partitioned : bool = False):
    """Run fuzzy c-means on playlist embeddings and write results to a CSV file.

    Args:
        playlist_embeddings: Dict mapping pid -> numpy embedding vector.
        num_clusters: Number of fuzzy c-means clusters.
        playlist_titles: Dict mapping pid -> playlist title string.
        playlist_tracks: Dict mapping pid -> list of track title strings.
        output_file: Path where the output clusters.csv will be written.
    """
    embedding_matrix = np.array(list(playlist_embeddings.values()))
    pids = list(playlist_embeddings.keys())

    if c_partitioned:
        # Use K-means to get initial guess for cluster centers
        # TODO
        pass
    fkmeans = cmeans(embedding_matrix.T, num_clusters, 2, error=0.005, maxiter=1000)
    # cluster_labels_matrix shape: (num_clusters, num_playlists)
    cluster_labels_matrix = fkmeans[1]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', newline='', encoding='utf8') as f:
        writer = csv.writer(f)
        writer.writerow(["Cluster Labels", "Playlist ID", "Playlist Title", "Tracks"])
        for pid, memberships in tqdm(zip(pids, cluster_labels_matrix.T), total=len(pids), desc="Clustering", unit="playlist"):
            writer.writerow([memberships.tolist(), pid, playlist_titles.get(pid, ""), ";".join(playlist_tracks.get(pid, []))])
    return np.argmax(cluster_labels_matrix, axis=0)  # hard labels for visualization


def spectral(playlist_embeddings, num_clusters,
                      playlist_titles, playlist_tracks, output_file):
    embedding_matrix = np.array(list(playlist_embeddings.values()), dtype=np.float32)
    pids = list(playlist_embeddings.keys())

    kwargs = {'n_clusters': num_clusters, 'random_state': 0}
    if not _CUML:
        kwargs.update({'affinity': 'nearest_neighbors', 'n_jobs': -1})
    sc = SpectralClustering(**kwargs)
    cluster_labels = np.asarray(sc.fit_predict(embedding_matrix))

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', newline='', encoding='utf8') as f:
        writer = csv.writer(f)
        writer.writerow(["Cluster ID", "Playlist ID", "Playlist Title", "Tracks"])
        for pid, label in tqdm(zip(pids, cluster_labels), total=len(pids), desc="Clustering", unit="playlist"):
            writer.writerow([label, pid, playlist_titles.get(pid, ""), ";".join(playlist_tracks.get(pid, []))])
    return cluster_labels

def dbscan(playlist_embeddings,
                      playlist_titles, playlist_tracks, output_file,
                      eps: float = 0.5, min_samples: int = 5):
    embedding_matrix = np.array(list(playlist_embeddings.values()), dtype=np.float32)
    pids = list(playlist_embeddings.keys())

    kwargs = {'eps': eps, 'min_samples': min_samples}
    if not _CUML:
        # ball_tree avoids materializing the O(n²) distance matrix
        kwargs.update({'n_jobs': -1, 'algorithm': 'ball_tree'})
    db = DBSCAN(**kwargs)
    cluster_labels = np.asarray(db.fit_predict(embedding_matrix))

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', newline='', encoding='utf8') as f:
        writer = csv.writer(f)
        writer.writerow(["Cluster ID", "Playlist ID", "Playlist Title", "Tracks"])
        for pid, label in tqdm(zip(pids, cluster_labels), total=len(pids), desc="Clustering", unit="playlist"):
            writer.writerow([label, pid, playlist_titles.get(pid, ""), ";".join(playlist_tracks.get(pid, []))])
    return cluster_labels
    
def gaussianmix(playlist_embeddings, num_clusters,
                      playlist_titles, playlist_tracks, output_file):
    embedding_matrix = np.array(list(playlist_embeddings.values()), dtype=np.float32)
    pids = list(playlist_embeddings.keys())

    gm = GaussianMixture(n_components=num_clusters, random_state=0)
    gm.fit(embedding_matrix)
    # posterior probabilities shape: (num_playlists, num_clusters)
    cluster_labels_matrix = np.asarray(gm.predict_proba(embedding_matrix))

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', newline='', encoding='utf8') as f:
        writer = csv.writer(f)
        writer.writerow(["Cluster Labels", "Playlist ID", "Playlist Title", "Tracks"])
        for pid, memberships in tqdm(zip(pids, cluster_labels_matrix), total=len(pids), desc="Clustering", unit="playlist"):
            writer.writerow([memberships.tolist(), pid, playlist_titles.get(pid, ""), ";".join(playlist_tracks.get(pid, []))])
    return np.argmax(cluster_labels_matrix, axis=1)  # hard labels for visualization