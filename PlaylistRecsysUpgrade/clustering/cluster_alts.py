import argparse
import os
import csv
import pickle
import numpy as np
from skfuzzy import cmeans, cmeans_predict
from tqdm import tqdm

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
