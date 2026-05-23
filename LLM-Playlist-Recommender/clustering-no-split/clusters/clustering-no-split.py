###################
# Code to cluster #
###################

import argparse
import os
import csv
import pickle
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm


def cluster_playlists(playlist_embeddings, num_clusters, playlist_titles, playlist_tracks, output_file):
    """Run K-means on playlist embeddings and write results to a CSV file.

    Args:
        playlist_embeddings: Dict mapping pid -> numpy embedding vector.
        num_clusters: Number of K-means clusters.
        playlist_titles: Dict mapping pid -> playlist title string.
        playlist_tracks: Dict mapping pid -> list of track title strings.
        output_file: Path where the output clusters.csv will be written.
    """
    embedding_matrix = np.array(list(playlist_embeddings.values()))
    pids = list(playlist_embeddings.keys())

    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init='auto')
    cluster_labels = kmeans.fit_predict(embedding_matrix)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', newline='', encoding='utf8') as f:
        writer = csv.writer(f)
        writer.writerow(["Cluster ID", "Playlist ID", "Playlist Title", "Tracks"])
        for pid, label in tqdm(zip(pids, cluster_labels), total=len(pids), desc="Clustering", unit="playlist"):
            writer.writerow([label, pid, playlist_titles.get(pid, ""), ";".join(playlist_tracks.get(pid, []))])


def run(embeddings_file, output_file, num_clusters=200):
    """Load embeddings from a pickle file and run clustering.

    Args:
        embeddings_file: Path to embeddings.pkl produced by track_embeddings_no-split.py.
        output_file: Path where clusters.csv will be written.
        num_clusters: Number of K-means clusters.
    """
    with open(embeddings_file, 'rb') as f:
        data = pickle.load(f)

    cluster_playlists(
        data["playlist_embeddings"],
        num_clusters=num_clusters,
        playlist_titles=data["playlist_titles"],
        playlist_tracks=data["playlist_tracks"],
        output_file=output_file,
    )


def main():
    parser = argparse.ArgumentParser(description="K-means clustering of playlist embeddings.")
    parser.add_argument("--embeddings_file", type=str,
                        default="/home/vellard/playlist_continuation/embeddings/embeddings.pkl")
    parser.add_argument("--output_file", type=str,
                        default="/home/vellard/playlist_continuation/clustering-no-split/clusters/200/clusters.csv")
    parser.add_argument("--num_clusters", type=int, default=200)
    args = parser.parse_args()
    run(args.embeddings_file, args.output_file, args.num_clusters)


if __name__ == "__main__":
    main()
