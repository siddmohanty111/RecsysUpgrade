"""

A module designed to perform locality sensitive hashing on the fkm clusters. The LSH step is used to identify subclusters of playlists with similar titles within each cluster, and to drop clusters where the largest subcluster contains <= 2% of the playlists in the cluster. Essentially if the a cluster is too broad (or wide) we drop it. 

Algorithm from https://mlwiki.org/index.php/Euclidean_LSH

"""

import numpy as np
from collections import Counter
import pandas as pd
from sklearn.metrics import pairwise_distances

def bucket_width_sampler(data, num_samples=1000, seed=123):
    """
    
    To determine a good bucket width for LSH, I sample playlists from the entire dataset and compute pairwise distances between their embeddings, then return the median distance as bucket width (to control for outliers).

    A small default sample of 1000 is used to avoid time complexity issues, as computing pairwise distances on the entire dataset of 1M playlists is not a scalable design choice. 

    :param data: np.ndarray of shape (num_playlists, embedding_dim)
    :param num_samples: int, number of playlists to sample for distance calculation
    :param seed: int, random seed for reproducibility
    :return: The median pairwise distance among the sampled embeddings, to be used as the bucket width.

    """
    rng = np.random.default_rng(seed)
    sample_indices = rng.choice(len(data), size=num_samples, replace=False)
    sample_embeddings = data[sample_indices]

    distances = pairwise_distances(sample_embeddings, metric='euclidean')
    
    # Return median distance as bucket width
    return np.median(distances)

def euclidean_LSH(cluster_playlist_embeddings : np.ndarray, num_projections=7, bucket_width=1.0, seed=123):
    """
    
    Performs LSH with Euclidean distance on the playlist embeddings for the given cluster.

    :param cluster_playlist_embeddings: np.ndarray of shape (num_playlists_in_cluster, embedding_dim)
    :param num_projections: int, number of random projections to use for hashing
    :param bucket_width: float, width of the buckets for hashing
    :param seed: int, random seed for reproducibility
    :return: List of hash values for each playlist in the cluster
    
    """

    rng = np.random.default_rng(seed)
    n_samples, n_features = cluster_playlist_embeddings.shape

    # random projection vectors a
    a = rng.standard_normal((n_features, num_projections))

    # random offsets b
    b = rng.uniform(0, bucket_width, size=num_projections)

    # Calculate projections as a * input data + b
    projections = np.dot(cluster_playlist_embeddings, a) + b

    hash_values = np.floor(projections / bucket_width).astype(int)

    # Python doesnt allow lists as dict keys (bc mutability) so we convert to str
    return [','.join(row.astype(str)) for row in hash_values]

def prune_clusters(embeddings, cluster_assignments_df : pd.DataFrame, threshold_percent=0.02):
    """
    
    :param embeddings: np.ndarray of shape (num_playlists, embedding_dim)
    :param cluster_assignments_df: DataFrame containing cluster assignments
    :param threshold_percent: float, clusters with largest subcluster containing <= threshold_percent of playlists will be dropped
    :return: List of cluster IDs and data points that are kept after pruning
    """

    pids = cluster_assignments_df['pid'].values
    cluster_assignments = cluster_assignments_df['cluster'].values

    kept_clusters = []
    kept_data_points = []

    # only grab unique cluster labels
    unique_clusters = np.unique(cluster_assignments)    

    # sample to find good bucket width (to keep things simple, I am doing this only once outside the loop)

    bucket_width = bucket_width_sampler(embeddings)

    for cluster_id in unique_clusters:

        # get all the data points in the cluster corresponding to cluster_id
        row_indices = np.where(cluster_assignments == cluster_id)[0]

        # extract the subset of pids and embeddings for the cluster corresponding to cluster_id
        cluster_pids = pids[row_indices]
        cluster_embeddings = embeddings[row_indices]

        # Perform LSH on the embeddings in this cluster
        buckets = euclidean_LSH(cluster_embeddings, bucket_width=bucket_width)

        # count how many playlists fall into each bucket
        bucket_counts = Counter(buckets)
        largest_bucket_size = max(bucket_counts.values())
        highest_percent = largest_bucket_size / len(cluster_pids)

        if highest_percent > threshold_percent:
            # cluster has a bucket with more than threshold_percent of the playlists, so we keep it
            kept_clusters.append(cluster_id)
            kept_data_points.extend(cluster_pids)
            print(f"Keeping cluster {cluster_id} (Max Bucket: {highest_percent:.2%} of playlists). {len(cluster_pids)} playlists in cluster.")
        else:
            # cluster is too broad, so do not keep
            print(f"Dropping cluster {cluster_id} (Max Bucket: {highest_percent:.2%} of playlists). {len(cluster_pids)} playlists in cluster.")

