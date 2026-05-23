##############################################################
# Code to precompute the tracks embeddings before clustering #
##############################################################

import argparse
import os
import pickle
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


def load_playlist_titles(file_path):
    """Load playlist titles from a CSV file.

    Args:
        file_path: Path to playlists.csv.

    Returns:
        Dict mapping pid -> playlist title.
    """
    titles = {}
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            pid, title = line.strip().split(',')[0], line.strip().split(',')[1]
            titles[pid] = title
    return titles


def load_playlist_track_titles(items_file_path, tracks_file_path):
    """Load track titles per playlist from items.csv and tracks.csv.

    Args:
        items_file_path: Path to items.csv.
        tracks_file_path: Path to tracks.csv.

    Returns:
        Dict mapping pid -> list of track title strings.
    """
    track_titles = {}
    with open(tracks_file_path, 'r', encoding='utf8') as f:
        for line in f:
            track_uri, track_name, *_ = line.strip().split(',')
            track_titles[track_uri] = track_name

    playlist_tracks = {}
    with open(items_file_path, 'r', encoding='utf8') as f:
        for line in f:
            pid, _, track_uri = line.strip().split(',')
            if pid not in playlist_tracks:
                playlist_tracks[pid] = []
            if track_uri in track_titles:
                playlist_tracks[pid].append(track_titles[track_uri])
    return playlist_tracks


def compute_track_embeddings(model, playlist_tracks):
    """Compute SentenceBERT embeddings for every unique track title.

    Args:
        model: A SentenceTransformer model instance.
        playlist_tracks: Dict mapping pid -> list of track title strings.

    Returns:
        Dict mapping track title -> numpy embedding vector.
    """
    unique_tracks = list(set(title for tracks in playlist_tracks.values() for title in tracks))
    track_embeddings_array = model.encode(unique_tracks, show_progress_bar=True, convert_to_numpy=True)
    track_embeddings = {track_title: emb for track_title, emb in zip(unique_tracks, track_embeddings_array)}
    return track_embeddings


def compute_playlist_embeddings(playlist_tracks, track_embeddings):
    """Represent each playlist as the mean of its track embeddings.

    Args:
        playlist_tracks: Dict mapping pid -> list of track title strings.
        track_embeddings: Dict mapping track title -> numpy embedding vector.

    Returns:
        Dict mapping pid -> mean embedding numpy vector.
    """
    playlist_embeddings = {}
    for pid, tracks in tqdm(playlist_tracks.items(), desc="Processing playlists", unit="playlist"):
        vecs = [track_embeddings[t] for t in tracks if t in track_embeddings]
        if vecs:
            playlist_embeddings[pid] = np.mean(vecs, axis=0)
    return playlist_embeddings


def run(playlists_csv, items_csv, tracks_csv, output_file,
        model_name='sentence-transformers/all-mpnet-base-v2'):
    """End-to-end embedding computation and serialisation.

    Args:
        playlists_csv: Path to playlists.csv.
        items_csv: Path to items.csv.
        tracks_csv: Path to tracks.csv.
        output_file: Path where embeddings.pkl will be written.
        model_name: HuggingFace model identifier for SentenceTransformer.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    model = SentenceTransformer(model_name)

    playlist_titles = load_playlist_titles(playlists_csv)
    playlist_tracks = load_playlist_track_titles(items_csv, tracks_csv)

    track_embeddings = compute_track_embeddings(model, playlist_tracks)
    playlist_embeddings = compute_playlist_embeddings(playlist_tracks, track_embeddings)

    embeddings_data = {
        "playlist_embeddings": playlist_embeddings,
        "playlist_titles": playlist_titles,
        "playlist_tracks": playlist_tracks,
    }

    with open(output_file, 'wb') as f:
        pickle.dump(embeddings_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Embeddings (no split) saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Precompute track-based playlist embeddings for clustering.")
    parser.add_argument("--playlists_csv", type=str, default="/data/playlist_continuation_data/csvs/playlists.csv")
    parser.add_argument("--items_csv", type=str, default="/data/playlist_continuation_data/csvs/items.csv")
    parser.add_argument("--tracks_csv", type=str, default="/data/playlist_continuation_data/csvs/tracks.csv")
    parser.add_argument("--output_file", type=str, default="/home/vellard/playlist_continuation/embeddings/embeddings.pkl")
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-mpnet-base-v2")
    args = parser.parse_args()
    run(args.playlists_csv, args.items_csv, args.tracks_csv, args.output_file, args.model_name)


if __name__ == "__main__":
    main()
