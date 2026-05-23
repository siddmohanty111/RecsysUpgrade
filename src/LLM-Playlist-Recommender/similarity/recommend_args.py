#######################################
# Code to generate the recommendation #
#######################################

import os
import argparse
import torch
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from collections import Counter
import csv
from transformers import AutoTokenizer, AutoModel
import math


##########################
# 1) Load Fine-tuned Model
##########################

def load_fine_tuned_model(model_dir, base_model_name='sentence-transformers/all-MiniLM-L6-v2'):
    # Load the tokenizer from the same base model
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Load the fine-tuned transformer model
    model = AutoModel.from_pretrained(model_dir)
    model.eval()

    # Minimal addition for GPU usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return tokenizer, model


##############################
# 2) Compute the playlist embedding
##############################

def get_playlist_embedding(playlist_name, tokenizer, model):
    '''if not isinstance(playlist_name, str):
        playlist_name = str(playlist_name)'''

    with torch.no_grad():
        inputs = tokenizer(playlist_name, return_tensors='pt', truncation=True, padding=True).to(model.device)
        outputs = model(**inputs)
        last_hidden = outputs.last_hidden_state  # shape: (batch_size, seq_len, hidden_size)
        embedding = last_hidden.mean(dim=1).squeeze().cpu().numpy()

    return embedding


#################################
# 3) Load Precomputed Embeddings
#################################

def load_playlist_embeddings(embeddings_file):
    with open(embeddings_file, 'rb') as f:
        playlist_embeddings = pickle.load(f)
    return playlist_embeddings


#########################################
# 4) Associate track metadata with tracks
#########################################

def load_playlist_tracks_with_artists(items_csv, tracks_csv):
    track_metadata = {}
    with open(tracks_csv, 'r', encoding='utf8') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Loading track metadata", unit=" tracks"):
            track_metadata[row["track_uri"]] = {
                "track_uri": row["track_uri"],
                "track_name": row["track_name"],
                "artist_name": row["artist_name"],
            }

    playlist_tracks = {}
    with open(items_csv, 'r', encoding='utf8') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Loading playlist-tracks pairs", unit=" pairs"):
            pid_str = row["pid"].strip()
            track_uri = row["track_uri"]

            if pid_str not in playlist_tracks:
                playlist_tracks[pid_str] = []
            if track_uri in track_metadata:
                playlist_tracks[pid_str].append(track_metadata[track_uri])

    return playlist_tracks


############################
# 5) Find Similar Playlists
############################

def find_similar_playlists(playlist_name, playlist_embeddings, tokenizer, model, top_k):
    playlist_embedding = get_playlist_embedding(playlist_name, tokenizer, model)

    similarities = []
    for pid, metadata in tqdm(playlist_embeddings.items(), desc="Scoring Playlists", unit=" playlists"):
        similarity = cosine_similarity([playlist_embedding], [metadata["embedding"]])[0][0]
        similarities.append((pid, similarity))

    sorted_playlists = sorted(similarities, key=lambda x: x[1], reverse=True)
    return sorted_playlists[:top_k]


########################################
# 6) Retrieve the Top Songs from those playlists
########################################

def get_top_songs_with_artists(similar_playlists, playlist_tracks, top_k):
    song_counter = Counter()
    for pid, _ in tqdm(similar_playlists, desc="Counting songs", unit="playlist"):
        pid_str = str(pid)
        if pid_str in playlist_tracks:
            for track_metadata in playlist_tracks[pid_str]:
                song_counter[(track_metadata["track_uri"], track_metadata["track_name"], track_metadata["artist_name"])] += 1
    return song_counter.most_common(top_k)


########################################
# 7) Evaluation metrics
########################################

def compute_metrics(recommended_songs, relevant_songs, top_n=10):
    # Sets
    G_T = set(relevant_songs)
    G_A = set(artist for _, artist in relevant_songs)

    R = len(G_T)
    top_r = recommended_songs[:R]
    S_T = set(top_r)
    S_A = set(artist for _, artist in top_r)

    # R-Precision with artist bonus
    exact_matches = S_T & G_T
    matched_artists = S_A & G_A
    track_score = len(exact_matches)
    artist_score = len(matched_artists) * 0.25
    r_precision = (track_score + artist_score) / R if R > 0 else 0.0

    # HIT@N, Precision, Recall, MRR
    hits = sum(1 for song in recommended_songs[:top_n] if song in G_T)
    hit_score = hits / min(top_n, len(G_T)) if len(G_T) > 0 else 0.0
    precision = hits / len(recommended_songs[:top_n]) if len(recommended_songs[:top_n]) > 0 else 0.0
    recall = hits / len(G_T) if len(G_T) > 0 else 0.0

    mrr = 0.0
    for i, song in enumerate(recommended_songs[:top_n]):
        if song in G_T:
            mrr = 1 / (i + 1)
            break

    # NDCG
    relevance_list = [1 if song in G_T else 0 for song in recommended_songs[:top_n]]

    def dcg(rel):
        return sum(rel_i / math.log2(idx + 2) for idx, rel_i in enumerate(rel))

    dcg_val = dcg(relevance_list)
    ideal_rel = sorted(relevance_list, reverse=True)
    idcg_val = dcg(ideal_rel)
    ndcg = dcg_val / idcg_val if idcg_val > 0 else 0.0

    return {
        "HIT@N": hit_score,
        "Precision@N": precision,
        "Recall@N": recall,
        "MRR@N": mrr,
        "R-Precision": r_precision,
        "NDCG": ndcg
    }


########################################
# 8) Main Function
########################################
def initialize(model_dir, playlist_embeddings_file, items_csv, tracks_csv, playlists_csv):
    tokenizer, model = load_fine_tuned_model(model_dir)
    print("Loaded tokenizer & fine-tuned model successfully.")

    playlist_embeddings = load_playlist_embeddings(playlist_embeddings_file)
    playlist_tracks = load_playlist_tracks_with_artists(items_csv, tracks_csv)
    playlist_titles = {}
    with open(playlists_csv, 'r', encoding='utf8') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Loading playlist titles", unit=" playlists"):
            pid_str = row["pid"].strip()
            playlist_titles[pid_str] = row["name"]

    return tokenizer, model, playlist_embeddings, playlist_tracks, playlist_titles


def recommend(playlist_name, top_k, tokenizer, model, playlist_embeddings, playlist_tracks, playlist_titles,
              printing=False):
    if playlist_name is None or len(playlist_name) == 0:
        raise "Empty playlist title"

    print(f"Generating recommendations for playlist: '{playlist_name}'...")
    top_playlists = find_similar_playlists(playlist_name, playlist_embeddings, tokenizer, model, top_k=50)

    print("\nTop Similar Playlists:")
    for i, (similar_pid, similarity) in enumerate(top_playlists, start=1):
        title = playlist_titles.get(str(similar_pid), "Unknown Playlist Title")
        print(f"{i}. Playlist Title: {title}, Similarity: {similarity:.4f}")

    top_songs = get_top_songs_with_artists(top_playlists, playlist_tracks, top_k=top_k)

    if printing:
        print("\nTop Recommended Songs:")
        for i, ((uri, song, artist), count) in enumerate(top_songs, start=1):
            print(f"{i}. Song: {song}, Artist: {artist}, Occurrences: {count}")

    return top_songs


def main():
    parser = argparse.ArgumentParser(description="Playlist Recommender Script")
    parser.add_argument("--model_dir", type=str,
                        default="/home/vellard/playlist_continuation/fine_tuned_model_no_scheduler_2",
                        help="Directory of the fine-tuned model")
    parser.add_argument("--playlist_embeddings_file", type=str,
                        default='/home/vellard/playlist_continuation/playlists_embeddings/final_embeddings/playlists_embeddings_scheduler.pkl',
                        help="Path to the playlist embeddings file")
    parser.add_argument("--csv_folder", type=str, default='/data/csvs/',
                        help="Path to the items, tracks and playlists CSV file")
    parser.add_argument("--playlist_title", '-t', type=str, required=True,
                        help="The seed title of the desired generated playlist")
    parser.add_argument("--top_k", '-k', type=int, default=10,
                        help="How many songs in output of the voting mechanism")

    args = parser.parse_args()

    top_k = args.top_k
    model_dir = args.model_dir
    playlist_name = args.playlist_title
    playlist_embeddings_file = args.playlist_embeddings_file
    items_csv = os.path.join(args.csv_folder, "items.csv")
    tracks_csv = os.path.join(args.csv_folder, "tracks.csv")
    playlists_csv = os.path.join(args.csv_folder, "playlists.csv")

    if playlist_name is None or len(playlist_name) == 0:
        raise "Empty playlist title"

    tokenizer, model, playlist_embeddings, playlist_tracks, playlist_titles = initialize(model_dir,
                                                                                         playlist_embeddings_file,
                                                                                         items_csv, tracks_csv,
                                                                                         playlists_csv)
    top_songs = recommend(playlist_name, top_k, tokenizer, model, playlist_embeddings, playlist_tracks, playlist_titles,
                          printing=True)

    # Write to CSV
    csv_filename = f"recommended_for_{playlist_name}.csv"
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        # Write header
        writer.writerow(["Rank", "Song", "Artist", "Occurrences"])
        # Write rows
        for i, ((song, artist), count) in enumerate(top_songs, start=1):
            writer.writerow([i, song, artist, count])

    print(f"\nTop recommended songs have been saved to '{csv_filename}'.")


if __name__ == "__main__":
    main()
