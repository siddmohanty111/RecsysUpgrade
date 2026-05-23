import argparse
import csv
import math
import os

from tqdm import tqdm

from recommend_args import initialize, recommend


def compute_metrics(recommended_songs, relevant_songs, top_n=10):
    # Sets
    G_T = set(relevant_songs)
    G_A = set(artist for uri, track, artist in relevant_songs)

    R = len(G_T)
    top_r = recommended_songs[:R]
    S_T = set(top_r)
    S_A = set(artist for uri, track, artist in top_r)

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


########
# Main #
########
def main():
    parser = argparse.ArgumentParser(description="Playlist Recommender Script")
    parser.add_argument("--model_dir", type=str, default="../model/fine_tuned_model_no_scheduler_2",
                        help="Directory of the fine-tuned model")
    parser.add_argument("--playlist_embeddings_file", type=str,
                        default='../model/playlists_embeddings_scheduler.pkl',
                        help="Path to the playlist embeddings file")
    parser.add_argument("--csv_folder", type=str, default='../data/',
                        help="Path to the items, tracks and playlists CSV file")
    parser.add_argument("--test_file", type=str,
                        # default='/home/vellard/playlist_continuation/clusters/clusters_test.csv', 
                        help="Path to the test file in CSV")
    parser.add_argument("--output_folder", '-o', default='output', type=str, help="Name of the output file")
    parser.add_argument("--top_k", '-k', type=int, help="How many songs in output of the voting mechanism")

    args = parser.parse_args()

    top_k = args.top_k
    model_dir = os.path.abspath(args.model_dir)
    playlist_embeddings_file = os.path.abspath(args.playlist_embeddings_file)
    items_csv = os.path.join(args.csv_folder, "items.csv")
    tracks_csv = os.path.join(args.csv_folder, "tracks.csv")
    playlists_csv = os.path.join(args.csv_folder, "playlists.csv")
    test_file = args.test_file
    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)

    # Choose the model directory
    clusters_test_csv = test_file

    tokenizer, model, playlist_embeddings, playlist_tracks, playlist_titles = initialize(model_dir,
                                                                                         playlist_embeddings_file,
                                                                                         items_csv, tracks_csv,
                                                                                         playlists_csv)

    # Batch evaluation
    all_results = []

    test_set = []
    if clusters_test_csv:
        with open(clusters_test_csv, 'r', encoding='utf8') as f:
            reader = csv.DictReader(f)
            for row in tqdm(reader, desc="Evaluating test playlists", unit="playlist"):
                cluster_id = row["Cluster ID"]
                test_pid = row["Playlist ID"].strip()
                playlist_name = row["Playlist Title"].strip()
                test_set.append((cluster_id, test_pid, playlist_name))
    else:
        test_set = [
            (False, "673925", "K-pop"),
            (False, "677580", "workout music"),
            (False, "321143", "Dance"),
            (False, "923247", "Rock"),
            (False, "301195", "Summer"),
            (False, "490485", "Hawaii"),
            (False, "575612", "Classic Country"),
            (False, "269088", "older songs"),
            (False, "606436", "2016"),
            (False, "701866", "Dance"),
            (False, "608829", "FINESSE"),
            (False, "273344", "Oldies"),
            (False, "501054", "Rock"),
            (False, "750528", "sports"),
            (False, "684261", "Christian"),
            (False, "44648", "gaming"),
            (False, "837665", "classics"),
            (False, "786219", "Party"),
            (False, "47214", "workout"),
            (False, "889395", "work"),
            (False, "497427", "Love songs"),
            (False, "677006", "Summer")
        ]

    for cluster_id, test_pid, playlist_name in test_set:
        top_songs = recommend(playlist_name, top_k, tokenizer, model, playlist_embeddings, playlist_tracks,
                              playlist_titles, printing=True)

        # Write to CSV
        csv_filename = os.path.join(output_folder, f'top_songs_for_{test_pid}.csv')
        with open(csv_filename, mode="w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            # Write header
            writer.writerow(["rank", "uri", "song", "artist", "occurrences"])
            # Write rows
            for i, ((uri, song, artist), count) in enumerate(top_songs, start=1):
                writer.writerow([i, uri, song, artist, count])

        relevant_songs_info = playlist_tracks.get(test_pid, [])
        relevant_songs = list({
            (trk["track_uri"],trk["track_name"], trk["artist_name"]) for trk in relevant_songs_info
        })
        csv_filename = os.path.join(output_folder, f'gold_standard_for_{test_pid}.csv')
        with open(csv_filename, mode="w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            # Write header
            writer.writerow(["rank", "uri", "song", "artist"])
            # Write rows
            for i, (uri, song, artist) in enumerate(relevant_songs, start=1):
                writer.writerow([i, uri, song, artist])


        recommended_songs = [track for track, _ in top_songs]
        metrics = compute_metrics(recommended_songs, relevant_songs, top_n=66)

        result_row = {
            "Cluster ID": cluster_id,
            "Playlist ID": test_pid,
            "Playlist Title": playlist_name,
            "HIT@66": metrics["HIT@N"],
            "Precision@66": metrics["Precision@N"],
            "Recall@66": metrics["Recall@N"],
            "MRR@66": metrics["MRR@N"],
            "R-Precision": metrics["R-Precision"],
            "NDCG@66": metrics["NDCG"]
        }
        all_results.append(result_row)

    # Save each result in a csv file
    fieldnames = [
        "Cluster ID", "Playlist ID", "Playlist Title",
        "HIT@66", "Precision@66", "Recall@66", "MRR@66",
        "R-Precision", "NDCG@66"
    ]
    results_csv = os.path.join(output_folder, f'results_{test_pid}')
    with open(results_csv, 'w', encoding='utf8', newline='') as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)

    print(f"\nResults saved in '{results_csv}'.")


if __name__ == "__main__":
    main()
