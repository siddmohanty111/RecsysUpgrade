#############################################
# Code to add the exact matches percentages #
#############################################

import argparse
import os
import csv
from collections import Counter


def analyze_clusters_with_exact_matches(input_file, output_file):
    """Annotate a clusters CSV with an 'Exact Match Percentage' column.

    For each cluster the percentage is computed as:
        (count of the most frequent playlist title) / (total playlists in cluster) * 100

    Args:
        input_file: Path to an existing clusters CSV (must have 'Cluster ID' and 'Playlist Title' columns).
        output_file: Path where the annotated CSV will be written.
    """
    cluster_titles = {}
    exact_match_percentages = {}

    with open(input_file, 'r', encoding='utf8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cluster_id = row["Cluster ID"]
            title = row["Playlist Title"]
            if cluster_id not in cluster_titles:
                cluster_titles[cluster_id] = []
            cluster_titles[cluster_id].append(title)

    for cluster_id, titles in cluster_titles.items():
        title_counts = Counter(titles)
        most_frequent_count = max(title_counts.values())
        total_titles = len(titles)
        exact_match_percentages[cluster_id] = (most_frequent_count / total_titles) * 100

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(input_file, 'r', encoding='utf8') as f_in, \
         open(output_file, 'w', newline='', encoding='utf8') as f_out:
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)

        header = next(reader)
        new_header = header[:3] + ["Exact Match Percentage"] + header[3:]
        writer.writerow(new_header)

        for row in reader:
            cluster_id = row[0]
            exact_match = exact_match_percentages[cluster_id]
            new_row = row[:3] + [f"{exact_match:.2f}"] + row[3:]
            writer.writerow(new_row)

    print(f"Clusters with percentages saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Add exact-match percentage column to clusters CSV.")
    parser.add_argument("--input_file", type=str,
                        default="/home/vellard/playlist_continuation/clustering-no-split/clusters/200/clusters.csv")
    parser.add_argument("--output_file", type=str,
                        default="/home/vellard/playlist_continuation/clustering-no-split/clusters/200/clusters_with_exact_matches.csv")
    args = parser.parse_args()
    analyze_clusters_with_exact_matches(args.input_file, args.output_file)


if __name__ == "__main__":
    main()


