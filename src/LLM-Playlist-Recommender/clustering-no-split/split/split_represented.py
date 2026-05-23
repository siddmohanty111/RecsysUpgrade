import argparse
import os
import csv
import random
from os import path
from tqdm import tqdm


def split_clusters(input_clusters_file, output_dir, seed=1, val_ratio=0.1, test_ratio=0.1):
    """Split a cleaned clusters CSV into train/val/test sets.

    Each cluster is split locally so every cluster has representation in all splits.

    Args:
        input_clusters_file: Path to the cleaned clusters CSV.
        output_dir: Directory where clusters_train.csv, clusters_val.csv, clusters_test.csv are written.
        seed: Random seed for reproducibility.
        val_ratio: Fraction of each cluster to use for validation.
        test_ratio: Fraction of each cluster to use for testing.
    """
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    output_clusters_train = path.join(output_dir, 'clusters_train.csv')
    output_clusters_val   = path.join(output_dir, 'clusters_val.csv')
    output_clusters_test  = path.join(output_dir, 'clusters_test.csv')

    clusters = {}
    print(f"Reading clusters from: {input_clusters_file}")
    with open(input_clusters_file, 'r', newline='', encoding='utf8') as clusters_file:
        clusters_reader = csv.DictReader(clusters_file)
        headers = clusters_reader.fieldnames
        for row in tqdm(clusters_reader, desc="Reading clusters.csv", unit="row"):
            cluster_id = row["Cluster ID"]
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(row)

    print(f"Total clusters found: {len(clusters)}")

    with open(output_clusters_train, 'w', newline='', encoding='utf8') as train_file, \
         open(output_clusters_val,   'w', newline='', encoding='utf8') as val_file, \
         open(output_clusters_test,  'w', newline='', encoding='utf8') as test_file:

        train_writer = csv.DictWriter(train_file, fieldnames=headers)
        val_writer   = csv.DictWriter(val_file,   fieldnames=headers)
        test_writer  = csv.DictWriter(test_file,  fieldnames=headers)

        train_writer.writeheader()
        val_writer.writeheader()
        test_writer.writeheader()

        for cluster_id, rows in tqdm(clusters.items(), desc="Splitting each cluster", unit="cluster"):
            random.shuffle(rows)
            nb_total = len(rows)
            nb_val   = int(val_ratio  * nb_total)
            nb_test  = int(test_ratio * nb_total)
            nb_train = nb_total - nb_val - nb_test

            for r in rows[:nb_train]:
                train_writer.writerow(r)
            for r in rows[nb_train : nb_train + nb_val]:
                val_writer.writerow(r)
            for r in rows[nb_train + nb_val:]:
                test_writer.writerow(r)

    print(f"Train CSV : {output_clusters_train}")
    print(f"Val CSV   : {output_clusters_val}")
    print(f"Test CSV  : {output_clusters_test}")


def main():
    parser = argparse.ArgumentParser(description="Split clusters CSV into train/val/test sets.")
    parser.add_argument("--input_clusters_file", type=str,
                        default="/home/vellard/playlist_continuation/clustering-no-split/clean/200/clusters_with_exact_matches.csv")
    parser.add_argument("--output_dir", type=str,
                        default="/home/vellard/playlist_continuation/clustering-no-split/split/represented/")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    args = parser.parse_args()
    split_clusters(args.input_clusters_file, args.output_dir, args.seed, args.val_ratio, args.test_ratio)


if __name__ == "__main__":
    main()

