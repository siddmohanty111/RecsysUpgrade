#########################################
# Code to remove miscellaneous clusters #
#########################################

import argparse
import os
import csv


def clean_clusters(input_file, output_file, threshold=2.0):
    """Remove clusters whose exact-match percentage is at or below a threshold.

    Args:
        input_file: Path to clusters CSV containing an 'Exact Match Percentage' column.
        output_file: Path where the cleaned CSV will be written.
        threshold: Clusters with a percentage <= this value are dropped.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(input_file, 'r', newline='', encoding='utf8') as infile, \
         open(output_file, 'w', newline='', encoding='utf8') as outfile:

        reader = csv.DictReader(infile, delimiter=',')
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter=',')
        writer.writeheader()

        for row in reader:
            try:
                percentage = float(row["Exact Match Percentage"].replace('%', '').strip())
                if percentage > threshold:
                    writer.writerow(row)
            except ValueError:
                continue


def main():
    parser = argparse.ArgumentParser(description="Remove miscellaneous clusters below an exact-match threshold.")
    parser.add_argument("--input_file", type=str,
                        default="/home/vellard/playlist_continuation/clustering-no-split/clusters/200/clusters_with_exact_matches.csv")
    parser.add_argument("--output_file", type=str,
                        default="/home/vellard/playlist_continuation/clustering-no-split/clean/200/clusters_with_exact_matches.csv")
    parser.add_argument("--threshold", type=float, default=2.0,
                        help="Clusters with exact-match percentage <= this value are removed.")
    args = parser.parse_args()
    clean_clusters(args.input_file, args.output_file, args.threshold)


if __name__ == "__main__":
    main()

