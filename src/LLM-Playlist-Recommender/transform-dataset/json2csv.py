######################################################
# adapted from previous project (mdp2csv.py) and gpt #
######################################################

import argparse
import csv
import os
import json
from os import listdir, path


def convert(input_dir, output_dir):
    """Convert MPD JSON slices to CSV files.

    Args:
        input_dir: Path to the directory containing MPD .json slice files.
        output_dir: Path to the directory where CSV files will be written.
    """
    os.makedirs(output_dir, exist_ok=True)

    items_file = open(path.join(output_dir, 'items.csv'), 'w', newline='', encoding='utf8')
    playlists_file = open(path.join(output_dir, 'playlists.csv'), 'w', newline='', encoding='utf8')
    tracks_file = open(path.join(output_dir, 'tracks.csv'), 'w', newline='', encoding='utf8')
    playlists_descr_file = open(path.join(output_dir, 'playlists_descr.csv'), 'w', newline='', encoding='utf8')

    items_writer = csv.writer(items_file)
    playlists_writer = csv.writer(playlists_file)
    tracks_writer = csv.writer(tracks_file)
    playlists_descr_writer = csv.writer(playlists_descr_file)

    items_writer.writerow(['pid', 'track_position', 'track_uri'])
    playlists_writer.writerow(['pid', 'name', 'collaborative', 'num_tracks', 'num_artists',
                               'num_albums', 'num_followers', 'num_edits', 'modified_at', 'duration_ms'])
    playlists_descr_writer.writerow(['pid', 'description'])
    tracks_writer.writerow(['track_uri', 'track_name', 'artist_uri', 'artist_name', 'album_uri', 'album_name', 'duration_ms'])

    unique_tracks = set()

    for mpd_slice in listdir(input_dir):
        if mpd_slice.endswith('.json'):
            with open(path.join(input_dir, mpd_slice), encoding='utf8') as json_file:
                print(f"Reading file {mpd_slice}...")
                json_slice = json.load(json_file)

                for playlist in json_slice['playlists']:
                    playlists_writer.writerow([
                        playlist['pid'], playlist['name'], playlist['collaborative'],
                        playlist.get('num_tracks', 0), playlist.get('num_artists', 0),
                        playlist.get('num_albums', 0), playlist.get('num_followers', 0),
                        playlist.get('num_edits', 0), playlist['modified_at'],
                        playlist.get('duration_ms', 0)
                    ])

                    if 'description' in playlist:
                        playlists_descr_writer.writerow([playlist['pid'], playlist['description']])

                    for track in playlist['tracks']:
                        items_writer.writerow([playlist['pid'], track['pos'], track['track_uri']])

                        if track['track_uri'] not in unique_tracks:
                            unique_tracks.add(track['track_uri'])
                            tracks_writer.writerow([
                                track['track_uri'], track['track_name'],
                                track['artist_uri'], track['artist_name'],
                                track['album_uri'], track['album_name'], track['duration_ms']
                            ])

    items_file.close()
    playlists_file.close()
    tracks_file.close()
    playlists_descr_file.close()

    print("Conversion complete! CSV files are available in the output folder.")


def main():
    parser = argparse.ArgumentParser(description="Convert MPD JSON slices to CSV files.")
    parser.add_argument("--input_dir", type=str, default='/data/million_playlist_dataset',
                        help="Path to the directory containing MPD .json slice files.")
    parser.add_argument("--output_dir", type=str, default='/data/playlist_continuation_data/csvs',
                        help="Path to the directory where CSV files will be written.")
    args = parser.parse_args()
    convert(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
