"""
Evaluation script for ranked playlists.
Compares LLM rankings against ground truth from subset_22.yml.
"""

import os
import sys
import csv
import json
import yaml
import argparse
import math
import pandas as pd
from pathlib import Path
from collections import defaultdict


def compute_metrics(recommended_songs, relevant_songs, top_n=10):
    """
    Compute evaluation metrics for playlist recommendations.
    Imported from testset_test_model_args.py
    """
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


def load_ground_truth(yaml_path: str) -> dict:
    """Load ground truth playlists from YAML file."""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    ground_truth = {}
    for playlist in data['playlists']:
        pid = playlist['pid']
        # Extract tracks as tuples of (uri, song, artist)
        tracks = [
            (track['uri'], track['song'], track['artist'])
            for track in playlist['tracks']
        ]
        ground_truth[pid] = {
            'title': playlist['playlist_title'],
            'tracks': tracks
        }
    
    return ground_truth


def load_ranked_playlist(csv_path: str) -> list:
    """Load ranked playlist from CSV file."""
    ranked_tracks = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Expected columns: rank, uri, song, artist, (optionally) occurrences
            uri = row.get('uri', '').strip()
            song = row.get('song', '').strip()
            artist = row.get('artist', '').strip()
            ranked_tracks.append((uri, song, artist))
    
    return ranked_tracks


def evaluate_llm_rankings(ranking_dir: Path, ground_truth: dict, top_n: int = 10) -> dict:
    """
    Evaluate all rankings for a specific LLM/timestamp combination.
    
    Args:
        ranking_dir: Directory containing ranked_{pid}.csv files
        ground_truth: Ground truth data
        top_n: Number of top recommendations to evaluate
    
    Returns:
        Dictionary with evaluation results per playlist
    """
    results = {}
    
    # Find all ranked CSV files
    ranked_files = list(ranking_dir.glob('ranked_*.csv'))
    
    for csv_file in ranked_files:
        # Extract PID from filename (e.g., ranked_673925.csv -> 673925)
        pid = csv_file.stem.replace('ranked_', '')
        
        if pid not in ground_truth:
            print(f"Warning: PID {pid} not found in ground truth, skipping")
            continue
        
        # Load ranked and ground truth tracks
        ranked_tracks = load_ranked_playlist(csv_file)
        relevant_tracks = ground_truth[pid]['tracks']
        
        # Compute metrics
        metrics = compute_metrics(ranked_tracks, relevant_tracks, top_n=top_n)
        
        results[pid] = {
            'playlist_title': ground_truth[pid]['title'],
            'metrics': metrics,
            'num_ranked': len(ranked_tracks),
            'num_relevant': len(relevant_tracks)
        }
    
    return results


def evaluate_voting_system(voting_dir: Path, ground_truth: dict, top_n: int = 10) -> dict:
    """
    Evaluate voting system rankings.
    
    Args:
        voting_dir: Directory containing top_songs_for_{pid}.csv files
        ground_truth: Ground truth data
        top_n: Number of top recommendations to evaluate
    
    Returns:
        Dictionary with evaluation results per playlist
    """
    results = {}
    
    # Find all voting system CSV files (top_songs_for_*.csv pattern)
    ranked_files = list(voting_dir.glob('top_songs_for_*.csv'))
    
    for csv_file in ranked_files:
        # Extract PID from filename (e.g., top_songs_for_673925.csv -> 673925)
        pid = csv_file.stem.replace('top_songs_for_', '')
        
        if pid not in ground_truth:
            print(f"Warning: PID {pid} not found in ground truth, skipping")
            continue
        
        # Load ranked and ground truth tracks
        ranked_tracks = load_ranked_playlist(csv_file)
        relevant_tracks = ground_truth[pid]['tracks']
        
        # Compute metrics
        metrics = compute_metrics(ranked_tracks, relevant_tracks, top_n=top_n)
        
        results[pid] = {
            'playlist_title': ground_truth[pid]['title'],
            'metrics': metrics,
            'num_ranked': len(ranked_tracks),
            'num_relevant': len(relevant_tracks)
        }
    
    return results


def compute_aggregate_metrics(results: dict) -> dict:
    """Compute aggregate metrics across all playlists."""
    if not results:
        return {}
    
    metric_names = ['HIT@N', 'Precision@N', 'Recall@N', 'MRR@N', 'R-Precision', 'NDCG']
    aggregates = {metric: [] for metric in metric_names}
    
    for pid, data in results.items():
        for metric in metric_names:
            aggregates[metric].append(data['metrics'][metric])
    
    # Compute means
    aggregate_means = {
        f'mean_{metric}': sum(values) / len(values) if values else 0.0
        for metric, values in aggregates.items()
    }
    
    return aggregate_means


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM rankings against ground truth"
    )
    parser.add_argument(
        '--ranking-dir', default='./ranking_results',
        help="Base directory containing ranking results"
    )
    parser.add_argument(
        '--voting-system-dir', default='./similarity/output',
        help="Directory containing voting system results (top_songs_for_*.csv)"
    )
    parser.add_argument(
        '--ground-truth', default='./subset_22.yml',
        help="Path to ground truth YAML file"
    )
    parser.add_argument(
        '--top-n', type=int, default=10,
        help="Number of top recommendations to evaluate (default: 10)"
    )
    parser.add_argument(
        '--output-dir', default='./evaluation_results',
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        '--llms', nargs='*',
        help="Specific LLMs to evaluate (default: all found)"
    )
    parser.add_argument(
        '--timestamp', default=None,
        help="Specific timestamp to evaluate (default: latest for each LLM)"
    )
    parser.add_argument(
        '--include-voting-system', action='store_true', default=True,
        help="Include voting system evaluation (default: True)"
    )
    parser.add_argument(
        '--exclude-voting-system', action='store_true',
        help="Exclude voting system from evaluation"
    )
    
    args = parser.parse_args()
    
    # Load ground truth
    if not os.path.exists(args.ground_truth):
        print(f"Error: Ground truth file not found: {args.ground_truth}")
        return 1
    
    print(f"Loading ground truth from {args.ground_truth}...")
    ground_truth = load_ground_truth(args.ground_truth)
    print(f"Loaded {len(ground_truth)} playlists\n")
    
    # Setup paths
    ranking_base = Path(args.ranking_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not ranking_base.exists():
        print(f"Error: Ranking directory not found: {ranking_base}")
        return 1
    
    # Find LLM directories
    llm_dirs = [d for d in ranking_base.iterdir() if d.is_dir()]
    
    if args.llms:
        llm_dirs = [d for d in llm_dirs if d.name in args.llms]
    
    if not llm_dirs:
        print("Error: No LLM directories found")
        return 1
    
    # Store all results
    all_results = {}
    aggregate_summary = {}
    
    # Evaluate each LLM
    for llm_dir in sorted(llm_dirs):
        llm_name = llm_dir.name
        print(f"\n{'='*60}")
        print(f"Evaluating: {llm_name}")
        print(f"{'='*60}")
        
        # Find timestamp directories
        timestamp_dirs = [d for d in llm_dir.iterdir() if d.is_dir()]
        
        if args.timestamp:
            timestamp_dirs = [d for d in timestamp_dirs if d.name == args.timestamp]
        
        if not timestamp_dirs:
            print(f"  No timestamp directories found for {llm_name}")
            continue
        
        # Use latest timestamp if multiple
        timestamp_dir = sorted(timestamp_dirs)[-1]
        timestamp = timestamp_dir.name
        
        print(f"  Timestamp: {timestamp}")
        
        # Evaluate rankings
        results = evaluate_llm_rankings(timestamp_dir, ground_truth, args.top_n)
        
        if not results:
            print(f"  No results found")
            continue
        
        # Compute aggregates
        aggregates = compute_aggregate_metrics(results)
        
        print(f"  Evaluated {len(results)} playlists")
        print(f"  Mean HIT@{args.top_n}: {aggregates['mean_HIT@N']:.4f}")
        print(f"  Mean Precision@{args.top_n}: {aggregates['mean_Precision@N']:.4f}")
        print(f"  Mean Recall@{args.top_n}: {aggregates['mean_Recall@N']:.4f}")
        print(f"  Mean MRR@{args.top_n}: {aggregates['mean_MRR@N']:.4f}")
        print(f"  Mean R-Precision: {aggregates['mean_R-Precision']:.4f}")
        print(f"  Mean NDCG@{args.top_n}: {aggregates['mean_NDCG']:.4f}")
        
        # Store results
        all_results[llm_name] = {
            'timestamp': timestamp,
            'playlists': results,
            'aggregates': aggregates
        }
        
        aggregate_summary[llm_name] = aggregates
        
        # Save detailed results for this LLM
        detailed_csv = output_dir / f'{llm_name}_{timestamp}_detailed.csv'
        with open(detailed_csv, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'playlist_id', 'playlist_title', 'num_ranked', 'num_relevant',
                f'HIT@{args.top_n}', f'Precision@{args.top_n}', 
                f'Recall@{args.top_n}', f'MRR@{args.top_n}', 
                'R-Precision', f'NDCG@{args.top_n}'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for pid, data in sorted(results.items()):
                writer.writerow({
                    'playlist_id': pid,
                    'playlist_title': data['playlist_title'],
                    'num_ranked': data['num_ranked'],
                    'num_relevant': data['num_relevant'],
                    f'HIT@{args.top_n}': data['metrics']['HIT@N'],
                    f'Precision@{args.top_n}': data['metrics']['Precision@N'],
                    f'Recall@{args.top_n}': data['metrics']['Recall@N'],
                    f'MRR@{args.top_n}': data['metrics']['MRR@N'],
                    'R-Precision': data['metrics']['R-Precision'],
                    f'NDCG@{args.top_n}': data['metrics']['NDCG']
                })
        
        print(f"  Saved detailed results to: {detailed_csv}")
    
    # Evaluate voting system if requested
    if args.include_voting_system and not args.exclude_voting_system:
        voting_dir = Path(args.voting_system_dir)
        
        if voting_dir.exists():
            print(f"\n{'='*60}")
            print(f"Evaluating: voting-system")
            print(f"{'='*60}")
            print(f"  Directory: {voting_dir}")
            
            # Evaluate voting system
            results = evaluate_voting_system(voting_dir, ground_truth, args.top_n)
            
            if results:
                # Compute aggregates
                aggregates = compute_aggregate_metrics(results)
                
                print(f"  Evaluated {len(results)} playlists")
                print(f"  Mean HIT@{args.top_n}: {aggregates['mean_HIT@N']:.4f}")
                print(f"  Mean Precision@{args.top_n}: {aggregates['mean_Precision@N']:.4f}")
                print(f"  Mean Recall@{args.top_n}: {aggregates['mean_Recall@N']:.4f}")
                print(f"  Mean MRR@{args.top_n}: {aggregates['mean_MRR@N']:.4f}")
                print(f"  Mean R-Precision: {aggregates['mean_R-Precision']:.4f}")
                print(f"  Mean NDCG@{args.top_n}: {aggregates['mean_NDCG']:.4f}")
                
                # Store results
                all_results['voting-system'] = {
                    'timestamp': 'N/A',
                    'playlists': results,
                    'aggregates': aggregates
                }
                
                aggregate_summary['voting-system'] = aggregates
                
                # Save detailed results
                detailed_csv = output_dir / f'voting-system_detailed.csv'
                with open(detailed_csv, 'w', newline='', encoding='utf-8') as f:
                    fieldnames = [
                        'playlist_id', 'playlist_title', 'num_ranked', 'num_relevant',
                        f'HIT@{args.top_n}', f'Precision@{args.top_n}', 
                        f'Recall@{args.top_n}', f'MRR@{args.top_n}', 
                        'R-Precision', f'NDCG@{args.top_n}'
                    ]
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for pid, data in sorted(results.items()):
                        writer.writerow({
                            'playlist_id': pid,
                            'playlist_title': data['playlist_title'],
                            'num_ranked': data['num_ranked'],
                            'num_relevant': data['num_relevant'],
                            f'HIT@{args.top_n}': data['metrics']['HIT@N'],
                            f'Precision@{args.top_n}': data['metrics']['Precision@N'],
                            f'Recall@{args.top_n}': data['metrics']['Recall@N'],
                            f'MRR@{args.top_n}': data['metrics']['MRR@N'],
                            'R-Precision': data['metrics']['R-Precision'],
                            f'NDCG@{args.top_n}': data['metrics']['NDCG']
                        })
                
                print(f"  Saved detailed results to: {detailed_csv}")
            else:
                print(f"  No results found")
        else:
            print(f"\nWarning: Voting system directory not found: {voting_dir}")
    
    # Save aggregate summary
    if aggregate_summary:
        print(f"\n{'='*60}")
        print("AGGREGATE SUMMARY")
        print(f"{'='*60}")
        
        summary_csv = output_dir / f'aggregate_summary_top{args.top_n}.csv'
        with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'llm', f'mean_HIT@{args.top_n}', f'mean_Precision@{args.top_n}',
                f'mean_Recall@{args.top_n}', f'mean_MRR@{args.top_n}',
                'mean_R-Precision', f'mean_NDCG@{args.top_n}'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for llm_name, metrics in sorted(aggregate_summary.items()):
                row = {'llm': llm_name}
                row.update({
                    f'mean_HIT@{args.top_n}': metrics['mean_HIT@N'],
                    f'mean_Precision@{args.top_n}': metrics['mean_Precision@N'],
                    f'mean_Recall@{args.top_n}': metrics['mean_Recall@N'],
                    f'mean_MRR@{args.top_n}': metrics['mean_MRR@N'],
                    'mean_R-Precision': metrics['mean_R-Precision'],
                    f'mean_NDCG@{args.top_n}': metrics['mean_NDCG']
                })
                writer.writerow(row)
                
                print(f"{llm_name:20s}: HIT={metrics['mean_HIT@N']:.4f}, "
                      f"Precision={metrics['mean_Precision@N']:.4f}, "
                      f"Recall={metrics['mean_Recall@N']:.4f}, "
                      f"MRR={metrics['mean_MRR@N']:.4f}, "
                      f"R-Prec={metrics['mean_R-Precision']:.4f}, "
                      f"NDCG={metrics['mean_NDCG']:.4f}")
        
        print(f"\nSummary saved to: {summary_csv}")
        
        # Save full results as JSON
        json_output = output_dir / f'full_results_top{args.top_n}.json'
        with open(json_output, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"Full results saved to: {json_output}")
    
    print(f"\n{'='*60}")
    print(f"Evaluation complete! Results saved to: {output_dir}")
    print(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
