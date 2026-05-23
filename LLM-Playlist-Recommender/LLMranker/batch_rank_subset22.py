"""
Batch ranking script for subset_22 playlists across multiple LLMs.
Runs the ranker on all 22 playlists with different models.
"""

import os
import sys
import json
import yaml
from pathlib import Path
from datetime import datetime
import argparse
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import ranker function
from ranker import rank_playlist, load_prompt

# Available LLMs to test
AVAILABLE_LLMS = {
    "llama3.1": {"description": "Ollama (llama3.1)"},
    "llama3.2": {"description": "Ollama (llama3.2)"},
    "mistral-large": {"description": "Ollama (mistral-large)"},
    "mistral-small3.2": {"description": "Ollama (mistral-small3.2)"},
    "zephyr": {"description": "Ollama (zephyr)"},
    "gemma3": {"description": "Ollama (gemma3)"},
    "qwen3": {"description": "Ollama (qwen3)"},
    "gpt4o": {"description": "OpenAI"},
    "gpt41": {"description": "OpenAI"},
    "gemini": {"description": "Google"},
    "claude-latest": {"description": "Anthropic"},
}

# Map user-friendly names to Ollama model IDs
MODEL_ALIASES = {
    "llama3.1": "llama3.1",
    "llama3.2": "llama3.2",
    "mistral-large": "mistral-large",
    "mistral-small3.2": "mistral-small3.2",
    "zephyr": "zephyr",
    "gemma3": "gemma3",
    "qwen3": "qwen3",
}


# Default LLMs to use
DEFAULT_LLMS = [
    "gpt4o",
    "gpt41",
    # "llama3.1",
    # "llama3.2",
    # "mistral-large",
    # "mistral-small3.2",
    # "zephyr",
    # "gemma3",
    # "gemini",
    # "claude-latest",
    # "qwen3",
]


def load_playlists_yaml(yaml_path: str) -> dict:
    """Load playlists from YAML file."""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data['playlists']


def run_ranker(title: str, songs_df, output_csv: str, 
               model: str, prompt_path: str = './LLMranker/prompt_ranker.txt',
               temperature: float = 0.2, reasoning: bool = False) -> bool:
    """Call rank_playlist function directly."""
    try:
        prompt_text = load_prompt(prompt_path)
        
        # Call rank_playlist directly
        df_ranked = rank_playlist(
            title=title,
            songs_df=songs_df,
            prompt_text=prompt_text,
            model=model,
            temperature=temperature,
            reasoning=reasoning,
        )
        
        # Save to CSV
        df_ranked.to_csv(output_csv, index=False)
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run playlist ranker across multiple LLMs"
    )
    parser.add_argument(
        '--llms', nargs='+', default=DEFAULT_LLMS,
        help=f"LLMs to test. Available: {', '.join(AVAILABLE_LLMS.keys())}"
    )
    parser.add_argument(
        '--yaml', default='./subset_22.yml',
        help="Path to playlists YAML file (for playlist titles and PIDs)"
    )
    parser.add_argument(
        '--voting-system-dir', default='./similarity/output',
        help="Directory containing voting system output (top_songs_for_*.csv)"
    )
    parser.add_argument(
        '--output-dir', default='./ranking_results',
        help="Base output directory for results"
    )
    parser.add_argument(
        '--temperature', type=float, default=0.2,
        help="LLM temperature parameter"
    )
    parser.add_argument(
        '--reasoning', action='store_true',
        help="Enable reasoning mode for structured outputs"
    )
    parser.add_argument(
        '--list-avail', action='store_true',
        help="List available LLMs and exit"
    )
    parser.add_argument(
        '--timestamp', default=None,
        help="Timestamp index for this run (default: current datetime"
    )
    
    args = parser.parse_args()
    
    # Show available LLMs
    if args.list_avail:
        print("Available LLMs:")
        for model, info in AVAILABLE_LLMS.items():
            print(f"  - {model}: {info['description']}")
        return 0
    
    # Validate LLMs
    for llm in args.llms:
        if llm not in AVAILABLE_LLMS:
            print(f"Unknown LLM: {llm}")
            print(f"Available: {', '.join(AVAILABLE_LLMS.keys())}")
            return 1
    
    # Load playlists
    if not os.path.exists(args.yaml):
        print(f"Error: {args.yaml} not found")
        return 1
    
    print(f"Loading playlists from {args.yaml}...")
    playlists = load_playlists_yaml(args.yaml)
    print(f"Loaded {len(playlists)} playlists\n")
    
    # Check voting system directory
    voting_dir = Path(args.voting_system_dir)
    if not voting_dir.exists():
        print(f"Error: Voting system directory not found: {voting_dir}")
        return 1
    
    # Create base output directory
    base_output = Path(args.output_dir)
    base_output.mkdir(parents=True, exist_ok=True)
    
    # Timestamp for this run
    timestamp = args.timestamp
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Process each LLM
    results_summary = {}
    
    for llm_name in args.llms:
        print(f"\n{'='*60}")
        print(f"Testing LLM: {llm_name}")
        print(f"{'='*60}")
        
        # Create LLM output directory
        llm_output_dir = base_output / llm_name / timestamp
        llm_output_dir.mkdir(parents=True, exist_ok=True)
        
        results_summary[llm_name] = {
            "total": len(playlists),
            "success": 0,
            "failed": 0,
            "failures": []
        }
        
        # Process each playlist
        for idx, playlist in enumerate(playlists, 1):
            pid = playlist['pid']
            title = playlist['playlist_title']
            
            print(f"  [{idx}/{len(playlists)}] {title} (PID: {pid})...", end=' ', flush=True)
            
            try:
                # Load songs from voting system output
                voting_csv = voting_dir / f"top_songs_for_{pid}.csv"
                
                if not voting_csv.exists():
                    print(f"✗ Voting system output not found: {voting_csv.name}")
                    results_summary[llm_name]["failed"] += 1
                    results_summary[llm_name]["failures"].append(f"{pid}: no voting system output")
                    continue
                
                # Read voting system recommendations
                songs_df = pd.read_csv(voting_csv)
                
                # Verify required columns exist
                if not all(col in songs_df.columns for col in ['uri', 'song', 'artist']):
                    print(f"✗ Invalid CSV format (missing required columns)")
                    results_summary[llm_name]["failed"] += 1
                    results_summary[llm_name]["failures"].append(f"{pid}: invalid CSV format")
                    continue
                
                if len(songs_df) == 0:
                    print(f"✗ No tracks in voting system output")
                    results_summary[llm_name]["failed"] += 1
                    results_summary[llm_name]["failures"].append(f"{pid}: empty voting system output")
                    continue
                
                # Keep only required columns
                songs_df = songs_df[['uri', 'song', 'artist']]
                
                # Run ranker
                output_csv = llm_output_dir / f"ranked_{pid}.csv"
                if os.path.exists(output_csv):
                    print(f"✓ Already exists, skipping")
                    continue
                model_id = MODEL_ALIASES.get(llm_name, llm_name)
                success = run_ranker(
                    title=title,
                    songs_df=songs_df,
                    output_csv=str(output_csv),
                    model=model_id,
                    temperature=args.temperature,
                    reasoning=args.reasoning,
                )
                
                if success:
                    results_summary[llm_name]["success"] += 1
                    print(f"✓ Completed")
                else:
                    results_summary[llm_name]["failed"] += 1
                    results_summary[llm_name]["failures"].append(f"{pid}: ranking failed")
                    
            except Exception as e:
                print(f"✗ Exception: {e}")
                results_summary[llm_name]["failed"] += 1
                results_summary[llm_name]["failures"].append(f"{pid}: {str(e)}")
        
        # Print LLM summary
        success = results_summary[llm_name]["success"]
        failed = results_summary[llm_name]["failed"]
        print(f"\n  Summary: {success}/{len(playlists)} successful")
        if failed > 0 and results_summary[llm_name]["failures"]:
            print(f"  Failed playlists:")
            for failure in results_summary[llm_name]["failures"][:3]:
                print(f"    - {failure}")
            if len(results_summary[llm_name]["failures"]) > 3:
                print(f"    ... and {len(results_summary[llm_name]['failures']) - 3} more")
    
    # Save overall summary
    summary_file = base_output / f"summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    for llm_name, stats in results_summary.items():
        success_rate = (stats['success'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"{llm_name:20s}: {stats['success']:2d}/{stats['total']} ({success_rate:5.1f}%)")
    
    print(f"\nResults saved to: {base_output}")
    print(f"Summary: {summary_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
