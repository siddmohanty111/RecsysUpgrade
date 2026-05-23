import argparse
import os
import torch
import pickle
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_fine_tuned_model(model_dir, base_model_name='sentence-transformers/all-MiniLM-L6-v2'):
    """Load a fine-tuned classification model and its tokenizer.

    Args:
        model_dir: Path to the saved fine-tuned model directory.
        base_model_name: Fallback base model name (unused here; tokenizer is loaded from model_dir).

    Returns:
        Tuple of (tokenizer, model, device).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, output_hidden_states=True)
    model.eval()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    return tokenizer, model, device


def get_embedding(text, tokenizer, model, device):
    """Compute a mean-pooled embedding from the last hidden state.

    Args:
        text: Playlist title string.
        tokenizer: HuggingFace tokenizer.
        model: HuggingFace model with hidden states enabled.
        device: torch.device.

    Returns:
        1-D numpy embedding vector.
    """
    if not isinstance(text, str) or pd.isna(text):
        text = ""

    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(device)
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        last_hidden_state = outputs.hidden_states[-1]
        embedding = last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

    return embedding


def load_playlist_titles(playlists_csv):
    """Load pid -> title mapping from playlists.csv.

    Args:
        playlists_csv: Path to playlists.csv.

    Returns:
        Dict mapping pid -> title string.
    """
    if not os.path.exists(playlists_csv):
        raise FileNotFoundError(f"CSV not found: {playlists_csv}")
    df = pd.read_csv(playlists_csv)
    df['name'] = df['name'].fillna('')
    return dict(zip(df['pid'], df['name']))


def compute_and_save_playlist_embeddings(playlists_csv, output_file, tokenizer, model, device):
    """Compute embeddings for all playlist titles and save to a pickle file.

    Args:
        playlists_csv: Path to playlists.csv.
        output_file: Path where the embeddings pickle will be written.
        tokenizer: HuggingFace tokenizer.
        model: Fine-tuned HuggingFace model.
        device: torch.device.
    """
    playlist_embeddings = {}
    pid_to_title = load_playlist_titles(playlists_csv)

    problematic_pids = []
    for pid, title in tqdm(pid_to_title.items(), desc="Computing embeddings", unit="playlist"):
        try:
            embedding = get_embedding(title, tokenizer, model, device)
            playlist_embeddings[pid] = {
                "embedding": embedding,
                "title": title,
            }
        except Exception as e:
            problematic_pids.append(pid)
            print(f"Error for pid {pid}: {e}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(playlist_embeddings, f)
    print(f"Playlist embeddings saved successfully to {output_file}.")

    if problematic_pids:
        problem_file = os.path.splitext(output_file)[0] + "_problematic_pids.pkl"
        with open(problem_file, 'wb') as f:
            pickle.dump(problematic_pids, f)
        print(f"Problematic pids saved in {problem_file}")


def run(playlists_csv, output_file, finetuned_model_dir):
    """End-to-end: load model, compute embeddings, save pickle.

    Args:
        playlists_csv: Path to playlists.csv.
        output_file: Path where the embeddings pickle will be written.
        finetuned_model_dir: Path to the saved fine-tuned model directory.
    """
    tokenizer, model, device = load_fine_tuned_model(finetuned_model_dir)
    print("Loaded fine-tuned classification model.")
    compute_and_save_playlist_embeddings(playlists_csv, output_file, tokenizer, model, device)


def main():
    parser = argparse.ArgumentParser(description="Generate playlist-title embeddings using the fine-tuned model.")
    parser.add_argument("--playlists_csv", type=str, default="/data/csvs/playlists.csv")
    parser.add_argument("--output_file", type=str,
                        default="/home/vellard/playlist_continuation/playlists_embeddings/final_embeddings/playlists_embeddings_scheduler.pkl")
    parser.add_argument("--finetuned_model_dir", type=str,
                        default="/home/vellard/playlist_continuation/fine_tuned_model_no_scheduler_2")
    args = parser.parse_args()
    run(args.playlists_csv, args.output_file, args.finetuned_model_dir)


if __name__ == "__main__":
    main()

