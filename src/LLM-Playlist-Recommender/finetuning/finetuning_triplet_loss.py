import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sentence_transformers import SentenceTransformer, InputExample, losses


def run(train_csv, val_csv, output_dir,
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        batch_size=8, epochs=50, learning_rate=2e-5):
    """Fine-tune a SentenceTransformer model using BatchAll triplet loss on cluster labels.

    Args:
        train_csv: Path to clusters_train.csv.
        val_csv: Path to clusters_val.csv.
        output_dir: Directory where the fine-tuned model will be saved.
        model_name: HuggingFace model identifier to start from.
        batch_size: DataLoader batch size.
        epochs: Number of training epochs.
        learning_rate: AdamW learning rate.
    """
    train_df = pd.read_csv(train_csv, low_memory=False)
    val_df = pd.read_csv(val_csv, low_memory=False)

    train_df['Cluster ID'] = train_df['Cluster ID'].astype(int)
    val_df['Cluster ID'] = val_df['Cluster ID'].astype(int)

    unique_train_labels = sorted(train_df['Cluster ID'].unique())
    label_mapping = {orig_label: new_label for new_label, orig_label in enumerate(unique_train_labels)}

    train_df['Mapped Label'] = train_df['Cluster ID'].map(label_mapping)
    val_df = val_df[val_df['Cluster ID'].isin(label_mapping.keys())].copy()
    val_df['Mapped Label'] = val_df['Cluster ID'].map(label_mapping)

    def create_input_examples(df):
        return [
            InputExample(texts=[str(row['Playlist Title'])], label=row['Mapped Label'])
            for _, row in df.iterrows()
        ]

    train_examples = create_input_examples(train_df)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    model = SentenceTransformer(model_name, device=device)
    train_loss = losses.BatchAllTripletLoss(model=model)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=100,
        output_path=output_dir,
        optimizer_class=AdamW,
        optimizer_params={'lr': learning_rate},
        show_progress_bar=True,
    )

    model.save(output_dir)
    print(f"Model saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune SentenceTransformer with triplet loss on cluster labels.")
    parser.add_argument("--train_csv", type=str, default='/home/vellard/playlist_continuation/clusters_train.csv')
    parser.add_argument("--val_csv", type=str, default='/home/vellard/playlist_continuation/clusters_val.csv')
    parser.add_argument("--output_dir", type=str, default='./final_triplet_model')
    parser.add_argument("--model_name", type=str, default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    args = parser.parse_args()
    run(args.train_csv, args.val_csv, args.output_dir,
        args.model_name, args.batch_size, args.epochs, args.learning_rate)


if __name__ == "__main__":
    main()

