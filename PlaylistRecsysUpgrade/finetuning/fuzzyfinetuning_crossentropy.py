import argparse
import ast
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EvalPrediction,
)


def run(
    train_csv,
    val_csv,
    output_dir,
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    batch_size=8,
    epochs=5,
    learning_rate=2e-5,
    warmup_steps=100,
):
    """Fine-tune a SentenceBERT model using soft cross-entropy loss on fuzzy cluster memberships.

    The input CSVs are produced by cluster_alts.fkmeans and contain a 'Cluster Labels'
    column with a list of per-cluster membership probabilities for each playlist.

    Loss is soft cross-entropy: L = -sum(p * log_softmax(logits)), which reduces to
    standard cross-entropy when labels are one-hot.

    Args:
        train_csv: Path to fuzzy clusters train CSV (columns: 'Cluster Labels', 'Playlist Title').
        val_csv: Path to fuzzy clusters val CSV.
        output_dir: Directory to save the fine-tuned model.
        model_name: HuggingFace model identifier.
        batch_size: Per-device batch size.
        epochs: Number of training epochs.
        learning_rate: AdamW learning rate.
        warmup_steps: Number of warm-up steps.
    """
    train_df = pd.read_csv(train_csv, low_memory=False)
    val_df = pd.read_csv(val_csv, low_memory=False)

    # Parse membership vectors stored as Python-list strings by fkmeans
    def parse_memberships(series):
        return np.array([ast.literal_eval(v) for v in series], dtype=np.float32)

    train_memberships = parse_memberships(train_df["Cluster Labels"])
    val_memberships = parse_memberships(val_df["Cluster Labels"])

    num_labels = train_memberships.shape[1]

    # Normalize rows to sum to 1 (fuzzy memberships should already, but enforce it)
    train_memberships /= train_memberships.sum(axis=1, keepdims=True)
    val_memberships /= val_memberships.sum(axis=1, keepdims=True)

    train_df = train_df[["Playlist Title"]].copy()
    val_df = val_df[["Playlist Title"]].copy()
    train_df["soft_labels"] = train_memberships.tolist()
    val_df["soft_labels"] = val_memberships.tolist()

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def tokenize_function(examples):
        texts = [str(t) for t in examples["Playlist Title"]]
        # No padding here; SoftLabelCollator uses tokenizer.pad() for dynamic batch padding
        return tokenizer(texts, truncation=True, max_length=512)

    # remove_columns drops 'Playlist Title' so the collator only sees tokenizer fields + labels
    tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["Playlist Title"])
    tokenized_val = val_dataset.map(tokenize_function, batched=True, remove_columns=["Playlist Title"])

    tokenized_train = tokenized_train.rename_column("soft_labels", "labels")
    tokenized_val = tokenized_val.rename_column("soft_labels", "labels")
    # No set_format needed; tokenizer.pad() in SoftLabelCollator converts everything to tensors

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        warmup_steps=warmup_steps,
        logging_strategy="epoch",
    )

    class SoftLabelCollator:
        """Pads token sequences dynamically and converts soft label lists to float tensors."""

        def __call__(self, features):
            # Pop labels first so tokenizer.pad() only sees tokenizer fields
            labels = torch.tensor(
                [f.pop("labels") for f in features], dtype=torch.float32
            )  # (B, num_labels)
            # tokenizer.pad() handles input_ids / attention_mask / token_type_ids padding + tensor conversion
            batch = tokenizer.pad(features, padding=True, return_tensors="pt")
            batch["labels"] = labels
            return batch

    class SoftLabelTrainer(Trainer):
        """Trainer subclass that computes soft cross-entropy loss against fuzzy membership targets."""

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            soft_labels = inputs.pop("labels").float()  # (B, num_labels)
            outputs = model(**inputs)
            logits = outputs.logits  # (B, num_labels)
            log_probs = F.log_softmax(logits, dim=-1)
            loss = -(soft_labels * log_probs).sum(dim=-1).mean()
            return (loss, outputs) if return_outputs else loss

    def compute_metrics(eval_pred: EvalPrediction) -> dict:
        logits, soft_labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        pred_labels = np.argmax(logits, axis=-1)
        # Treat the highest-membership cluster as the ground-truth hard label
        true_labels = np.argmax(soft_labels, axis=-1)
        accuracy = float((pred_labels == true_labels).mean())
        return {"accuracy": accuracy}

    trainer = SoftLabelTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
        data_collator=SoftLabelCollator(),
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    with open(f"{output_dir}/trainer_metrics.json", "w") as f:
        json.dump(trainer.state.log_history, f, indent=4)

    print(f"Fine-tuned model saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune SentenceBERT with soft cross-entropy loss on fuzzy cluster memberships."
    )
    parser.add_argument("--train_csv", type=str, required=True,
                        help="Path to fuzzy clusters train CSV (from cluster_alts.fkmeans).")
    parser.add_argument("--val_csv", type=str, required=True,
                        help="Path to fuzzy clusters val CSV.")
    parser.add_argument("--output_dir", type=str, default="./fuzzy_fine_tuned_model")
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    args = parser.parse_args()
    run(
        args.train_csv, args.val_csv, args.output_dir,
        args.model_name, args.batch_size, args.epochs,
        args.learning_rate, args.warmup_steps,
    )


if __name__ == "__main__":
    main()

