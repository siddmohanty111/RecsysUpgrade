import argparse
import json
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    EarlyStoppingCallback,
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
    dropout=0.1,
    early_stopping_patience=5,
    label_smoothing=0.1,
):
    """Fine-tune a SentenceBERT model using standard cross-entropy loss on hard cluster assignments.

    The input CSVs are produced by hard clustering methods and contain a 'Cluster Labels'
    column with a single integer cluster ID for each playlist.

    Loss is standard cross-entropy: L = -log_softmax(logits)[true_label].

    Args:
        train_csv: Path to hard clusters train CSV (columns: 'Cluster Labels', 'Playlist Title').
        val_csv: Path to hard clusters val CSV.
        output_dir: Directory to save the fine-tuned model.
        model_name: HuggingFace model identifier.
        batch_size: Per-device batch size.
        epochs: Number of training epochs.
        learning_rate: AdamW learning rate.
        warmup_steps: Number of warm-up steps.
        dropout: Dropout probability applied to hidden, attention, and classifier layers.
        early_stopping_patience: Stop training if val accuracy does not improve for this many epochs.
        label_smoothing: Label smoothing factor for cross-entropy loss (0 = disabled).
    """
    train_df = pd.read_csv(train_csv, low_memory=False)
    val_df = pd.read_csv(val_csv, low_memory=False)

    train_labels = train_df["Cluster Labels"].astype(int).values
    val_labels = val_df["Cluster Labels"].astype(int).values

    num_labels = int(max(train_labels.max(), val_labels.max())) + 1

    train_df = train_df[["Playlist Title"]].copy()
    val_df = val_df[["Playlist Title"]].copy()
    train_df["hard_labels"] = train_labels.tolist()
    val_df["hard_labels"] = val_labels.tolist()

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
    config.hidden_dropout_prob = dropout
    config.attention_probs_dropout_prob = dropout
    config.classifier_dropout = dropout
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

    def tokenize_function(examples):
        texts = [str(t) for t in examples["Playlist Title"]]
        return tokenizer(texts, truncation=True, max_length=512)

    tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["Playlist Title"])
    tokenized_val = val_dataset.map(tokenize_function, batched=True, remove_columns=["Playlist Title"])

    tokenized_train = tokenized_train.rename_column("hard_labels", "labels")
    tokenized_val = tokenized_val.rename_column("hard_labels", "labels")

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
        label_smoothing_factor=label_smoothing,
    )

    class HardLabelCollator:
        """Pads token sequences dynamically and converts hard label integers to long tensors."""

        def __call__(self, features):
            labels = torch.tensor(
                [f.pop("labels") for f in features], dtype=torch.long
            )  # (B,)
            batch = tokenizer.pad(features, padding=True, return_tensors="pt")
            batch["labels"] = labels
            return batch

    def compute_metrics(eval_pred: EvalPrediction) -> dict:
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        pred_labels = np.argmax(logits, axis=-1)
        accuracy = float((pred_labels == labels).mean())
        return {"accuracy": accuracy}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
        data_collator=HardLabelCollator(),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    with open(f"{output_dir}/trainer_metrics.json", "w") as f:
        json.dump(trainer.state.log_history, f, indent=4)

    print(f"Fine-tuned model saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune SentenceBERT with standard cross-entropy loss on hard cluster assignments."
    )
    parser.add_argument("--train_csv", type=str, required=True,
                        help="Path to hard clusters train CSV (integer 'Cluster Labels' column).")
    parser.add_argument("--val_csv", type=str, required=True,
                        help="Path to hard clusters val CSV.")
    parser.add_argument("--output_dir", type=str, default="./hard_fine_tuned_model")
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout probability for hidden, attention, and classifier layers.")
    parser.add_argument("--early_stopping_patience", type=int, default=5,
                        help="Stop training after this many epochs without val accuracy improvement.")
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Label smoothing factor (0 = disabled).")
    args = parser.parse_args()
    run(
        args.train_csv, args.val_csv, args.output_dir,
        args.model_name, args.batch_size, args.epochs,
        args.learning_rate, args.warmup_steps,
        args.dropout, args.early_stopping_patience, args.label_smoothing,
    )


if __name__ == "__main__":
    main()
