import argparse
import json
import pandas as pd
import evaluate
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EvalPrediction


def run(train_csv, val_csv, output_dir,
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        batch_size=8, epochs=100, learning_rate=2e-5, warmup_steps=100):
    """Fine-tune a SentenceBERT model for cluster classification using cross-entropy loss.

    Args:
        train_csv: Path to clusters_train.csv.
        val_csv: Path to clusters_val.csv.
        output_dir: Directory where the fine-tuned model will be saved.
        model_name: HuggingFace model identifier to start from.
        batch_size: Per-device batch size.
        epochs: Number of training epochs.
        learning_rate: AdamW learning rate.
        warmup_steps: Number of warm-up steps.
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

    num_labels = len(label_mapping)
    train_dataset = Dataset.from_pandas(train_df[['Playlist Title', 'Mapped Label']])
    val_dataset = Dataset.from_pandas(val_df[['Playlist Title', 'Mapped Label']])

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def tokenize_function(examples):
        texts = [str(text) for text in examples["Playlist Title"]]
        return tokenizer(texts, truncation=True, padding="max_length")

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)

    tokenized_train = tokenized_train.rename_column("Mapped Label", "labels")
    tokenized_val = tokenized_val.rename_column("Mapped Label", "labels")
    tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    tokenized_val.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

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

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred: EvalPrediction) -> dict:
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        predictions = logits.argmax(axis=-1)
        results = metric.compute(predictions=predictions, references=labels)
        return results if results is not None else {}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    with open(f"{output_dir}/trainer_metrics.json", "w") as f:
        json.dump(trainer.state.log_history, f, indent=4)

    print(f"Fine-tuned model saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune SentenceBERT with cross-entropy loss on cluster labels.")
    parser.add_argument("--train_csv", type=str, default='/home/vellard/playlist_continuation/clusters_train.csv')
    parser.add_argument("--val_csv", type=str, default='/home/vellard/playlist_continuation/clusters_val.csv')
    parser.add_argument("--output_dir", type=str, default='./fine_tuned_model_no_scheduler_2')
    parser.add_argument("--model_name", type=str, default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    args = parser.parse_args()
    run(args.train_csv, args.val_csv, args.output_dir,
        args.model_name, args.batch_size, args.epochs, args.learning_rate, args.warmup_steps)


if __name__ == "__main__":
    main()

