import re
import matplotlib.pyplot as plt

FILE = "fuzzy_clustering_finetune_output.txt"

with open(FILE, "r") as f:
    text = f.read()

# Find the table block between the header row and end of table
table_match = re.search(
    r"Epoch\s+Training Loss\s+Validation Loss\s+Accuracy(.*?)(?=Writing model shards|$)",
    text,
    re.DOTALL,
)
if not table_match:
    raise ValueError("Could not find epoch table in file.")

table_text = table_match.group(1)

# Extract all numbers in order; they appear 4 per row: epoch, train_loss, val_loss, accuracy
numbers = re.findall(r"\b\d+\.\d+|\b\d+\b", table_text)
numbers = [float(n) for n in numbers]

if len(numbers) % 4 != 0:
    raise ValueError(f"Expected multiple of 4 values, got {len(numbers)}")

epochs, train_losses, val_losses, accuracies = [], [], [], []
for i in range(0, len(numbers), 4):
    epochs.append(int(numbers[i]))
    train_losses.append(numbers[i + 1])
    val_losses.append(numbers[i + 2])
    accuracies.append(numbers[i + 3])

accuracies_pct = [a * 100 for a in accuracies]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.plot(epochs, train_losses, marker="o", label="Training Loss", zorder=3)
ax1.plot(epochs, val_losses, marker="s", linestyle="--", label="Validation Loss", zorder=3)
ax1.set_ylabel("Loss")
ax1.set_title("Training & Validation Loss per Epoch")
ax1.legend()
ax1.grid(True, alpha=0.3)
# Zoom in: use a tight margin around the actual range so tiny differences are visible
loss_min = min(min(train_losses), min(val_losses))
loss_max = max(max(train_losses), max(val_losses))
loss_pad = max((loss_max - loss_min) * 0.5, 1e-4)
ax1.set_ylim(loss_min - loss_pad, loss_max + loss_pad)
ax1.yaxis.set_major_formatter(plt.FormatStrFormatter("%.5f"))

ax2.plot(epochs, accuracies_pct, marker="o", color="green", label="Accuracy (%)", zorder=3)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Validation Accuracy per Epoch")
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, max(accuracies_pct) * 1.15 + 1)

plt.tight_layout()
plt.savefig("training_plot.png", dpi=150)
plt.show()
print("Plot saved to training_plot.png")
