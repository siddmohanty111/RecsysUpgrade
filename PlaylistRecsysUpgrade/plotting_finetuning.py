import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Dataset 1 — FKM Pruned Clusters 

# hardcoding the lists because I forgot to save notebook output... oops

fkm_pruned_epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                     11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

fkm_pruned_train_loss = [
    1.279700, 1.173494, 1.160145, 1.147947, 1.137908,
    1.132518, 1.125240, 1.117030, 1.109899, 1.103282,
    1.097661, 1.092719, 1.087986, 1.082651, 1.078308,
    1.073454, 1.069006, 1.065136, 1.061458, 1.057690,
]

fkm_pruned_val_loss = [
    1.180584, 1.182341, 1.168530, 1.168186, 1.179570,
    1.170313, 1.170764, 1.169661, 1.170147, 1.173978,
    1.180272, 1.177817, 1.184933, 1.200326, 1.197340,
    1.199803, 1.205601, 1.207800, 1.217049, 1.220904,
]

fkm_pruned_accuracy = [
    0.543741, 0.538142, 0.542796, 0.544955, 0.533219,
    0.540571, 0.547686, 0.547417, 0.545663, 0.544955,
    0.547113, 0.546675, 0.544179, 0.543066, 0.543876,
    0.542965, 0.543977, 0.543437, 0.540132, 0.542088,
]

# Dataset 2 — FKM Un-pruned Clusters 

fkm_unpruned_epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                       11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

fkm_unpruned_train_loss = [
    3.912103, 3.912102, 3.912102, 3.912102, 3.912102,
    3.912102, 3.912102, 3.912102, 3.912102, 3.912102,
    3.912102, 3.912102, 3.912102, 3.912102, 3.912102,
    3.912102, 3.912102, 3.912102, 3.912102, 3.912102,
]

fkm_unpruned_val_loss = [
    3.912022, 3.912022, 3.912022, 3.912022, 3.912022,
    3.912022, 3.912022, 3.912022, 3.912022, 3.912022,
    3.912022, 3.912022, 3.912022, 3.912022, 3.912022,
    3.912022, 3.912022, 3.912022, 3.912022, 3.912022,
]

fkm_unpruned_accuracy = [
    0.000030, 0.023540, 0.000350, 0.000070, 0.000070,
    0.411656, 0.005740, 0.000630, 0.003640, 0.000000,
    0.048930, 0.411656, 0.058139, 0.016480, 0.020860,
    0.020340, 0.043590, 0.000120, 0.155978, 0.411426,
]


# Dataset 3 — HDBSCAN Un-pruned Clusters (hard cross-entropy fine-tuning)
# Early stopping triggered at epoch 11 — full 20 epochs were not completed.

hdbscan_unpruned_epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

hdbscan_unpruned_train_loss = [
    2.546912, 2.288411, 2.242926, 2.214833, 2.194046,
    2.175144, 2.158459, 2.143848, 2.128599, 2.115070, 2.101206,
]

hdbscan_unpruned_val_loss = [
    2.313397, 2.266356, 2.248575, 2.238079, 2.238598,
    2.236557, 2.235260, 2.236173, 2.247049, 2.236657, 2.248050,
]

hdbscan_unpruned_accuracy = [
    0.590612, 0.595321, 0.600677, 0.601972, 0.600883,
    0.602678, 0.602354, 0.602501, 0.599441, 0.602119, 0.599823,
]

# Dataset 4 — HDBSCAN Pruned Clusters (hard cross-entropy fine-tuning)
# Early stopping triggered at epoch 17 — full 20 epochs were not completed.

hdbscan_pruned_epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                         11, 12, 13, 14, 15, 16, 17]

hdbscan_pruned_train_loss = [
    2.530431, 2.277381, 2.232451, 2.205051, 2.183403,
    2.165470, 2.148890, 2.133005, 2.118213, 2.104566,
    2.091311, 2.078516, 2.065698, 2.054201, 2.041474,
    2.030193, 2.018503,
]

hdbscan_pruned_val_loss = [
    2.314075, 2.270408, 2.249701, 2.243198, 2.236693,
    2.242097, 2.239102, 2.251275, 2.243088, 2.247701,
    2.251803, 2.255956, 2.262657, 2.265312, 2.277292,
    2.281776, 2.285145,
]

hdbscan_pruned_accuracy = [
    0.587317, 0.594087, 0.599172, 0.598492, 0.600976,
    0.601153, 0.599734, 0.598788, 0.601567, 0.600650,
    0.600946, 0.601626, 0.600207, 0.600532, 0.598551,
    0.599586, 0.600266,
]

# Plots

def _make_convergence_plot(epochs, train_loss, val_loss, accuracy, title):
    """Shared helper: loss curves (left) and accuracy curve (right)."""
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    ax_loss.plot(epochs, train_loss,
                 color="#008000", linestyle="--", linewidth=2, label="Training Loss")
    ax_loss.plot(epochs, val_loss,
                 color="#008000", linestyle="-", linewidth=2, label="Validation Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Training & Validation Loss")
    ax_loss.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax_loss.ticklabel_format(useOffset=False, style="plain", axis="y")
    ax_loss.legend()
    ax_loss.grid(True, linestyle=":", alpha=0.5)

    ax_acc.plot(epochs, accuracy,
                color="C0", linestyle="-", linewidth=2, marker="o", markersize=4,
                label="Validation Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Validation Accuracy")
    ax_acc.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax_acc.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax_acc.legend()
    ax_acc.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.show()


def plot_fkm_pruned():
    _make_convergence_plot(
        fkm_pruned_epochs, fkm_pruned_train_loss, fkm_pruned_val_loss,
        fkm_pruned_accuracy, "FKM Pruned — Fine-tuning Convergence",
    )


def plot_fkm_unpruned():
    _make_convergence_plot(
        fkm_unpruned_epochs, fkm_unpruned_train_loss, fkm_unpruned_val_loss,
        fkm_unpruned_accuracy, "FKM Un-pruned — Fine-tuning Convergence",
    )

def plot_hdbscan_unpruned():
    _make_convergence_plot(
        hdbscan_unpruned_epochs, hdbscan_unpruned_train_loss, hdbscan_unpruned_val_loss,
        hdbscan_unpruned_accuracy, "HDBSCAN Un-pruned — Fine-tuning Convergence",
    )

def plot_hdbscan_pruned():
    _make_convergence_plot(
        hdbscan_pruned_epochs, hdbscan_pruned_train_loss, hdbscan_pruned_val_loss,
        hdbscan_pruned_accuracy, "HDBSCAN Pruned — Fine-tuning Convergence",
    )


plot_fkm_pruned()
plot_fkm_unpruned()
plot_hdbscan_unpruned()
plot_hdbscan_pruned()

