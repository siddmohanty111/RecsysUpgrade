# PlaylistRecsysUpgrade

> [!WARNING]
> **Some notebooks may appear invalid on GitHub, even though they are fully functional.** This is because GitHub is missing a widget used in these notebooks. If opened in Google Colab, the code will be fully visible along with its output. If opened locally, the code will be visible, but parts of the output might show rendering issues. 

An extension of the [LLM-Playlist-Recommender](https://github.com/elea-vellard/LLM-Playlist-Recommender) paper, exploring alternative clustering strategies — Fuzzy K-Means and UMAP+HDBSCAN — for playlist recommendation via SBERT finetuning.

---

## Data

The dataset is the **Spotify Million Playlist Dataset**, available at:

> https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge/dataset_files

A free AICrowd account is required to access the data. Download it and store it somewhere accessible before proceeding. Some of the paths I use in this code are designed for my machine/colab account. Any user will need to adjust these paths to work on their machine.

---

## Reproducing Results

Follow these steps in order:

### 1. Download the Data

Download the Spotify Million Playlist Dataset from the link above and store it in a known location on your machine.

### 2. Run the Base Pipeline (`LLM-Playlist-Recommender`)

Open and run **`LLM-Playlist-Recommender/colab_pipeline.ipynb`**.

This notebook:
- Converts the raw JSON data to CSVs
- Generates mean-pooled SBERT embeddings for playlist titles and tracks
- Replicates the full original pipeline: 200 K-Means clusters → SBERT finetuning

Dependencies for this step are in `LLM-Playlist-Recommender/requirements.txt`. The remaining steps use the `PlaylistRecsysUpgrade` subrepo.

### 3. Generate Fuzzy Clusters

Open and run **`fuzzy_clustering_comparison_with_pruning.ipynb`**.

This notebook:
- Generates Fuzzy K-Means cluster assignments
- Saves both the pruned and unpruned fuzzy cluster labels

### 4. Finetune on Fuzzy Clusters

Open and run **`fuzzy_finetuning.ipynb`**.

This notebook finetunes a SBERT model on both the pruned and unpruned fuzzy cluster labels from Step 3.

### 5. UMAP + HDBSCAN Clustering

Open and run **`clustering/UMAP_HDBSCAN.ipynb`**.

This notebook:
- Performs UMAP dimensionality reduction and HDBSCAN hyperparameter tuning
- Runs HDBSCAN clustering with and without pruning
- Saves the resulting cluster assignments

### 6. Finetune on HDBSCAN Clusters

Open and run **`hdbscan_finetuning.ipynb`**.

This notebook finetunes a SBERT model on the HDBSCAN cluster labels from Step 5.

### 6.5. (Optional) Plot Finetuning Results

If you want to visualize training curves from the finetuning runs, run:

```bash
python plotting_finetuning.py
```

### 7. Evaluate Models

Open and run **`model_evaluation.ipynb`**.

This notebook compares the performance of all finetuned models against the results reported in the original paper.

---

## Repository Structure

```
PlaylistRecsysUpgrade/
├── fkm_clustering_comparison_with_pruning.ipynb   # Step 3: Fuzzy K-Means clustering
├── fuzzy_finetuning.ipynb                         # Step 4: Finetune on fuzzy clusters
├── hdbscan_finetuning.ipynb                       # Step 6: Finetune on HDBSCAN clusters
├── model_evaluation.ipynb                         # Step 7: Model evaluation
├── plotting_finetuning.py                         # Step 6.5 (optional): Plot training curves
├── clustering/
│   ├── cluster_alts.py
│   ├── lsh_cluster_picking.py
│   └── UMAP_HDBSCAN.ipynb                         # Step 5: UMAP + HDBSCAN clustering
└── finetuning/
    ├── fuzzyfinetuning_crossentropy.py
    └── hardfinetuning_crossentropy.py
```

---

## Requirements

Install dependencies with:

```bash
pip install -r ../requirements.txt
```

---

## Related Work

This project builds on:

- **LLM-Playlist-Recommender** — [elea-vellard/LLM-Playlist-Recommender](https://github.com/elea-vellard/LLM-Playlist-Recommender)

I added colab_pipeline.ipynb and final_imp_base_report_plots.ipynb to this repository, but the remaining code is entirely their work.
