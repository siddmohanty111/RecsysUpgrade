"""

Clustering module that performs cluster cleaning and uses new clustering algorithms, such as 
spectral clustering and fuzzy c-means.

"""

__version__ = "0.1.0"

from .clean_clusters import clean_clusters
from .cluster_alts import fuzzy
from .lsh_cluster_picking import prune_clusters

__all__ = [
    "clean_clusters", 
    "fuzzy", 
    "prune_clusters"
]