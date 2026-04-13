"""

A set of cleaning functions to be applied to the clusters obtained from the clustering step. Instead of dropping clusters where the modal playlist title
appears in <= 2% of clusters, we perform a locality sensitive hashing (LSH) step inside each cluster to identify 100 subclusters of playlists with similar titles, and drop clusters where the largest subcluster contains <= 2% of the playlists in the cluster.

"""

