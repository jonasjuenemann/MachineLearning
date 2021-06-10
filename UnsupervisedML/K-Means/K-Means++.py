"""Generell Unterschied zum K-Means -> Beginner Centroids werden nicht mehr voellig zufaellig verteilt.
Der erste Step aus dem K-Means Algorithmus wird dafuer geaendert in:
1.1 The first cluster centroid is randomly picked from the data points.
1.2 For each remaining data point, the distance from the point to its nearest cluster centroid is calculated.
1.3 The next cluster centroid is picked according to a probability proportional to the distance of each point to its nearest cluster centroid.
    This makes it likely for the next cluster centroid to be far away from the already initialized centroids.
Kurzgefasst: es wird ein Datapoint sehr weit weg vom ersten Centroid (bzw. den bisherigen Centroids) als naechster Centroid gewaehlt
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from copy import deepcopy
from sklearn.cluster import KMeans

x = [1, 1, 4, 4]
y = [1, 3, 1, 3]

values = np.array(list(zip(x, y)))

centroids_x = [2.5, 2.5]
centroids_y = [1, 3]

centroids = np.array(list(zip(centroids_x, centroids_y)))

model = KMeans(init=centroids, n_clusters=2)
# Initial centroids
# plt.scatter(centroids[:, 0], centroids[:, 1], marker='D', s=100)

results = model.fit_predict(values)

plt.scatter(x, y, c=results, alpha=1)

# Cluster centers
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], marker='v', s=100)

ax = plt.subplot()
ax.set_xticks([0, 1, 2, 3, 4, 5])
ax.set_yticks([0, 1, 2, 3, 4])

plt.title('K-Means random Initialization')
plt.show()
print("The model's inertia is " + str(model.inertia_))

# k-means++ ist der default, muss also nicht eingegeben werden, "random" fuer den nromalen Zufalls init (oder wie bei uns: vorgegebene (zufalls-) Centroids)
model = KMeans(init="k-means++", n_clusters=2)

results = model.fit_predict(values)

plt.scatter(x, y, c=results, alpha=1)

# Cluster centers
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], marker='v', s=100)

ax = plt.subplot()
ax.set_xticks([0, 1, 2, 3, 4, 5])
ax.set_yticks([0, 1, 2, 3, 4])

plt.title('K-Means random Initialization')
plt.show()
print("The model's inertia is " + str(model.inertia_))