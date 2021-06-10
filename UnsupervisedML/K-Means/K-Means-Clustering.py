import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
import pandas as pd

iris = datasets.load_iris()
# print(iris.data)
print(iris.target)
print(iris.DESCR)

# Store iris.data
samples = iris.data
# Create x and y
x = samples[:, 0]  # all rows, column 0
y = samples[:, 1]
sepal_length_width = np.array(
    list(zip(x, y)))  # aus der description koennnen wir ja ersehen, dass column_0 die laenge, column_1 die Weite ist.

# Plot x and y
plt.scatter(x, y, alpha=0.5)
# Show the plot
plt.show()

# Number of clusters
k = 3
# Create x coordinates of k random centroids
centroids_x = np.random.uniform(min(x), max(x), k)
# Create y coordinates of k random centroids
centroids_y = np.random.uniform(min(y), max(y), k)
# print(centroids_y)
# Create centroids array
centroids = np.array(list(zip(centroids_x, centroids_y)))  # ohne np.array waere dies ein array von tupeln
print(centroids)
"""
Wir weiter unten gemacht mit Einfaerbung
# Make a scatter plot of x, y
plt.scatter(x, y)
# Make a scatter plot of the centroids
plt.scatter(centroids_x, centroids_y, color="red")
# Display plot
plt.show()
"""


# Step 2: Assign samples to nearest centroid

# Distance formula
def distance(a, b):
    one = (a[0] - b[0]) ** 2
    two = (a[1] - b[1]) ** 2
    distance = (one + two) ** 0.5
    return distance


# Cluster labels for each point (either 0, 1, or 2)
labels = np.zeros(len(samples))  # ist dasselbe wie len(x)
# Distances to each centroid
distances = np.zeros(k)
# Assign to the closest centroid

from copy import deepcopy

centroids_old = np.zeros(centroids.shape)
error = np.zeros(3)
error[0] = distance(centroids[0], centroids_old[0])
error[1] = distance(centroids[1], centroids_old[1])
error[2] = distance(centroids[2], centroids_old[2])
while error.all() != 0:
    centroids_old = deepcopy(centroids)  # stored centroids in centroids_old
    # print(sepal_length_width[0])
    plt.scatter(centroids[:, 0], centroids[:, 1], c="red", s=50, zorder=2)
    # Assign to the closest centroid
    for i in range(len(samples)):
        distances[0] = distance(sepal_length_width[i], centroids[0])
        distances[1] = distance(sepal_length_width[i], centroids[1])
        distances[2] = distance(sepal_length_width[i], centroids[2])
        label = np.argmin(distances)  # np.argmin gibt den index mit dem minimum wieder
        labels[i] = label
    for i in range(k):
        points = []
        for j in range(len(sepal_length_width)):
            if labels[j] == i:
                points.append(sepal_length_width[j])
        # print(points)
        x = [item[0] for item in points]
        y = [item[1] for item in points]
        if i == 0:
            plt.scatter(x, y, alpha=0.6, color="green")
        if i == 1:
            plt.scatter(x, y, alpha=0.6, color="blue")
        if i == 2:
            plt.scatter(x, y, alpha=0.6, color="pink")
        centroids[i] = np.mean(points,
                               axis=0)  # nimmt automatisch die indizes und berechnet mit auf den entsprechenden indice ueber das ganze array den Durchschnitt
        # Das ist recht maechtig, sonst braeuchte man hier eine for-Schleife, s.u.
    plt.show()
    error[0] = distance(centroids[0], centroids_old[0])
    error[1] = distance(centroids[1], centroids_old[1])
    error[2] = distance(centroids[2], centroids_old[2])
    """
    sum_1 = 0
    sum_2 = 0
    for i in range(len(points)):
        sum_1 += points[i][0]
        sum_2 += points[i][1]
    sum_1 = sum_1/len(points)
    sum_2 = sum_2/len(points)
    centroids[i] = [sum_1, sum_2]
    """
    print(centroids)

"""Um sich das Endergebnis anzuschauen:"""
colors = ['r', 'g', 'b']
for i in range(k):
    points = np.array([sepal_length_width[j] for j in range(len(sepal_length_width)) if labels[j] == i])
    plt.scatter(points[:, 0], points[:, 1], c=colors[i], alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='D', s=150)

plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')

plt.show()

"""K-Means mittels sklearn"""
from sklearn.cluster import KMeans

# Use KMeans() to create a model that finds 3 clusters
model = KMeans(n_clusters=3)
# Use .fit() to fit the model to samples
model.fit(samples)
# Use .predict() to determine the labels of samples
labels = model.predict(samples)
# Print the labels
print(labels)

new_samples = np.array([[5.7, 4.4, 1.5, 0.4],
                        [6.5, 3., 5.5, 0.4],
                        [5.8, 2.7, 5.1, 1.9]])

# Predict labels for the new_samples
new_labels = model.predict(new_samples)
# Print the labels
print(new_labels)

"""Scatterplot fuer sk_learn"""

x = samples[:, 0]
y = samples[:, 1]

plt.scatter(x, y, c=labels, alpha=0.5)

plt.title("sklearn-plot")
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')

plt.show()

"""Evaluation of the k-Means Model"""
target = iris.target

species = np.chararray(target.shape, itemsize=150)

for i in range(len(samples)):
    if target[i] == 0:
        species[i] = 'setosa'
    elif target[i] == 1:
        species[i] = 'versicolor'
    elif target[i] == 2:
        species[i] = 'virginica'

df = pd.DataFrame({'labels': labels, 'species': species})

print(df)

ct = pd.crosstab(df['labels'], df['species'])
print(ct)

print("Model Inertia (-> Distanz von Punkten zu Centroids) ")
print(model.inertia_)  # distance from each cluster to its centroid -> lower = better
# gleichzeitig will man aber auch eine moeglichst gerine Anzahl Cluster -> Konflikt

num_clusters = list(range(1, 9))
inertias = []

for i in num_clusters:
    model = KMeans(n_clusters=i)
    model.fit(samples)
    inertias.append(model.inertia_)

plt.plot(num_clusters, inertias, '-o')

plt.title("Development of Inertia through additional Centroids")
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')

plt.show()
