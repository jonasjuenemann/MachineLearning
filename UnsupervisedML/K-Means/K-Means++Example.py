import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import random
import timeit

mu = 1
std = 0.5

np.random.seed(100)

xs = np.append(np.append(np.append(np.random.normal(0.25, std, 100), np.random.normal(0.75, std, 100)),
                         np.random.normal(0.25, std, 100)), np.random.normal(0.75, std, 100))

ys = np.append(np.append(np.append(np.random.normal(0.25, std, 100), np.random.normal(0.25, std, 100)),
                         np.random.normal(0.75, std, 100)), np.random.normal(0.75, std, 100))

values = list(zip(xs, ys))

model = KMeans(init='random', n_clusters=2) #man kann die Initialization randomisieren, default ist aber "init=k-means++"
results = model.fit_predict(values)
print("The inertia of model that randomly initialized centroids is " + str(model.inertia_))

colors = ['r', 'b']
plt.subplot(211)
for i in range(2):
    points = np.array([values[j] for j in range(len(values)) if results[j] == i])
    plt.scatter(points[:, 0], points[:, 1], c=colors[i], alpha=0.6)

plt.title('Centroids Initialized Randomly')

plt.xlabel('A')
plt.ylabel('B')

plt.subplot(212)
model = KMeans(n_clusters=2)
results = model.fit_predict(values)
print("The inertia of model that initialized the centroids using KMeans++ is " + str(model.inertia_))

colors = ['b', 'r']

for i in range(2):
    points = np.array([values[j] for j in range(len(values)) if results[j] == i])
    plt.scatter(points[:, 0], points[:, 1], c=colors[i], alpha=0.6)

plt.title('Centroids Initialized Using KMeans++')

plt.xlabel('A')
plt.ylabel('B')

plt.tight_layout()
plt.show()
