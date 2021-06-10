import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
import pandas as pd

digits = datasets.load_digits()

print(digits.DESCR)
print(digits.data)
print(digits.target)
# digits_df = pd.DataFrame(digits.data) #DataFrame(digits) funktioniert nicht, offensichtlich haben hier arrays unterschied.Laengen
# print(digits_df)

plt.gray()

plt.matshow(digits.images[0])

plt.show()

print(digits.target[0])

# image zeigt eine 0
# Dasselbe fuer die ersten 64 Zahlen in einem plot

# Figure size (width, height)
fig = plt.figure(figsize=(6, 6))

# Adjust the subplots
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

# For each of the 64 images
for i in range(64):
    # Initialize the subplots: add a subplot in the grid of 8 by 8, at the i+1-th position
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    # Display an image at the i-th position
    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
    # Label the image with the target value
    ax.text(0, 7, str(digits.target[i]))
plt.show()

# K-Means
from sklearn.cluster import KMeans

# How many Clusters? Well since there are 10 digits -> make it 10 clusters
model = KMeans(n_clusters=10, random_state=42)
model.fit(digits.data)

fig = plt.figure(figsize=(8, 3))
fig.suptitle('Cluster Center Images', fontsize=14, fontweight='bold')
for i in range(10):
    # Initialize subplots in a grid of 2X5, at i+1th position
    ax = fig.add_subplot(2, 5, 1 + i)
    # Display images
    ax.imshow(model.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)
plt.show()

"""Dieses Array kann man aus der zugehoerigen html entnehmen.
Dort wird per JavaScript Die Zahleneingabe in ein 64bit array ueberfuehrt"""

new_samples = np.array([
[0.00,3.34,7.46,7.62,7.54,6.16,1.13,0.00,2.04,7.62,5.39,2.58,6.92,7.61,1.37,0.00,3.04,7.62,0.90,3.57,7.61,3.56,0.00,0.00,1.73,7.54,6.61,7.62,6.07,0.08,0.00,0.00,0.00,6.22,7.62,7.62,7.61,5.77,0.23,0.00,0.00,7.61,5.31,0.61,3.63,7.61,3.41,0.00,0.00,4.77,7.62,6.16,6.55,7.62,3.34,0.00,0.00,0.22,4.10,4.57,4.57,2.58,0.00,0.00],
[0.00,0.00,0.00,0.00,0.60,0.00,0.00,0.00,0.00,0.00,0.00,1.50,7.55,2.05,0.00,0.00,0.00,0.00,0.23,6.01,7.62,2.90,0.00,0.00,0.00,0.07,4.93,7.62,7.62,3.05,0.00,0.00,0.00,2.81,7.62,4.01,7.62,3.05,0.00,0.00,0.00,0.76,2.58,0.68,7.62,2.81,0.00,0.00,0.00,0.00,0.00,1.52,7.62,1.97,0.00,0.00,0.00,0.00,0.00,0.99,6.70,0.99,0.00,0.00],
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.91,1.90,0.76,0.76,0.00,0.00,0.00,0.00,4.48,7.62,7.62,7.61,6.68,1.35,0.00,0.00,0.52,1.97,2.29,3.71,7.62,3.80,0.00,0.00,0.00,2.73,5.33,6.93,7.62,5.02,0.68,0.00,0.00,3.04,7.62,6.93,6.62,6.86,1.60,0.00,0.00,4.94,7.54,1.81,0.00,0.00,0.00,0.00,0.00,5.02,3.49,0.00,0.00,0.00,0.00],
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.23,4.71,4.41,0.00,0.00,0.00,0.00,0.00,4.41,7.55,4.71,0.22,0.00,0.00,0.00,2.89,7.62,4.18,7.54,2.97,0.00,0.00,1.89,7.54,7.39,3.20,7.62,4.88,2.58,0.00,1.97,6.32,7.31,7.62,7.62,7.62,7.16,0.00,0.00,0.00,0.00,2.20,7.62,2.21,0.00,0.00,0.00,0.00,0.00,3.81,7.23,0.37,0.00,0.00]
])

new_labels = model.predict(new_samples)
for i in range(len(new_labels)):
    if new_labels[i] == 0:
        print(3, end='')
    elif new_labels[i] == 1:
        print(0, end='')
    elif new_labels[i] == 2:
        print(8, end='')
    elif new_labels[i] == 3:
        print(1, end='')
    elif new_labels[i] == 4:
        print(9, end='')
    elif new_labels[i] == 5:
        print(2, end='')
    elif new_labels[i] == 6:
        print(4, end='')
    elif new_labels[i] == 7:
        print(7, end='')
    elif new_labels[i] == 8:
        print(6, end='')
    elif new_labels[i] == 9:
        print(5, end='')
