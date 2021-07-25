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
[0.00,0.00,0.00,0.00,2.67,1.83,0.00,0.00,0.00,0.00,0.84,4.96,7.63,4.58,0.00,0.00,0.00,0.99,6.86,7.32,7.62,3.65,0.00,0.00,0.30,6.71,7.09,1.22,7.55,4.04,0.00,0.00,0.31,5.11,1.52,0.00,6.10,5.64,0.00,0.00,0.00,0.00,0.00,1.07,6.48,6.10,0.00,0.00,0.00,0.00,0.00,2.90,7.62,6.10,0.00,0.00,0.00,0.00,0.00,0.23,4.58,4.42,0.00,0.00],
[0.23,0.76,0.30,0.00,0.00,0.00,0.00,0.00,4.35,7.62,7.47,5.95,4.20,2.97,0.69,0.00,0.92,2.59,3.96,5.88,7.32,7.62,4.42,0.00,0.00,0.00,0.00,0.00,2.74,7.62,2.44,0.00,0.00,0.00,0.00,1.07,7.09,5.95,0.08,0.00,0.00,0.00,1.14,6.63,7.09,0.91,0.00,0.00,0.00,0.46,6.17,7.02,1.37,0.00,0.00,0.00,0.00,4.35,7.62,7.08,5.11,5.26,0.69,0.00],
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,4.35,5.03,3.05,1.00,0.00,0.00,0.00,0.00,4.50,6.79,7.62,7.63,4.96,0.00,0.00,0.00,0.00,3.51,7.47,6.48,3.20,0.00,0.00,0.00,2.59,7.62,6.48,0.69,0.00,0.00,0.00,0.00,1.98,7.62,7.62,5.80,0.00,0.00,0.00,0.00,5.87,7.63,7.17,4.20,0.00,0.00,0.00,0.00,1.37,1.53,0.00,0.00,0.00,0.00,0.00],
[0.00,0.00,0.00,0.61,2.90,0.23,0.00,0.00,0.00,0.00,0.52,6.48,7.55,1.15,0.00,0.00,0.00,0.92,6.10,7.40,6.55,2.21,0.00,0.00,0.77,7.17,7.25,2.13,6.86,3.81,0.00,0.00,2.90,7.62,6.64,5.34,7.40,6.48,4.88,0.00,0.23,3.51,5.11,5.57,7.62,6.25,4.88,0.00,0.00,0.00,0.00,0.53,7.40,1.91,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00]
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
