from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

"""Perceptrons can’t solve problems that aren’t linearly separable. 
However, if you combine multiple perceptrons together, you now have a neural net that can solve these problems!"""

data = [[0, 0], [0, 1], [1, 0], [1, 1]]
labels = [0, 0, 0, 1]
plt.scatter([x[0] for x in data], [y[1] for y in data], c=labels)
# c -> color: points with label 1 sind andersferbig (hier gelb)
classifier = Perceptron(max_iter=40)
# default 1000 -> brauchen wir bei unseren leichten Daten aber nicht
classifier.fit(data, labels)
print(classifier.decision_function([[0, 0], [1, 1], [0.5, 0.5]]))
print(classifier.score(data, labels))
x_values = np.linspace(0, 1, 100)
# print(x_values) # 0-1 in hundert Schritten (also 0.01,0.02...)
y_values = np.linspace(0, 1, 100)
point_grid = list(product(x_values, y_values))  # basically ein outer_join
print(len(point_grid))
distances = classifier.decision_function(point_grid)
abs_distances = [abs(x) for x in distances]  # list of 10000 numbers -> 100x100
distances_matrix = np.reshape(abs_distances, (100, 100))  # ->100 Listen mit jeweils hundert werten
#print(len(abs_distances[0]))
#print(len(abs_distances))
plt.show()
heatmap = plt.pcolormesh(x_values, y_values, distances_matrix)
plt.colorbar(heatmap)
plt.show()


"""XOR gate"""
labels = [0, 1, 1, 0]
plt.scatter([x[0] for x in data], [y[1] for y in data], c=labels)
# c -> color: points with label 1 sind andersferbig (hier gelb)
classifier = Perceptron(max_iter=40)
# default 1000 -> brauchen wir bei unseren leichten Daten aber nicht
classifier.fit(data, labels)
print(classifier.score(data, labels))
plt.show()

"""OR gate"""
labels = [0, 1, 1, 1]
plt.scatter([x[0] for x in data], [y[1] for y in data], c=labels)
# c -> color: points with label 1 sind andersferbig (hier gelb)
classifier = Perceptron(max_iter=40)
# default 1000 -> brauchen wir bei unseren leichten Daten aber nicht
classifier.fit(data, labels)
print(classifier.score(data, labels))
plt.show()
