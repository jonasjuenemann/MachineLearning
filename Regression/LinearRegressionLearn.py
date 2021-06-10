import math
from scipy.spatial import distance
import seaborn
import matplotlib.pyplot as plt

"""distance Bestimmung:"""

# multi-dimensional points:
two_d = [10, 2]
five_d = [30, -1, 50, 0, 2]
four_d = [7, 5, 3, 4]


# to find the difference between two points, htey need to have the same number of dimensions
# euklidian distance: sqrt((a1 - b1)² + (a2 - b2)² + ...)
def euclidean_distance(pt1, pt2):
    distance = 0
    for i in range(len(pt1)):
        distance += (pt1[i] - pt2[i]) ** 2
    distance = math.sqrt(distance)
    return distance


print(euclidean_distance([5, 1], [1, 5]))


def manhattan_distance(pt1, pt2):
    distance = 0
    for i in range(len(pt1)):
        distance += abs(pt1[i] - pt2[i])
    return distance


print(manhattan_distance([5, 1], [1, 5]))


def hamming_distance(pt1, pt2):
    distance = 0
    for i in range(len(pt1)):
        if not (pt1[i] == pt2[i]):
            distance += 1
    return distance


print(hamming_distance([1, 2], [1, 100]))
print(hamming_distance([5, 4, 9], [1, 7, 9]))

# Funktionen in SciPy implementiert:

print(distance.euclidean([1, 2], [4, 0]))
print(distance.cityblock([1, 2], [4, 0]))
print(distance.hamming([5, 4, 9], [1, 7, 9]))
# hamming funktioniert etwas anders, gibt Intervall von 0 bis 1 an, 1 -> alle unterschiedlich, 0 -> keins unterschiedlich


"""Lineare Regression:"""

months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
revenue = [52, 74, 79, 95, 115, 110, 129, 126, 147, 146, 156, 184]

# slope:
m = 12
# intercept:
b = 40

y = [month * m + b for month in months]

plt.plot(months, revenue, "o")

plt.plot(months, y)

plt.title("Sandra's Lemonade Stand")

plt.xlabel("Months")
plt.ylabel("Revenue ($)")

plt.show()

"""Total Loss -> Art des Vergleichs zweier Modelle durch Quadrierung der Abweichungen"""

x = [1, 2, 3]
y = [5, 1, 3]

# y = x
m1 = 1
b1 = 0

# y = 0.5x + 1
m2 = 0.5
b2 = 1

y_predicted1 = [m1 * x + b1 for x in x]
y_predicted2 = [m2 * x + b2 for x in x]
print(y_predicted1)
print(y_predicted2)

total_loss1 = 0
for i in range(len(y)):
    total_loss1 += (y[i] - y_predicted1[i]) ** 2
print(total_loss1)

total_loss2 = 0
for i in range(len(y)):
    total_loss2 += (y[i] - y_predicted2[i]) ** 2
print(total_loss2)


# Gradient Descent for Intercept, bestimmt den "Fehler-Wert" fuer einen bestimmten Intercept b

def get_gradient_at_b(x, y, b, m):
    diff = 0
    N = len(x)
    for i in range(N):
        diff += (y[i] - (m * x[i] + b))
    b_gradient = (-2 / N) * diff
    return b_gradient


# Gradient Descent for Slope

def get_gradient_at_m(x, y, b, m):
    diff = 0
    N = len(x)
    for i in range(N):
        diff += x[i] * (y[i] - (m * x[i] + b))
    m_gradient = (-2 / N) * diff
    return m_gradient


def step_gradient(b_current, m_current, x, y, learning_rate):
    b_gradient = get_gradient_at_b(x, y, b_current, m_current)
    m_gradient = get_gradient_at_m(x, y, b_current, m_current)
    b = b_current - (learning_rate * b_gradient)
    m = m_current - (learning_rate * m_gradient)
    return [b, m]


def gradient_descent(x, y, learning_rate, num_iterations):
    b = 0
    m = 0
    for i in range(num_iterations):
        b, m = step_gradient(b, m, x, y, learning_rate)
    return [b, m]


"""Benutzung der Functions ueber die LinearRegressionFunctions-Klasse, da sonst alle Prozesse hier auch in der anderen Klasse ausgefuehrt werden"""


b, m = gradient_descent(months, revenue, 0.001, 10000)

print(b, m)

y = [month * m + b for month in months]

plt.plot(months, revenue, "o")

plt.plot(months, y)

plt.title("Sandra's Lemonade Stand")

plt.xlabel("Months")
plt.ylabel("Revenue ($)")

plt.show()

# current intercept guess:
b = 0
# current slope guess:
m = 0
