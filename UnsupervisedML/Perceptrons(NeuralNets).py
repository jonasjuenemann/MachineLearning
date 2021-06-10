class Perceptron:
    def __init__(self, num_inputs=2, weights=None):
        if weights is None:
            weights = [2, 1]
        self.num_inputs = num_inputs
        self.weights = weights

    def weighted_sum(self, inputs):
        # create variable to store weighted sum
        weighted_sum = 0
        for i in range(self.num_inputs):
            weighted_sum += inputs[i] * self.weights[i]
            # complete this loop
        return weighted_sum

    def activation(self, weighted_sum):
        # Complete this method
        if weighted_sum >= 0:
            return 1
        return -1

    """Wichtig -> Das trainings_Set ist hier ein dictionary, dass neben dem input, auch sein entsprechendes label hat"""

    def training(self, training_set):
        foundLine = False
        while not foundLine:
            total_error = 0
            for inputs in training_set:
                prediction = self.activation(self.weighted_sum(inputs))
                actual = training_set[inputs]
                error = actual - prediction
                total_error += abs(error)
                for i in range(self.num_inputs):
                    self.weights[i] += error * inputs[i]
            if total_error == 0:
                foundLine = True


cool_perceptron = Perceptron()
small_training_set = {(0, 3): 1, (3, 0): -1, (0, -3): -1, (-3, 0): 1}
print(cool_perceptron.weighted_sum([24, 55]))
print(cool_perceptron.activation(55))
print(cool_perceptron.weights)
print(cool_perceptron.activation(cool_perceptron.weighted_sum(list(small_training_set)[0])))
print(cool_perceptron.activation(cool_perceptron.weighted_sum(list(small_training_set)[1])))
print(cool_perceptron.activation(cool_perceptron.weighted_sum(list(small_training_set)[2])))
print(cool_perceptron.activation(cool_perceptron.weighted_sum(list(small_training_set)[3])))
cool_perceptron.training(small_training_set)
print(cool_perceptron.weights)
print(cool_perceptron.activation(cool_perceptron.weighted_sum(list(small_training_set)[0])))
print(cool_perceptron.activation(cool_perceptron.weighted_sum(list(small_training_set)[1])))
print(cool_perceptron.activation(cool_perceptron.weighted_sum(list(small_training_set)[2])))
print(cool_perceptron.activation(cool_perceptron.weighted_sum(list(small_training_set)[3])))



""""""
import matplotlib.pyplot as plt
import random


def generate_training_set(num_points):
    x_coordinates = [random.randint(0, 50) for i in range(num_points)]
    y_coordinates = [random.randint(0, 50) for i in range(num_points)]
    training_set = dict()
    for x, y in zip(x_coordinates, y_coordinates):
        if x <= 45 - y:
            training_set[(x, y)] = 1
        elif x > 45 - y:
            training_set[(x, y)] = -1
    return training_set


training_set = generate_training_set(30)

x_plus = []
y_plus = []
x_minus = []
y_minus = []

for data in training_set:
    if training_set[data] == 1:
        x_plus.append(data[0])
        y_plus.append(data[1])
    elif training_set[data] == -1:
        x_minus.append(data[0])
        y_minus.append(data[1])

fig = plt.figure()
ax = plt.axes(xlim=(-25, 75), ylim=(-25, 75))

plt.scatter(x_plus, y_plus, marker='+', c='green', s=128, linewidth=2)
plt.scatter(x_minus, y_minus, marker='_', c='red', s=128, linewidth=2)

plt.title("Training Set")

plt.show()
