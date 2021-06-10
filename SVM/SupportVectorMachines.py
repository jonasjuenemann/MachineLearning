import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Support Vectoren -> Punkte am naechsten an der TrennLinie
red_support_vector = [1, 6]
blue_support_vector_one = [0.5, 2]
blue_support_vector_two = [2.5, 2]

# Margin Site -> Differenz zwischen Trennlinie und Punkt (auf y-Scala)
margin_size = 2

classifier = SVC(kernel='linear')
points = [[1, 2], [1, 5], [2, 2], [7, 5], [9, 4], [8, 2]]
labels = [1, 1, 1, 0, 0, 0]
classifier.fit(points, labels)
print(classifier.predict([[3, 2]]))
print(classifier.support_vectors_)

# Error Margin C -> Hoch: keine falsch eingeordneten Punkte, aber Margin Site wird sehr klein sein, (hard margin)
# :> Gefahr Overfitting: relies too heavily on training data, including Outliers
# -> Klein: eventl. ein paar falsch eingordnete Punkte, aber hohe Margin (soft Margin)
# :> Gefahr Underfitting: so much error allowed that the data isnt accurately represented

"""
points.extend([[3,3], [2,4], [10, 6]])
labels.extend([0, 1, 0])
classifier = SVC(kernel='linear', C = 0.5)
classifier.fit(points, labels)

draw_points(points, labels)
draw_margin(classifier)

plt.show()
"""

training_data, validation_data, training_labels, validation_labels = train_test_split(points, labels, train_size=0.8,
                                                                                      test_size=0.2, random_state=100)

classifier = SVC(kernel='linear')
classifier.fit(training_data, training_labels)
print(classifier.score(validation_data, validation_labels))

"""Wenn die Daten nicht auf verschiedenen des Plots liegen, sondern z.B. innen/außen ->
"poly" als kernel anstatt von linear"""

classifier = SVC(kernel='poly', degree=2)
classifier.fit(training_data, training_labels)
print(classifier.score(validation_data, validation_labels))

"Was passiert hier:"
classifier = SVC(kernel="linear", random_state=1)
classifier.fit(training_data, training_labels)
print(classifier.score(validation_data, validation_labels))

print(training_data[0])

"""Hier wird eine dritte Dimension zu den Punkten hinzugefuegt, was dazu fueht, dass sich die
zusammenhaengenden mittigen Punkten von den aeußeren in ihren Werten trennen"""
new_training = []
new_validation = []
for point in training_data:
    new_training.append([2 ** 0.5 * point[0] * point[-1], point[0] ** 2, point[-1] ** 2])
for point in validation_data:
    new_validation.append([2 ** 0.5 * point[0] * point[-1], point[0] ** 2, point[-1] ** 2])
classifier.fit(new_training, training_labels)
print(classifier.score(new_validation, validation_labels))

"""default ist der rbf Kernel der unendlich Dimensionen zu den Punkten hinzufuegt.
Dieser hat einen gamma-Wert, nicht unaehnlich dem C-Wert. 
Hier wuerde ebenso ein großer Wert eventl. zu Overfitting fuehren.
(Zu sehr an den trainingsdaten orientiert, um eventl. Outlier mit einzunehmen wird die Error Margin zu klein
Ein kleiner Wert koennte zu Underfitting fuehren"""

classifier = SVC(kernel="rbf", gamma=1)
classifier.fit(training_data, training_labels)
print(classifier.score(validation_data, validation_labels))

"""gamma zu erhoehen fuehrt zu hoeherer Ungenauigkeit wegen Overfitting,
ein niedriger Wert zu Ungenauigkeit wegen Underfitting"""