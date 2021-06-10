import math

star_wars = [125, 1977, 11000000]
raiders = [115, 1981, 18000000]
mean_girls = [97, 2004, 17000000]


def distance(lst1, lst2):
    distance = 0
    for i in range(len(lst1)):
        distance += (lst1[i] - lst2[i]) ** 2
    distance = math.sqrt(distance)
    return distance


print(distance(star_wars, raiders))
print(distance(star_wars, mean_girls))

"""Normalization um die Dinge wie hier, dass das Budget viel zu viel ausmacht, zu verhindern
"""

release_dates = [1897, 1998, 2000, 1948, 1962, 1950, 1975, 1960, 2017, 1937, 1968, 1996, 1944, 1891, 1995, 1948, 2011,
                 1965, 1891, 1978]


def min_max_normalize(lst):
    minimum = min(lst)
    maximum = max(lst)
    normalized = []
    for val in lst:
        normalized.append((val - minimum) / (maximum - minimum))
    return normalized


print(min_max_normalize(release_dates))


def normalize_index(i, *lsts):
    helplist = []
    for lst in lsts:
        helplist.append(lst[i])
    helplist = min_max_normalize(helplist)
    for x in range(len(lsts)):
        lsts[x][i] = helplist[x]


normalize_index(1, star_wars, raiders, mean_girls)
print(star_wars, raiders, mean_girls)

for i in range(len(star_wars)):
    normalize_index(i, star_wars, raiders, mean_girls)

normalize_index(1, star_wars, raiders, mean_girls)
print(star_wars, raiders, mean_girls)

# bestimme k naechste nachbarn
from data.data_movies import movie_dataset, movie_labels

print(movie_dataset['Bruce Almighty'])
print(movie_labels['Bruce Almighty'])


def classify(unknown, dataset, labels, k):
    distances = []
    for title in dataset:
        distance_to_point = distance(dataset[title], unknown)
        distances.append([distance_to_point, title])
    distances.sort()
    neighbors = distances[:k]
    num_good = 0
    num_bad = 0
    for movie in neighbors:
        title = movie[1]
        if labels[title] == 1:
            num_good += 1
        else:
            num_bad += 1
    if num_good > num_bad:
        return 1
    else:
        return 0


print(classify([.4, .2, .9], movie_dataset, movie_labels, 5))

print("Call Me By Your Name" in movie_dataset)
my_movie = [2.8634276635608227e-05, 0.3242320819112628, 1.0112359550561798]  # [350000, 132, 2017]
# movie_dataset["Call Me By Your Name"] = [2.8634276635608227e-05, 0.3242320819112628, 1.0112359550561798] so wuerden wir den Film ins Datenset eingliedern,
# das wollen wir aber an dieser Stelle noch gar nicht! Erstmal den Film klassifizieren
print(movie_dataset.values())  # Da in dieser Liste die Punkte schon genormalized sind,
# ist, ohne die urspruenglichen Punkte zu kennnen, eine weitere Normalisierung von my_movie manuell schwierig
print(classify(my_movie, movie_dataset, movie_labels, 5))

# how to build training/validation sets from dictionaries, gleiche Funktion wie in Linear Regression: funktioniert nicht!, eventuell einfach manuell machen sonst
"""
print(validation_set["Bee Movie"])
print(validation_labels["Bee Movie"])

guess = classify(validation_set["Bee Movie"], training_set, training_labels, 5)
print(guess)
"""
print(movie_dataset["Bee Movie"])
print(movie_labels["Bee Movie"])

guess1 = classify(movie_dataset["Bee Movie"], movie_dataset, movie_labels, 5)
print("classify Bee Movie")
print(guess1)


def find_validation_accuracy(training_set, training_labels, validation_set, validation_labels, k):
    num_correct = 0.0
    for movie in validation_set:
        guess = classify(validation_set[movie], training_set, training_labels, k)
        if validation_labels[movie] == guess:
            num_correct += 1
    return num_correct / len(validation_labels)


"""Anstatt das k-Nearest-Neighbors Modell selbst zu programmieren koennen wir auch hier natuerlich wieder sklearn benutzen"""

from sklearn.neighbors import KNeighborsClassifier

movie_dataset_values = list(movie_dataset.values())
movie_labels_values = list(movie_labels.values())
print(movie_dataset)

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(movie_dataset_values, movie_labels_values)

print(classifier.predict([[.45, .2, .5], [.25, .8, .9], [.1, .1, .9]]))

from data.data_movies import movie_ratings


def predict_w_Regressor(unknown, dataset, movie_ratings, k):
    distances = []
    # Looping through all points in the dataset
    for title in dataset:
        movie = dataset[title]
        distance_to_point = distance(movie, unknown)
        # Adding the distance and point associated with that distance
        distances.append([distance_to_point, title])
    distances.sort()
    # Taking only the k closest points
    neighbors = distances[0:k]
    rating_sum = 0
    for neighbor in neighbors:
        rating_sum += movie_ratings[neighbor[1]]
    return float("{:.2f}".format(rating_sum / len(neighbors)))


print(movie_ratings["Life of Pi"])
print("predict_w_Regressor nicht geweighted")
print(predict_w_Regressor([0.016, 0.300, 1.022], movie_dataset, movie_ratings, 5))


def predict_w_Regressor_weighted(unknown, dataset, movie_ratings, k):
    distances = []
    # Looping through all points in the dataset
    for title in dataset:
        movie = dataset[title]
        distance_to_point = distance(movie, unknown)
        # Adding the distance and point associated with that distance
        distances.append([distance_to_point, title])
    distances.sort()
    # Taking only the k closest points
    neighbors = distances[0:k]
    numerator = 0
    denominator = 0
    for neighbor in neighbors:
        numerator += movie_ratings[neighbor[1]] / neighbor[0]  # rating/distance
        denominator += 1 / neighbor[0]  # 1/distance
    return float("{:.2f}".format(
        numerator / denominator))  # (rating/distance)/(1/distance) -> je kleiner distance -> desto mehr ist der Wert wert

print("predict_w_Regressor_weighted")
print(predict_w_Regressor_weighted([0.016, 0.300, 1.022], movie_dataset, movie_ratings, 5))

# Auch hier wieder: mit sklearn:
from sklearn.neighbors import KNeighborsRegressor
movie_ratings_values = list(movie_ratings.values())

#default ist weights="uniform" -> kein eingebautes weighting, also alle gleich
regressor = KNeighborsRegressor(n_neighbors=5, weights="distance")
regressor.fit(movie_dataset_values, movie_ratings_values)
print(regressor.predict([[0.016, 0.300, 1.022], [0.0004092981, 0.283, 1.0112], [0.00687649, 0.235, 1.0112]]))