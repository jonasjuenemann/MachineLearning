"""
regression:



Lineare Regression:
from sklearn.linear_model import LinearRegression

your_model = LinearRegression()

your_model.fit(x_training_data, y_training_data) #.coef = Koeffizienten, .intercept: Y_achsenWert bei x (und allen anderen Einflussvariablen z, ...) = 0

predictions = your_model.predict(your_x_data)

__________________________________________________________
classification:



Naive Bayes:
from sklearn.naive_bayes import MultinomialNB

your_model = MultinomialNB()

your_model.fit(x_training_data, y_training_data)

predictions = your_model.predict(your_x_data) #retured eine Liste mit geschätzen Klassen fuer jeden Datapoint

probabilities = your_model.predict_proba(your_x_data) #returened eine Liste mit der Wahrscheinlichkeit fuer jede Klasse (vermutlich: waehlt default die höchste WKs Klasse fuer predictions aus)

from sklearn.neigbors import KNeighborsClassifier

your_model = KNeighborsClassifier()



K-Nearest Neighbors
from sklearn.neigbors import KNeighborsClassifier

your_model = KNeighborsClassifier()

your_model.fit(x_training_data, y_training_data)

predictions = your_model.predict(your_x_data)

probabilities = your_model.predict_proba(your_x_data)



K-Means
from sklearn.cluster import KMeans

your_model = KMeans(n_clusters=4, init='random') # n_clusters: number of Clusters to form, init : k-means++[default] oder random[K-Means], random_state: seed used by the random number generator [opt.]

your_model.fit(x_training_data)

predictions = your_model.predict(your_x_data)

___________________________________________________________

Validating the Model

accuracy, recall, precision, and F1 score:
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

print(accuracy_score(true_labels, guesses))
print(recall_score(true_labels, guesses))
print(precision_score(true_labels, guesses))
print(f1_score(true_labels, guesses))

the confusion matrix:
from sklearn.metrics import confusion_matrix

print(confusion_matrix(true_labels, guesses))

____________________________________________________________

Training Sets and Test Sets:
Zum Testen der Effektivität eines ML Algorithmus kann man aus predefinierten Datenpunkten (die ueblicherweise ja vorhanden sind, nehmen:
Idee ist: wir benutzen die Daten die wir schon haben (vergangenheit), geben aber vor diese nicht zu kennen (geben unserem Algo nicht)
und ueberpruefen anhand dieser, wie akkurat der Algo arbeitet
- training set (80%)
- validation set
- test set (20%) -> Am Ende, Daten die vorher nicht verwendet wurden, substitut fuer Real-World Daten um zu gucken,
wo accuracy, precision, recall, and F1 score (kommt noch) liegen.



from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2)

#train_size -> proportion of the dataset to include in the train split
#test_size -> proportion of the dataset to include in the test split
#random_state -> seed used by the random number generator [optional]
"""