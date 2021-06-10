import string

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Create your df here:
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

df = pd.read_csv("profiles.csv")
print(df.head())
# print(df.job.value_counts())
"""
plt.hist(df.age, bins=20)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.xlim(16, 80)
plt.show()
"""
# print(df.sign.value_counts())

"""since a lot of our data is categorial (strings), which is good for labels but not good for features"""

drink_mapping = {"not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5}

df["drinks_code"] = df.drinks.map(drink_mapping)

# print(df.smokes.value_counts())
smokes_mapping = {"no": 0, "sometimes": 1, "when drinking": 2, "yes": 3, "trying to quit": 4}

df["smokes_code"] = df.smokes.map(smokes_mapping)

# print(df.drugs.value_counts())
drugs_mapping = {"never": 0, "sometimes": 1, "often": 2}

df["drugs_code"] = df.drugs.map(drugs_mapping)

essay_cols = ["essay0", "essay1", "essay2", "essay3", "essay4", "essay5", "essay6", "essay7", "essay8", "essay9"]

# Removing the NaNs
all_essays = df[essay_cols].replace(np.nan, '', regex=True)
# Combining the essays
all_essays = all_essays[essay_cols].apply(lambda x: ' '.join(x), axis=1)


def cut_code(a_string):
    a_string = a_string.replace('\n', '')
    a_string = a_string.replace('<br />', '')
    return a_string


# print(cut_code(all_essays[0]))

all_essays = all_essays.apply(lambda x: cut_code(x))

df["essay_len"] = all_essays.apply(lambda x: len(x))


def word_length(long_string):
    if long_string.isspace():
        return 0
    string_list = long_string.split()  # splittet an jedem Leerzeichen
    y = len(long_string.split())
    z = 0
    for string in string_list:
        z += len(string)
    return z / y


df["avg_word_length"] = all_essays.apply(lambda x: word_length(x))


def frequency_I_me(long_string):
    long_string = long_string.lower()
    long_string = long_string.translate(
        str.maketrans(string.punctuation, ' ' * len(string.punctuation)))  # replaced punctuation mit whitespace
    sum = 0
    sum += long_string.count(" i ")
    sum += long_string.count(" me ")
    return sum


df["frequency"] = all_essays.apply(lambda x: frequency_I_me(x))

# print(df[df["avg_word_length"] == 0])

"""Jetzt muessen wir unsere Daten noch normalisieren (sonst verquere Ergebnisse)"""
"""
feature_data = df[['smokes_code', 'drinks_code', 'drugs_code']]

x = feature_data.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)

feature_data = pd.DataFrame(x_scaled, columns=feature_data.columns)


feature_data["sign"] = df[["sign"]]

# print(feature_data)
# print(labels)
feature_data = feature_data.dropna()
sign_dict = {"scorpio an it matters a lot": "scorpio", "gemini and it&rsquo;s fun to think about": "gemini",
             "scorpio and it&rsquo;s fun to think about": "scorpio",
             "leo and it&rsquo;s fun to think about": "leo",
             "taurus and it&rsquo;s fun to think about": "taurus",
             "libra and it&rsquo;s fun to think about": "libra",
             "virgo and it&rsquo;s fun to think about": "virgo",
             "virgo but it doesn&rsquo;t matter": "no matter",
             "cancer and it&rsquo;s fun to think about ": "cancer",
             "aries and it&rsquo;s fun to think about": "ares",
             "pisces and it&rsquo;s fun to think about": "pisces",
             "sagittarius and it&rsquo;s fun to think about": "sagittarius",
             "taurus but it doesn&rsquo;t matter": "no matter",
             "gemini but it doesn&rsquo;t matter": "no matter",
             "cancer but it doesn&rsquo;t matter ": "no matter",
             "leo but it doesn&rsquo;t matter ": "no matter",
             "libra but it doesn&rsquo;t matter ": "no matter",
             "aquarius but it doesn&rsquo;t matter": "no matter",
             "aquarius and it&rsquo;s fun to think about": "aquarius",
             "sagittarius but it doesn&rsquo;t matter": "no matter",
             "aries but it doesn&rsquo;t matter": "no matter",
             "capricorn but it doesn&rsquo;t matter": "no matter",
             "capricorn and it&rsquo;s fun to think about": "capricorn",
             "pisces but it doesn&rsquo;t matter": "no matter",
             "scorpio but it doesn&rsquo;t matter": "no matter"}
feature_data["sign"] = feature_data["sign"].replace(sign_dict)
#print(feature_data["sign"].value_counts())
labels = feature_data["sign"]
feature_data = feature_data.drop(["sign"], axis=1)

training_data, validation_data, training_labels, validation_labels = train_test_split(feature_data,
                                                                                      labels,
                                                                                      test_size=0.2, random_state=100)

"""
"""Prediciton der Zodiacs mit Knearestneighbour"""
"""
accuracies = []
for i in range(1, 11):
    classifier = KNeighborsClassifier(n_neighbors=i)
    classifier.fit(training_data, training_labels)
    accuracies.append(classifier.score(validation_data, validation_labels))

k_list = list(range(1, 11))
plt.ylabel("k-Neighbors")
plt.xlabel("Accuracy")
plt.plot(k_list, accuracies)
plt.show()

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(training_data, training_labels)

guesses = []
for i in range(len(validation_data.index)):
    guesses.append(classifier.predict([validation_data.iloc[i]]))

guesses = [x[0] for x in guesses]
validation_labels = list(validation_labels)
print(guesses)
print(validation_labels)
print(accuracy_score(validation_labels, guesses))
print(recall_score(validation_labels, guesses, average="macro"))
print(precision_score(validation_labels, guesses, average="macro"))
print(f1_score(validation_labels, guesses, average="macro"))
"""
"""
Predict sex
Problem DInge mit education vorauszusagen: education hat hier 9000 versch. Werte
Es waere also ein sehr umfangreiches Mapping noetig (aehnlich wie bei signs)
print(df["education"].value_counts())
"""
""" Prediction der Zodiacs mit der SVM, leicht hoehere Genauigkeit, aber sehr lange runtime.
Normalerweise wird die SVM aber auch dazu genutzt, zwischen 2 labels zu differenzieren. 
Dies ist hier nicht sinnvoll, allgemein ist die Nutzung von SVM hier vllt. nicht zu empfehlen
classifier = SVC(kernel="rbf", gamma=1)
classifier.fit(training_data, training_labels)
print(classifier.score(validation_data, validation_labels))

"""
"""Prediction education mit essay word counts"""
""" 
feature_data = df[["avg_word_length", "frequency", "essay_len"]]

x = feature_data.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x) #macht fit und transform in einem Schritt, fuer spaetere zus. Daten koennte man transform(X) nutzen

feature_data = pd.DataFrame(x_scaled, columns=feature_data.columns)

feature_data["education"] = df[["education"]]
feature_data = feature_data.dropna()
labels = feature_data["education"]
feature_data = feature_data.drop(["education"], axis=1)
#sinnvoll (aber auch aufwendig) waere es sicherlich, education hier auf ein simpleres level runterzubrechen, aktuell gibt es hier sehr viele versch. Werte, was die Auswertung ungenau machen wird

training_data, validation_data, training_labels, validation_labels = train_test_split(feature_data,
                                                                                      labels,
                                                                                      test_size=0.2, random_state=100)

#leicht besserer Classifier 0.44 zu 0.41 beim KNeighbors
classifier = SVC(kernel="rbf") #default gamma (C-like Parameter) = 'scale', 1 / (n_features * X.var())
classifier.fit(training_data, training_labels)
print(classifier.score(validation_data, validation_labels))



classifier = KNeighborsClassifier(n_neighbors=8)
classifier.fit(training_data, training_labels)
print(classifier.score(validation_data, validation_labels))
"""

"""Regression -  income with length of essays and average word length"""

"""
feature_data = df[["avg_word_length", "essay_len"]]

x = feature_data.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
feature_data = pd.DataFrame(x_scaled, columns=feature_data.columns)

feature_data["income"] = df[["income"]]
feature_data = feature_data[feature_data["income"] > 0]
feature_data = feature_data.dropna()
labels = feature_data["income"]
feature_data = feature_data.drop(["income"], axis=1)

x_train, x_test, y_train, y_test = train_test_split(feature_data, labels, train_size=0.8, test_size=0.2, random_state=6)

mlr = LinearRegression()
mlr.fit(x_train, y_train)
print(mlr.coef_)
y_predict = mlr.predict(x_test)
plt.scatter(y_test, y_predict, alpha=0.4)
plt.plot(range(1000000), range(1000000), c="red")
plt.xlabel("Actual Income")
plt.ylabel("Predicted Income")
plt.title("Actual vs. Predicts Income")
plt.show()
print(mlr.score(x_test, y_test))
#Kurz gesagt: funktioniert einfach mal ueberhaupt nicht, man koennte oben die vielen -1 Werte rausnehmen, das beingt zumindest etwas, aber gut ist immer noch weit entfernt.
print(df["income"].value_counts())
"""
""" Age mit frequency"""
"""
feature_data = df[["frequency", "avg_word_length", "essay_len", 'smokes_code', 'drinks_code', 'drugs_code', "income"]]

x = feature_data.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
feature_data = pd.DataFrame(x_scaled, columns=feature_data.columns)

feature_data["age"] = df[["age"]]
feature_data = feature_data.dropna()
labels = feature_data["age"]
feature_data = feature_data.drop(["age"], axis=1)

x_train, x_test, y_train, y_test = train_test_split(feature_data, labels, train_size=0.8, test_size=0.2, random_state=6)
mlr = LinearRegression()
mlr.fit(x_train, y_train)
print(mlr.coef_)
y_predict = mlr.predict(x_test)
plt.scatter(y_test, y_predict, alpha=0.4)
plt.plot(range(18, 70), range(18, 70), c="red")
plt.xlabel("Actual Age")
plt.ylabel("Predicted Age")
plt.title("Actual vs. Predicted Age")
plt.show()
print(mlr.score(x_test, y_test))
"""

"""sex mit height (categorial)"""
feature_data = df[["height"]]

x = feature_data.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
feature_data = pd.DataFrame(x_scaled, columns=feature_data.columns)

feature_data["sex"] = df[["sex"]]
feature_data = feature_data.dropna()
labels = feature_data["sex"]
feature_data = feature_data.drop(["sex"], axis=1)

training_data, validation_data, training_labels, validation_labels = train_test_split(feature_data, labels, train_size=0.8, test_size=0.2, random_state=6)


classifier = SVC(kernel="rbf") #default gamma (C-like Parameter) = 'scale', 1 / (n_features * X.var())
classifier.fit(training_data, training_labels)
print(classifier.score(validation_data, validation_labels))
#82% -> wenig erstaunlicherweise funktioniert das echt gut.