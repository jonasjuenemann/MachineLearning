import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

# Da unser age mal wieder den markwurdigen ï»¿ ASCII-Bug hat, den es manchmal gibt wenn man csv aus excel exportiert -> encoding='utf-8-sig'
income_data = pd.read_csv("../data/data_income.csv", header=0, delimiter=", ", encoding='utf-8-sig')
# income_data["sex"] = income_data["sex"].map({'Male': 0, 'Female': 1})
# in unserer csv ist ein space hinter jedem Komma, was uns hier nerft
"""
income_data.columns = income_data.columns.str.replace(' ', '') #funktioniert
income_data["workclass"] = income_data["workclass"].str.strip() 
# funktioniert auch, muesste man jetzt fuer jede relevante Column machen
sehr viel einfacher -> delimiter beim importieren
"""
print(income_data.head())
print(income_data.iloc[0])

labels = income_data["income"] # Als Liste in Liste -> immernoch ein dataframe, als reine Liste -> pandasSeries
# es ist auch moeglich das hier als DataFrame and sklearn zu uebergeben, gibt aber eine Warnung.
data = income_data[["age", "capital-gain", "capital-loss", "hours-per-week"]]
# man koennte hier auch noch sex benutzen, hier muesste man aber die Werte umwandeln
#print(labels.tolist()) -> geht nur wenn labels eine PandaSeries ist.
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state=1)

forest = RandomForestClassifier(random_state=1)
forest.fit(train_data, train_labels)
print("Predictions of a forest")
print(forest.score(test_data, test_labels))

data = income_data[["age", "capital-gain", "capital-loss", "hours-per-week", "sex"]]
data["sex"] = data["sex"].map({'Male': 0, 'Female': 1}) # gibt eine etwas mysteriöse Warnungt,
# das man einen value auf der Kopie eines Slice eines DataFrames setzen will.
# funktioniert aber trotzdem. (Die Warnung ist tatseachlich eig. fuer einen anderen Use-Case gedacht, s. PandasTest
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state=1)
forest = RandomForestClassifier(random_state=1)
forest.fit(train_data, train_labels)
print("Predictions of a forest")
print(forest.score(test_data, test_labels))
#print(income_data["native-country"].value_counts())
data = income_data[["age", "capital-gain", "capital-loss", "hours-per-week", "sex", "native-country"]]
pd.options.mode.chained_assignment = None # Da mir die Warnung auf den Geist geht.
data["sex"] = data["sex"].map({'Male': 0, 'Female': 1})
data["native-country"] = data["native-country"].map(lambda x: 0 if (x == "United-States") else 1)
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state=1)
forest = RandomForestClassifier(random_state=1)
forest.fit(train_data, train_labels)
print("Predictions of a forest")
print(forest.score(test_data, test_labels))
print("Feature Importances")
print(forest.feature_importances_)

classifier = tree.DecisionTreeClassifier()
classifier.fit(test_data, test_labels)
print("Score eines Baums")
print(classifier.score(test_data, test_labels))