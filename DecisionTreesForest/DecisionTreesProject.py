import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

flags = pd.read_csv("../data/flags.csv", header=0)

# print(flags.columns)
print(flags.head())

labels = flags[["Landmass"]]
"""Diese Daten sind jeweils 0 bzw. 1 falls Farbe vorkommt."""
data = flags[["Red", "Green", "Blue", "Gold", "White", "Black", "Orange"]]
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state=1)
tree = DecisionTreeClassifier(random_state=1)
tree.fit(train_data, train_labels)
print(tree.score(test_data, test_labels))
print("Predcition fuer eine Flagge mit [1,0,0,1,0,1,0] -> Also Schwarz,Rot,Gold, Ergebnis wird auch numerisch nach dem" +
      " Schema 1=N.America, 2=S.America, 3=Europe, 4=Africa, 4=Asia, 6=Oceania erfolgen")
print((lambda x : "Europe" if x == 3 else "N.America" if x == 1 else "Oceania")(tree.predict([[1,0,0,1,0,1,0]])))
"""Das koennte man jetzt so erweitern, dass alle zugeordnet werden, spar ich mir hier.
Alternativ koennte man das ganze natuerlich auch in eine externe Funktion ausgliedern"""

scores = []
for i in range(20, 0, -1):
    tree = DecisionTreeClassifier(random_state=1, max_depth=i)
    tree.fit(train_data, train_labels)
    scores.append(tree.score(test_data, test_labels))
    # print("Score with max_depth {}".format(i))
    # print(tree.score(test_data, test_labels))

plt.title("Base: Colors to Continent")
plt.plot(range(20, 0, -1), scores)
plt.xlabel("Max-Depth")
plt.ylabel("Score")
plt.show()

data = flags[["Red", "Green", "Blue", "Gold",
 "White", "Black", "Orange",
 "Circles",
"Crosses","Saltires","Quarters","Sunstars",
"Crescent","Triangle"]]
# labels = flags[["Landmass"]] ->unveraendert
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state=1)
tree = DecisionTreeClassifier(random_state=1)
tree.fit(train_data, train_labels)
print(tree.score(test_data, test_labels))

scores = []
for i in range(1, 21):
    tree = DecisionTreeClassifier(random_state=1, max_depth=i)
    tree.fit(train_data, train_labels)
    scores.append(tree.score(test_data, test_labels))
    # print("Score with max_depth {}".format(i))
    # print(tree.score(test_data, test_labels))
plt.title("Advanced: Colors and others to Continent")
plt.plot(range(1, 21), scores)
plt.xlabel("Max-Depth")
plt.ylabel("Score")
plt.show()

"""If the tree is too short, we’re underfitting and not accurately representing the training data.
If the tree is too big, we’re getting too specific and relying too heavily on the training data."""

labels = flags[["Language"]]
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state=1)
tree = DecisionTreeClassifier(random_state=1)
tree.fit(train_data, train_labels)
print(tree.score(test_data, test_labels))

scores = []
for i in range(1, 21):
    tree = DecisionTreeClassifier(random_state=1, max_depth=i)
    tree.fit(train_data, train_labels)
    scores.append(tree.score(test_data, test_labels))
    # print("Score with max_depth {}".format(i))
    # print(tree.score(test_data, test_labels))

plt.title("Base: Colors and Others to Language")
plt.plot(range(1, 21), scores)
plt.xlabel("Max-Depth")
plt.ylabel("Score")
plt.show()


data = flags[["Red", "Green", "Blue", "Gold",
 "White", "Black", "Orange",
 "Circles",
"Crosses","Saltires","Quarters","Sunstars",
"Crescent","Triangle"]]
labels = flags[["Landmass"]]
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state=1)
tree = DecisionTreeClassifier(random_state=1)
tree.fit(train_data, train_labels)
print(tree.score(test_data, test_labels))

scores = []
for i in range(2, 101):
    tree = DecisionTreeClassifier(random_state=1, max_leaf_nodes=i)
    tree.fit(train_data, train_labels)
    scores.append(tree.score(test_data, test_labels))
    # print("Score with max_depth {}".format(i))
    # print(tree.score(test_data, test_labels))
plt.title("Advanced: Colors and others to Continent")
plt.plot(range(2, 101), scores)
plt.xlabel("Max-Leave-Nodes")
plt.ylabel("Score")
plt.show()