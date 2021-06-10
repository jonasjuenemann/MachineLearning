import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from svm_visualization import draw_boundary
from players import aaron_judge, jose_altuve, david_ortiz

fig, ax = plt.subplots()
print(aaron_judge.head())
print(aaron_judge.columns)
print(aaron_judge.description.unique())  # different valuesof the description feature
print(aaron_judge.type.unique())
aj = pd.DataFrame(aaron_judge)
aj['type'] = aj['type'].map({'S': 1, 'B': 0})
print(aj['type'])
print(aj["plate_x"])
aj = aj[["plate_x", "plate_z", "type"]]
aj = aj.dropna()  # man koennte hier auch subset = ['A','B','c']
print(aj)

plt.scatter(aj["plate_x"], aj["plate_z"], c=aj['type'], cmap=plt.cm.coolwarm, alpha=0.25)

training_set, validation_set = train_test_split(aj, random_state=1)
for i in range(1, 11):
    for x in range(1, 11):
        classifier = SVC(kernel="rbf", gamma=i, C=x)
        classifier.fit(training_set[["plate_x", "plate_z"]], training_set["type"])
        draw_boundary(ax, classifier)
        print("Score for gamma {} and C {}".format(x, i))
        print(classifier.score(validation_set[["plate_x", "plate_z"]], validation_set["type"]))
        plt.show()
