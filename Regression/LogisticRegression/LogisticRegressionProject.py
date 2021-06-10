import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

passengers = pd.read_csv("../../data/data_passengers.csv")
print(passengers.head())

# passengers['Sex'] = 1 wuerde alle Geschlechter auf 1 setzen

passengers['Sex'] = passengers['Sex'].map({'male': 0, 'female': 1})

print(passengers['Age'].values)  # gibt einem das ganze als Liste aus, passengers['Age'] als Spalte (mit index)

passengers = passengers.set_index(["PassengerId"])

# passengers = passengers.reset_index()

passengers['Age'].fillna(value=passengers['Age'].mean(), inplace=True)

passengers["FirstClass"] = passengers['Pclass'].map(lambda x: 1 if (x == 1) else 0)
# man koennte auch .apply anwenden, Unterschied ist, dass sich .map lediglich auf eine Series,
# .apply sich auf ein ganzen DataFrame anweden laesst
passengers["SecondClass"] = passengers['Pclass'].map(lambda x: 1 if (x == 2) else 0)

features = passengers[["Sex", "Age", "FirstClass", "SecondClass"]]
survival = passengers[["Survived"]]

scaler = StandardScaler()
features = scaler.fit_transform(features)

x_train, x_test, y_train, y_test = train_test_split(features, survival, test_size=0.2)

# Fuer die Logistische Regression benutzt Regularization, braucht daher scaled Daten
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = LogisticRegression()
# will find coefficients with the lowest log_loss
model.fit(x_train, y_train)
print("Ueberleben laut Modell:")
print(model.predict(x_train))
print("Tatsaechliches Ueberleben:")
print(y_train)

print("Train score:")
print(model.score(x_train, y_train))
print("Test score:")
print(model.score(x_test, y_test))
print("Feature Coefficients:")
print(model.coef_)

Jack = np.array([0.0, 20.0, 0.0, 0.0])
Rose = np.array([1.0, 17.0, 1.0, 0.0])
Jonas = np.array([0.0, 25.0, 1.0, 0.0])

new_passengers = np.array([Jack, Rose, Jonas])
print(new_passengers)

new_passengers = scaler.transform(new_passengers)
print(new_passengers)
print(model.predict(new_passengers))
# Column1 -> WK zu sterben, Col2 -> Ueberleben
print(model.predict_proba(new_passengers))