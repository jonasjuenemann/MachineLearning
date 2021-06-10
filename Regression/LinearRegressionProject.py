
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

df = pd.read_csv("../data/data_honeyproduction.csv")

print(df.head())

prod_per_year = df.groupby("year").totalprod.mean().reset_index()  # gibt das Jahr und dessen Durchschnitt an totalprod an
# das reset_index() -> alter index ist as column geadded (wenn man das nicht will, koennte man den drop Parameter verwenden)
# ist noetig, damit man im naechsten Schritt die Column per key auswaehlen kann


print(prod_per_year)
X = prod_per_year["year"]
X = X.values.reshape(-1, 1) # fuegt die Liste mit Indizes zu einem Array of arrays zusammen, s.u. links ndarray[[..],[..], ..], das wir zum plotten brauchen
print(X)
y = prod_per_year["totalprod"]
print(y)

plt.scatter(X, y)

regr = linear_model.LinearRegression()
regr.fit(X, y)  # learning_rate und num_iterations sind default von sklearn
m = regr.coef_
print(m)
b = regr.intercept_
print(b)
#StartWert ist also: 181208083 + 1998 * -88303 -> 4778689

prod_predict = regr.predict(X) # predicte Werte von y auf X, wir erhalten also eine Reihe (Array von y Werten)

plt.plot(X, prod_predict)

plt.show()

X_future = np.array(range(2013, 2051))
X_future = X_future.reshape(-1, 1)

plt.scatter(X, y)
plt.plot(X, prod_predict)
future_predict = regr.predict(X_future)

print(future_predict[-1])

plt.plot(X_future, future_predict)


plt.show()