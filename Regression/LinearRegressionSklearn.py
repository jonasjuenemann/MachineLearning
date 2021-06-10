import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.datasets import load_boston

df = pd.read_csv("../data/data_heights.csv")

X = df["height"]
y = df["weight"]

plt.plot(X, y, 'o')

# b, m = gradient_descent(X, y, 0.0001, 1000) #funktioniert, braucht aber
# y_predictions = [x * m + b for x in X]

# plt.plot(X, y_predictions)

plt.show()

temperature = np.array(range(60, 100, 2))
temperature = temperature.reshape(-1, 1) # 2-D Array
sales = [65, 58, 46, 45, 44, 42, 40, 40, 36, 38, 38, 28, 30, 22, 27, 25, 25, 20, 15, 5]

line_fitter = LinearRegression()
line_fitter.fit(temperature, sales)

sales_predict = line_fitter.predict(temperature)
sales_101_predict = line_fitter.predict([[101]])
# [line_fitter[0] + line_fitter[1]*temp for temp in temperature], funktioniert nicht 'LinearRegression() does not support indexing'
#print(sales_101_predict)
print(sales_predict)
#sales_predict = np.append(sales_predict, sales_101_predict)
#print(sales_predict)
plt.plot(temperature, sales, 'o')
plt.plot(101, sales_101_predict, 'o', color="red")
#temperature = np.append(temperature, [101])
#print(temperature)
plt.plot(temperature, sales_predict)

plt.xlabel("Temperature")
plt.ylabel("Sales")

plt.show()

# Boston housing dataset
boston = load_boston()

df = pd.DataFrame(boston.data, columns=boston.feature_names)

# Set the x-values to the nitrogen oxide concentration:
X = df[['NOX']] # erzeugt ein zwei-D Array, was fuer die fit() Funktion wichtig ist, da hier jedes Element als einzelne Liste genommen wird (ermoeglicht auch mehrere x Parameter -> s. Mult. LR)
# Y-values are the prices:
y = boston.target

# Can we do linear regression on this?

line_fitter = LinearRegression()
line_fitter.fit(X, y)

prices_predict = line_fitter.predict(X)

plt.scatter(X, y, alpha=0.4) #alpha -> eine Art Opacity
# Plot line here:
plt.plot(X, prices_predict)
plt.title("Boston Housing Dataset")
plt.xlabel("Nitric Oxides Concentration")
plt.ylabel("House Price ($)")
plt.show()
