import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

streeteasy = pd.read_csv("../data/data_manhattan.csv")

df = pd.DataFrame(streeteasy)
print(df.head())

df_park = df.loc[df["neighborhood"] == "Battery Park City"]
# erstellt eine neue Tabelle, in der nur noch diese Eintraege vorhanen sind

# df_park = df_park.T -> Transpose Methode, dreht die Tabelle einmal um: Row indices -> column values

x = df[['bedrooms', 'bathrooms', 'size_sqft', 'min_to_subway', 'floor', 'building_age_yrs', 'no_fee',
        'has_roofdeck', 'has_washer_dryer', 'has_doorman', 'has_elevator', 'has_dishwasher', 'has_patio', 'has_gym']]
print(x.head())
y = df[['rent']]
# unterschied zwischen List of List [[]] and List [] ist, dass hier "rent" der Spaltenname bleibt,
# waehrend sonst lediglich die values aus rent uebernommen werden (normale key anfrage),
# So erschaffen ein 2-D Array, hier koennte das tatsaechlich egal sein, da wir das fuer y benutzen, an dem wir plotten, bei X ist dies relevant
print(y.head())

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=6)
print(x_train.shape)  # gibt einfach die Groeße der Tabelle rows*columns wieder, liegt wie erwartet bei (80:20)
print(x_test.shape)

print(y_train.shape)
print(y_test.shape)

# Hier wie gehabt, dass wir jetzt mehr als ein x als Faktor aendert erstmal wenig

mlr = LinearRegression()

mlr.fit(x_train, y_train)
print("Koeffizienten")
print(mlr.coef_)

y_predict = mlr.predict(x_test) # Wichtig: Wir nehmen hier die test Werte, da wir ja y_predict auch mit y_test messen wollen

# sonny_apartment = [[1, 1, 620, 16, 1, 98, 1, 0, 1, 0, 0, 1, 1, 0]]

predict = mlr.predict(x_test[:1])
# print(x_test[:1])  # x_test[0] oder x_test[[0]] wuerde nicht funktionieren, da er hier nach den column names sucht
# Alternativ:
# print(x_test.iloc[[0]])  # index 0 aus DataFrame Sicht
# print(x_test.loc[[631]])  # index 631 aufgrund Bennenung

print("Predicted rent: $%.2f" % predict)

# Create a scatter plot
plt.scatter(y_test, y_predict, alpha=0.4)
#
# # Create x-axis label and y-axis label
# plt.xlabel("Actual Prices")
# plt.ylabel("Predicted Prices")
#
# # Create a title
# plt.title("Actual vs. Predicts Prices")
#
# # Show the plot
plt.show()

# Außerdem: Um Zusammenhaenge in den Datem zu zeigen, kann man verschiedene einzelne Parameter gegen den y abzeichnen:
plt.scatter(df[['size_sqft']], df[['rent']], alpha=0.4)
plt.xlabel("size_sqft")
plt.ylabel("rent")

plt.show()

plt.scatter(df[['building_age_yrs']], df[['rent']], alpha=0.4)
plt.xlabel("building_age_yrs")
plt.ylabel("rent")

plt.show()

plt.scatter(df[['min_to_subway']], df[['rent']], alpha=0.4)
plt.xlabel("min_to_subway")
plt.ylabel("rent")

plt.show()

print("Train score:")
print(mlr.score(x_train, y_train)) # gibt den R² Wert, (s.DA Regression) -> misst durchschnittliche Abweichung y_predict von y, um die Varianz von y ergeanzt
# 0.7 oder mehr ist generell gut -> - Werte sehr schlecht

print("Test score:")
print(mlr.score(x_test, y_test))

#man koennte jetzt Columns mit schwachen Koeffizienten aus dem Modell werfen, bspw. patio, um das Modell exakter zu machen

# pd.options.display.max_columns = 60 stellt number of row/col shown ein
# pd.options.display.max_colwidth = 500

"""
Load Data

Um index bzw. column Namen als Zahl bzw. Liste zu bekommen
print(len(businesses.index))
print(list(reviews.columns))

users.describe() #gibt allg. Informationen ueber die Tabelle

for col in users.columns[1:]:  #Wenn das ganze fuer die erste Tabellenspalte (nicht die Indizes!, die zaehlen in pandas nicht als column, sondern z.B. ne id) 
    print(users[col].min()) # gibt minimum der Spalte an
    print(users[col].max()) # gibt maximum der Spalte an

Merge Data

df = pd.merge(businesses, reviews, how='left', on='business_id') fuer merges von zwei Tables (inner, left, right, outer)

Cleaning Data

df.drop(features_to_remove, axis=1, inplace=True) , axis=1 -> columns, axis=0 -> rows, features_to_remove -> ["...", "..."] column names zum removen,
inplace -> soll df abgeändert werden, oder ein neuer dataframe gebildet werden

df.isna().any() -> check for missing values bzw. NaN, checkt fuer jede column einzeln

df.fillna({'column_1':val_to_replace_na,
           'column_2':val_to_replace_na,
           'column_3':val_to_replace_na},
          inplace=True)
          -> checkt fur diese columns die NaN und ersetzt sie mit entsprechendem calue (bspw. 0)

Exploring Data

df.corr() -> schaut sich alle Koeffizienten zueinander in einer Matrix an, DiagonalWerte logischerweise nutzlos

Data Selection

Split into train/test sets

Create and train Model

Eval and understand Model

plt.ylim(1,5) -> möglichkeit Raender fuer Werte bei einem plot anzugeben (hier fuer y)

Further Modeling:
Example Function:


def model_these_features(feature_list):
    
    # define ratings and features, with the features limited to our chosen subset of data
    ratings = df.loc[:,'stars']
    features = df.loc[:,feature_list]
    
    # perform train, test, split on the data
    X_train, X_test, y_train, y_test = train_test_split(features, ratings, test_size = 0.2, random_state = 1)
    
    # don't worry too much about these lines, just know that they allow the model to work when
    # we model on just one feature instead of multiple features. Trust us on this one :)
    if len(X_train.shape) < 2:
        X_train = np.array(X_train).reshape(-1,1)
        X_test = np.array(X_test).reshape(-1,1)
    
    # create and fit the model to the training data
    model = LinearRegression()
    model.fit(X_train,y_train)
    
    # print the train and test scores
    print('Train Score:', model.score(X_train,y_train))
    print('Test Score:', model.score(X_test,y_test))
    
    # print the model features and their corresponding coefficients, from most predictive to least predictive
    print(sorted(list(zip(feature_list,model.coef_)),key = lambda x: abs(x[1]),reverse=True))
    
    # calculate the predicted Yelp ratings from the test data
    y_predicted = model.predict(X_test)
    
    # plot the actual Yelp Ratings vs the predicted Yelp ratings for the test data
    plt.scatter(y_test,y_predicted)
    plt.xlabel('Yelp Rating')
    plt.ylabel('Predicted Yelp Rating')
    plt.ylim(1,5)
    plt.show()
    
    Anwendung auf eigenen Case:
    
    zum Erstellen einer Tabelle fuer verschiedene Features der Columns:
    pd.DataFrame(list(zip(features.columns,features.describe().loc['mean'],features.describe().loc['min'],features.describe().loc['max'])),columns=['Feature','Mean','Min','Max'])
    
    Numpy Funktion .reshape -> (1, -1) fuegt eine Dimension hinzu und packt alle Werte in die erste Liste, (-1, 1) fuegt auch eine Dim hinzu, packt aber alle Werte in eigene Liste
"""