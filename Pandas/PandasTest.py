x = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print(x)
print(x[1])  # greift auf die Reihe zu
print(x[1][2])  # greift auf die Spalte zu

from sklearn.datasets import load_boston
import pandas as pd

boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df = df[:6]  # hier nimmt er die row indices, df[6] wuerde nicht funktionieren
print(df)
print("df.iloc[5]")
print(df.iloc[5])
# hier machen indexLocator(iloc) und der normale Locator (loc) dasselbe, da hier rows unbenannt bzw. index-benannt
print("df[[\"LSTAT\"]]")
print(df[["LSTAT"]])
print("df.loc[5][0]")
print(df.loc[5][0])
print("df.loc[:3][\"LSTAT\"]")
print(df.loc[:3]["LSTAT"])
print("""df.loc[5, ["CRIM", "LSTAT"]]""")
print(df.loc[5, ["CRIM", "LSTAT"]])
print("""df.loc[5, "CRIM": "INDUS"]""")
print(df.loc[5, "CRIM": "INDUS"])
print("df[\"CRIM\"] >= 0.01")
print(df["CRIM"] >= 0.01)
print("df[df[\"CRIM\"] >= 0.01]")
print(df[df["CRIM"] >= 0.01])
print(
    """df.where((df["CRIM"] >= 0.01) & (df["LSTAT"] >= 4),inplace=True)""")  # and und or Zeichen sind & bzw. | in pandas
df.where((df["CRIM"] >= 0.01) & (df["LSTAT"] >= 4), inplace=True)
print(df)
print("""df.dropna()""")
df.dropna(inplace=True)
print(df)

data = {'Name': ['Jai', 'Princi', 'Gaurav', 'Anuj'],
        'Age': [27, 24, 22, 32],
        'Address': ['Delhi', 'Kanpur', 'Allahabad', 'Kannauj'],
        'Qualification': ['Msc', 'MA', 'MCA', 'Phd']}

df2 = pd.DataFrame(data)

print(df2[df2.columns[1:4]])
print(df2.iloc[0, :])  # nimmt row 0 und alle columns
print(df2.iloc[0:2, 1:3])
print(df2.loc[df2['Age'] == 22])
for i in range(len(df2.index)):
    print(list(df2.iloc[i]))
print("Namen aus data als Liste")
print(df2["Name"].tolist())
# df2['sex'] = df2['sex'].map({'male': 0, 'female': 1})
print("title als Spalte")


# df2['title'] = df2['Qualification'].map(lambda x: "Dr." if (x == "Phd") else "Mr.")
# andere Moeglichkeiten waearen .applymap (fuer jedes Element einzeln)(DF) oder apply (fuer eine Column als ganzes)(DF oder DS)
# diese beiden accepten allerdings nur callables
def title(y):
    if y == "Phd":
        return "Dr."
    else:
        return "Mr."


df2['title'] = df2['Qualification'].map(title)

print(df2)
print("""df2[df2["Age"] > 22]""")
print(df2[df2["Age"] > 22])
print("""df2[df2["Age"] > 22]["Address"]""")
print(df2[df2["Age"] > 22]["Address"])
df2[df2["Age"] > 22]["Address"] = "Test"
# print(df2) #aendert nichts, gibt auch die (bekannte) Warning
df2.loc[df2["Age"] > 22, "Address"] = "Test"
print(df2)  # aendert entsprechend die Addresse fuer alle ueber 22
df2 = df2[df2["Age"] > 22]
df2["Address"] = "New"
print(df2)  # Cuttet alle mit Age unter 23 (und setzt neue Addresse im zweiten Schritt)
"""passengers['Sex'] = passengers['Sex'].map({'male': 0, 'female': 1})"""
print()
print(df2["title"].value_counts())  # gibt alle Werte der Series sowie ihre Haeufigkeit wieder (sehr praktisch)

"""
Beispiel fuer die Nutzung einer .apply Funktion in pandas
 def custom_rating(genre,rating):
    if 'Thriller' in genre:
        return min(10,rating+1)
    elif 'Comedy' in genre:
        return max(0,rating-1)
    else:
        return rating
        
df['CustomRating'] = df.apply(lambda x: custom_rating(x['Genre'],x['Rating']),axis=1)
"""

x = "Zehn zahme Ziegen"
print(x.split())


def find_average_word_length(long_string):
    string_list = long_string.split() #splittet an jedem Leerzeichen
    y = len(long_string.split())
    z = 0
    for string in string_list:
        z += len(string)
    return z / y

print(find_average_word_length(x))

s = [0] * 5
print(s)