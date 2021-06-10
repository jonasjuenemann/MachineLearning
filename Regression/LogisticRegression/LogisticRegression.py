import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from data.data_exam import hours_studied, passed_exam, calculated_coefficients, intercept

# from plotter import plot_data

# Create linear regression model
model = LinearRegression()
model.fit(hours_studied, passed_exam)

# Plug sample data into fitted model
sample_x = np.linspace(-16.65, 33.35, 300).reshape(-1, 1)
probability = model.predict(sample_x).ravel()

# Function to plot exam data and linear regression curve
# plot_data(model)

# Show the plot
plt.show()

print(model.coef_)  # coeff ist hier nicht der selbe da hier noch -> LinearRegression, keine Logistic
print(model.intercept_)

# Define studios and slacker here
slacker = model.predict([[1]])
studious = model.predict([[19]])
print(slacker)
print(studious)


# plt.plot(19, studious, 'o', color="green")


# Problem: slacker liegt bei -0,13..., studious bei 1,1... -> kein sinnvoller Wert -> Logitische Regression:
# model = LogisticRegression()


# Manuell:
# np.dot() zur Berechnung von Matrix features mit Vector coefficents
def log_odds(features, coefficients, intercept):
    return np.dot(features, coefficients) + intercept


# Calculate the log-odds for the Codecademy University data here
calculated_log_odds = log_odds(hours_studied, calculated_coefficients, intercept)
print(calculated_log_odds)


# gibt dem ganzen eine Probability von 0 -> 1, eben genau der Sinn einer Logistic Regression
def sigmoid(z):
    denominator = 1 + np.exp(-z)
    return 1 / denominator


# Calculate the sigmoid of the log-odds here
probabilities = sigmoid(calculated_log_odds)
print(probabilities)


def log_loss(probabilities_i, actual_class):
    return np.sum(-(1 / actual_class.shape[0]) * (actual_class * np.log(probabilities_i) +
                                                  (1 - actual_class) * np.log(1 - probabilities_i)))


# Calculate and print loss_1 here
# loss_1 = log_loss(probabilities , passed_exam)
# -> funktioniert nicht, da .shape[0] aus vorheriger Funktion sich nicht auf Listen anweden laesst ???
# print(loss_1)
# Calculate and print loss_2 here
# loss_2 = log_loss(probabilities_2 , passed_exam)
# print(loss_2)

#Classification threshold: ab welcher WK das Ganz als positiv angesehen wird

def predict_class(features, coefficients, intercept, threshold):
    calculated_log_odds = log_odds(features, coefficients, intercept)
    probabilities = sigmoid(calculated_log_odds)
    # h_array = [[1 if value >= threshold else 0 for value in probabilities]]
    return np.where(probabilities >= threshold, 1, 0)


# Make final classifications on Codecademy University data here



final_results = predict_class(hours_studied, calculated_coefficients, intercept, 0.5)
print(final_results)

from data.data_exam import hours_studied_scaled, passed_exam, exam_features_scaled_train, exam_features_scaled_test, \
    passed_exam_2_train, passed_exam_2_test, guessed_hours_scaled

# Create and fit logistic regression model here
model = LogisticRegression()
model.fit(hours_studied_scaled, passed_exam)
# Save the model coefficients and intercept here
calculated_coefficients = model.coef_
intercept = model.intercept_
# Predict the probabilities of passing for next semester's students here
# model.predict(features) #1 oder 0
passed_predictions = model.predict_proba(guessed_hours_scaled)  # Chances 0-1
# print(passed_predictions)
# Create a new model on the training data with two features here
model_2 = LogisticRegression()
model_2.fit(exam_features_scaled_train, passed_exam_2_train)

# Predict whether the students will pass here
passed_predictions_2 = model_2.predict(exam_features_scaled_test)

print(passed_predictions_2)
print(passed_exam_2_test)

coefficients = model_2.coef_
coefficients = coefficients.tolist()[0]
print(coefficients)

plt.bar([1, 2], coefficients)
plt.xticks([1, 2], ['hours studied', 'math courses taken'])
plt.xlabel('feature')
plt.ylabel('coefficient')

plt.show()

"""
plt.plot(guessed_hours_scaled, passed_predictions) #alpha -> eine Art Opacity
# Plot line here:
#plt.plot(exam_features_scaled_train, exam_features_scaled_test)
plt.ylabel("predictions")
plt.xlabel("Hours")
plt.show()
"""
