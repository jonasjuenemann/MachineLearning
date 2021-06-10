labels = [1, 0, 0, 1, 1, 1, 0, 1, 1, 1]
guesses = [0, 1, 1, 1, 1, 0, 1, 0, 1, 0]

true_positives, true_negatives, false_positives, false_negatives = 0, 0, 0, 0

for i in range(len(labels)):
    if (labels[i] == 1) and (guesses[i] == 1):
        true_positives += 1
    if (labels[i] == 0) and (guesses[i] == 0):
        true_negatives += 1
    if (labels[i] == 0) and (guesses[i] == 1):
        false_positives += 1
    if (labels[i] == 1) and (guesses[i] == 0):
        false_negatives += 1

accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
print(accuracy)
recall = true_positives / (true_positives + false_negatives)
# Bei Werten die sehr oft falsch sind (z.B. Schnee morgen?), also nur die positiven guesses im Verhaeltnis zu den positiven Ereignissesn relevant sind
print(recall)
precision = true_positives / (true_positives + false_positives)
# Damit man nun nicht einfach immer True eingeben kann (haette recall von 1)
print(precision)
f_1 = (precision*recall) / (precision+recall) * 2
# Harmonisches Mittel aus Precision und Recall fuer einen
print(f_1)


"""Kann man auch aus sklearn bestimmen:"""

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

labels = [1, 0, 0, 1, 1, 1, 0, 1, 1, 1]
guesses = [0, 1, 1, 1, 1, 0, 1, 0, 1, 0]

print(accuracy_score(labels, guesses))
print(recall_score(labels, guesses))
print(precision_score(labels, guesses))
print(f1_score(labels, guesses))
