"""WK Rechnung nach Bayes-Theorem bereits aus DuW bzw. DA bekannt."""

p_disease_and_correct = (1.0 / 100000) * (99.0 / 100)
print(p_disease_and_correct)
p_no_disease_and_incorrect = (99999.0 / 100000) * (1.0 / 100)
print(p_no_disease_and_incorrect)

# P(positive result | rare disease)
p_positive_given_disease = (99.0 / 100)
# print(p_positive_given_disease)
p_disease = (1.0 / 100000)
# print(p_disease)
p_positive = (1.0 / 100000) * (99.0 / 100) + (99999.0 / 100000) * (1.0 / 100)
p_positive = p_disease_and_correct + p_no_disease_and_incorrect  # dasselbe
p_disease_given_positive = p_positive_given_disease * p_disease / p_positive
print(p_disease_given_positive)

# der letzte String ist hierbei immer die unabh. Variabel, der erste die gegebene
p_spam = 0.2
p_nospam = 0.8
p_spam_enhancement = 0.05
p_spam_noenhancement = 0.95
p_nospam_enhancement = 0.001
p_nospam_noenhancement = 0.999
# p_enhancement_spam = ?
p_enhancement = p_spam * p_spam_enhancement + p_nospam * p_nospam_enhancement
print("p_enhancement")
print(p_enhancement)
p_noenhancement = p_spam * p_spam_noenhancement + p_nospam * p_nospam_noenhancement
print("p_noenhancement")
print(p_noenhancement)
p_enhancement_spam = 0.05 * 0.2 / p_enhancement
p_enhancement_nospam = 0.001 * 0.8 / p_enhancement
print(p_enhancement_spam)
p_noenhancement_spam = 0.95 * 0.2 / p_noenhancement
p_noenhancement_nospam = 0.999 * 0.8 / p_noenhancement
print(p_noenhancement_spam)
p_spam = p_enhancement * p_enhancement_spam + p_noenhancement * p_noenhancement_spam
print(p_spam)
p_nospam = p_enhancement * p_enhancement_nospam + p_noenhancement * p_noenhancement_nospam
print(p_nospam)

from data.data_reviews import neg_list, pos_list, neg_counter, pos_counter

# Lists of positive/negative reviews, and Counter (via the Collection.counter) of the words in there

total_reviews = len(neg_list) + len(pos_list)
# print(total_reviews) es sind 100
percent_pos = float(len(pos_list)) / total_reviews
percent_neg = float(len(neg_list)) / total_reviews
# 0.5 each

review = "This crib was amazing"

total_pos = sum(pos_counter.values())  # Summe aller Wörter in positiv
total_neg = sum(neg_counter.values())
# Startwerte fuer die WKeiten, wird in der Vorschleife spaeter realistischer
pos_probability = 1.0
neg_probability = 1.0
review_words = review.split()
for word in review_words:
    word_in_pos = pos_counter[word]
    pos_probability = pos_probability * (word_in_pos / total_pos)
    # This is basically the probability of the word being "This" (for review[0]) when we know the review is positive
    word_in_neg = neg_counter[word]
    neg_probability = neg_probability * (word_in_neg / total_neg)
"""pos_probability (und neg) außerhalb der for-Schleife geben jeweils die Chance, dass der Satz “This crib was amazing” wird, 
wenn wir wissen dass das review positiv (negativ) ist."""
# Großes Problem an unserer Berechnung -> Sollte ein Wort aus dem review bisher nicht in pos_counter vorgekommen sein, wird die pos_probability des Satzes automatisch 0 (Multiplikation mit 0)
# Die Loesung is sog. Smoothing
pos_probability = 1.0
neg_probability = 1.0
for word in review_words:
    word_in_pos = float(pos_counter[word])
    print("Haeuftigkeit {} in positiven Bewertungen".format(word))
    print(word_in_pos)
    word_in_neg = float(neg_counter[word])
    print("Haeuftigkeit {} in negativen Bewertungen".format(word))
    print(word_in_neg)
    pos_probability *= (word_in_pos + 1.0) / (total_pos + len(pos_counter))
    # print(pos_probability) -> geht je laenger der Satz auch automatisch immer weiter runter
    neg_probability *= (word_in_neg + 1.0) / (total_neg + len(neg_counter))
print("""Chance das ein zufaelliger positiver Satz genau "This crib was amazing" ist:""")
print(pos_probability)
print("""Chance das ein zufaelliger negativer Satz genau "This crib was amazing" ist:""")
print(neg_probability)
p_pos_review = pos_probability
p_neg_review = neg_probability

"""Aus diesem Teil koennen wir nun auf die WK das ein gegebenes review "This crib was amazing" positiv ist, schließen"""

p_review = (pos_probability * percent_pos + neg_probability * percent_neg)
print("""Chance das ein zufaelliger Satz egal welcher Art genau "This crib was amazing" ist:""")
print(p_review)
p_noreview = (1 - pos_probability) * percent_pos + (1 - neg_probability) * percent_neg
print("""Chance das ein zufaelliger Satz egal welcher Art nicht genau "This crib was amazing" ist:""")
print(p_noreview)

p_review_pos = pos_probability * percent_pos / p_review
print("""Chance das ein Satz "{}" positiv ist:""".format(review))
print(p_review_pos)
p_review_neg = neg_probability * percent_neg / p_review
print("""Chance das ein Satz "{}" negativ ist:""".format(review))
print(p_review_neg)
if p_review_pos > p_review_neg:
    print("The review is positive")
else:
    print("The review is negative")

"""Fuer unser Beispiel review macht der Classifier es tatsaechlich falsch, das Ergebnis ist aber auch sehr uneindeutig (52 zu 48%)
Probiert man das ganze z.B. mit "this worked very well" bekommt man ein deutlich positives Ergebnis"""
"""Um das Gnaze jetzt mit sklearn zu machen:"""
from sklearn.feature_extraction.text import CountVectorizer
"""
vectorizer = CountVectorizer()
vectorizer.fit(["Training review one", "Second review"])
counts = vectorizer.transform(["one review two review"])
print(counts)  # sollte als .toarray() geprintet werden, sonst nur schwer verstaendlich
# counts stores [1, 2, 0, 0] for ["one", "review", "second", "training"], da "two" im vectorizer nicht gefittet wurde, hier auch keine Zuordnung, das ist natuerlich nur noch so seminuetzlich
# wenn unser CountVectorizer mehrere tausend Woerter (oder mehr) umfasst
print(vectorizer.vocabulary_)
# gibt den index fuer das entsprechende Wort im Array vom vectorizer wieder

# print(["1", "2"] + ["3", "4"]) ListenAddition (Macht dasselbe wie extend)
"""
review = "This crib was amazing"
counter = CountVectorizer()
counter.fit(neg_list + pos_list)
review_counts = counter.transform([review])  # nachher als test point, uebergeben werden muss aber eine Liste, ein String geht nicht!
print(review_counts.toarray())
print("Counter Vocabulary")
print(counter.vocabulary_)
training_counts = counter.transform(neg_list + pos_list)  # unser trainings-set

from sklearn.naive_bayes import MultinomialNB

print(len(neg_list), len(pos_list))
# labels werden entsprechend der Laenge der neg/pos_list zugeordnet
training_labels = []
#negativ
for i in range(50):
    training_labels.append(0)
#positiv
for i in range(50):
    training_labels.append(1)
classifier = MultinomialNB()
classifier.fit(training_counts, training_labels)
print(classifier.predict(review_counts))
# Man beachte, dass wir hier um mit sklearn arbeiten zu koennen, das transformte training_set, bzw. Ziel zur Bestimmung brauchen. classifier.predict(review) wuerde entsprechend nicht funktionieren
print(classifier.predict_proba(review_counts))
# erste nummer -> negative, zweite -> positiv
review = "This just does not work and the disappointment would rise"
review_counts = counter.transform([review])
print(classifier.predict(review_counts))
print(classifier.predict_proba(review_counts))
"""Unser Datensatz ist recht klein, entsprechend maeßig funktioniert unsere Einordnung"""
review = "I bought this for our baby and it kepps it on track. Works really great"
review_counts = counter.transform([review])
print(classifier.predict(review_counts))
print(classifier.predict_proba(review_counts))

"""Moegliche Verbesserung (neben der offensichtlichen: groeßeres trainingsSet):
Alles in lowercase transformen. Punctuation entfernen.
Außerdem: bigram (oder sogar trigram) Modell, indem immer zwei Wörter zusammen bleiben. ("this crib is great" -> “This crib”, “crib is”, and “is great”)
"""
"""
import string
x = ["TEST! test","test.","Test,"]
for i in range(len(x)):
    x[i] = x[i].lower().translate(str.maketrans('', '', string.punctuation))
    #removed alle string punctuation, aber keine Whitespaces, ansonsten koennte man eine custom Funktion schreiben, die ein Set zu entfernender Punctation kriegt und durch den String iteriert.
print(x)
"""