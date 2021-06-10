from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

emails = fetch_20newsgroups()
# print(emails.target_names)
emails = fetch_20newsgroups(categories=['rec.sport.baseball', 'rec.sport.hockey'])
print(emails.target_names)
print(emails.data[5])
print(emails.target[5])
# Die eins hier bedeutet, dass das label an index 1 steht, es ist also eine Hockey Mail

train_emails = fetch_20newsgroups(categories=['rec.sport.baseball', 'rec.sport.hockey'], subset="train", shuffle=True,
                                  random_state=108)
test_emails = fetch_20newsgroups(categories=['rec.sport.baseball', 'rec.sport.hockey'], subset="test", shuffle=True,
                                 random_state=108)
counter = CountVectorizer()
counter.fit(test_emails.data + train_emails.data)
train_counts = counter.transform(train_emails.data)
test_counts = counter.transform(test_emails.data)
classifier = MultinomialNB()
print(train_emails.target)
classifier.fit(train_counts, train_emails.target)
print(classifier.score(test_counts, test_emails.target))

"""Die Genauigkeit bei der Entscheidung zwischen Baseball und Hockey Emails ist 97%. Das ist ganz gut.
Zum Vergleich koennen wir uns noch die Genauigkeit bei zwei definitiv disjunkten Themengebieten (Sport und Tech) anschauen"""


train_emails = fetch_20newsgroups(categories=['comp.sys.ibm.pc.hardware', 'rec.sport.hockey'], subset="train",
                                  shuffle=True, random_state=108)
test_emails = fetch_20newsgroups(categories=['comp.sys.ibm.pc.hardware', 'rec.sport.hockey'], subset="test",
                                 shuffle=True, random_state=108)
counter = CountVectorizer()
counter.fit(test_emails.data + train_emails.data)
train_counts = counter.transform(train_emails.data)
test_counts = counter.transform(test_emails.data)
classifier = MultinomialNB()
classifier.fit(train_counts, train_emails.target)
print(classifier.score(test_counts, test_emails.target))

"""Wie erwartet ist die 99 Genauigkeit hier sogar noch besser."""