from sklearn.feature_extraction.text import CountVectorizer
from sklearn import tree

good_phrases = []
bad_phrases = []

with open('good.txt', 'r') as f:
    good_phrases = [phrase.strip() for phrase in f.readlines()]

with open('bad.txt', 'r') as f:
    bad_phrases = [phrase.strip() for phrase in f.readlines()]

training_texts = good_phrases + bad_phrases
training_labels = ["Frase positiva"] * len(good_phrases) + ["Frase negativa"] * len(bad_phrases)

vectorizer = CountVectorizer()
vectorizer.fit(training_texts)


training_vectors = vectorizer.transform(training_texts)

testing_phrases = []

phrase = input('Digite uma frase:')

testing_phrases.append(phrase)
testing_vectors = vectorizer.transform(testing_phrases)


# Adicionar regularização L2
classifier = tree.DecisionTreeClassifier()
classifier.fit(training_vectors, training_labels)

prediction = classifier.predict(testing_vectors)
for i in range(0, len(prediction)):
    print(f"*{testing_phrases[i]} - {prediction[i]}\n")
