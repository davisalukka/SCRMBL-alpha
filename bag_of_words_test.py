from sklearn.feature_extraction.text import TfidfVectorizer
#list of text documetns
text = ["setting up bag of words","urgency modelling","Using CNN"]
#create the transform
vectorizer = TfidfVectorizer()
#tokenize and  build vvocab
vectorizer.fit(text)
#summarize
print(vectorizer.vocabulary_)
print(vectorizer.idf_)
#encode document
vector = vectorizer.transform([text[0]])
#summarize encoded vector
print(vector.shape)
print(vector.toarray())

