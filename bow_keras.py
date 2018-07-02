from keras.preprocessing.text import Tokenizer
#define 5 documents
docs = ['SCRMBL test one','bag of words under keras','parameter estimates coming shortly']
#create the tokenizer
t = Tokenizer()
#fit the tokenizer on the documents
t.fit_on_texts(docs)
#sumarize what was learned
print(t.word_counts)
print(t.document_count)
print(t.word_index)
print(t.word_docs)
#integer encode documents
encoded_docs = t.texts_to_matrix(docs, mode='count')
print(encoded_docs)

