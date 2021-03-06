from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
#define problem
vocab_size = 100
max_length = 32
#defining the model
model = Sequential()
model.add(Embedding(vocab_size, 8, input_length = max_length))
model.add(Flatten())
model.add(Dense(1, activation = 'sigmoid'))
#compiling the model
model.compile(optimizer =  'adam', loss = 'binary_crossentropy', metrics = ['acc'])
#summarize the model
model.summary()
