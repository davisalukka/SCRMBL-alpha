from keras.models import Sequential
from keras.layers import Dense               
from keras.layers import Flatten         
from keras.layers.embeddings import Embedding 
#defining the problem
vocab_size = 100
max_length = 200
#define model
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length = max_length))
model.add(Conv1D(filters=32,kernel_size = 8, activation = 'relu'))
model.add(MaxPooling1D(pool_size = 2))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.summary()