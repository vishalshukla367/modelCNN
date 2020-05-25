from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import os
def add_layer(model):
        model.add(Dense(16, activation='relu'))

# load the dataset
dataset = loadtxt(r'/usr/mlcnn/pima-indians-diabetes.csv', delimiter=',')
# split into input (X) and output (y) variables
print("Welcomne ! yes ho bhaisab tk ,so now final testing")
X = dataset[:,0:8]
y = dataset[:,8]
# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
add_layer(model)
model.summary()
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X, y, epochs=150, batch_size=10)
# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))
os.system('echo {} | cat >> /usr/mlcnn/accuracy.txt'.format(str((accuracy*100))))
