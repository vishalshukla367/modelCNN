
import pandas as pd
dataset = pd.read_csv('weight-height.csv')

y = dataset['Weight']

from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import Adam

model = Sequential()

model.add(Dense(units=1 , input_shape=(1,)  ))


model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.000001))

model.fit(X,y, epochs=20)

W , B = model.get_weights()

W

B

W[0,0] = 0.0
B[0] = 0.0

model.set_weights((W,B))
model.get_weights()




