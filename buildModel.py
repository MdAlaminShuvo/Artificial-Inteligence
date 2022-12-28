from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

inputs = Input((756,),name= 'Input')
x1 = Dense(16, activation= 'relu', name = 'Hidden1')(inputs)
x2 = Dense(32, activation= 'tanh', name = 'Hidden2')(x1)
x3 = Dense(64, activation= 'sigmoid', name = 'Hidden3')(x2)
outputs = Dense(10,activation= 'softmax', name = 'Output')(x3)

model5 = Model(inputs, outputs)
model5.summary()
