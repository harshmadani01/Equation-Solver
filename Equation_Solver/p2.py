import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt

df=pd.read_csv('C:\\Users\\JAY PATEL\\Desktop\\train.csv',index_col=False)
labels = df[['784']]

df.drop(df.columns[[784]], axis=1, inplace=True)

labels = np.array(labels)
cat=to_categorical(labels,num_classes=18, dtype="uint8")

label = []
for i in range(len(labels)):
    label.append(np.array(df[i:i+1]).reshape(28,28,1))

model = Sequential()
model.add(Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(15, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(18, activation='softmax'))

# use of adam optimizer to update neural network weights based on iterations
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # accuracy = (correct prediction)/(total)
history = model.fit(np.array(label), cat, epochs=10, batch_size=200,shuffle=True,verbose=1)

model_json = model.to_json()
with open("create_model_final.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("create_model_final.h5")