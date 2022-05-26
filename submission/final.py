import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

#definition of the data that was selected via the sensor
df = pd.read_csv('LabeledDataFrame.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values


X = X.reshape(12000,64,128)

#definition of the training data and the test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)

#sns.histplot(y_train)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix

#defining model
model=Sequential()
#adding convolution layer
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=[64,128,1]))
#model.add(Conv2D(32,(3,3),activation='relu',input_shape=(64,128,1)))
#adding pooling layer
model.add(MaxPool2D(2,2))
#adding fully connected layer
model.add(Flatten())
model.add(Dense(128,activation='relu'))
#adding output layer
model.add(Dense(128,activation='relu'))
#adding output layer
model.add(Dense(4,activation='softmax'))
#compiling the model
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#fitting the model
model.fit(X_train,y_train,epochs=20)



pred = model.predict(X_test)
predictions=np.zeros(len(pred))
for i in range(len(pred)):
    maxx = pred[i].argmax()
    predictions[i]=maxx
  
        
accuracy_score(y_test,predictions)   #71.3%
confusion_matrix(y_test,predictions)
model.save('model1')

###################################
