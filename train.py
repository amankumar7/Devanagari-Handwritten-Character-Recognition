import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Activation,Flatten,Dropout
from tensorflow.keras.models import Sequential
from data_preprocessing import data
from tensorflow.keras.callbacks import TensorBoard
import time




m_name = "Dev-{}".format(int(time.time()))
tb = TensorBoard(log_dir='logs/{}'.format(m_name))



x_train,y_train,lables_id = data('train')
x_train,y_train,lables_id = data('test')

rec_labels={v:k for k,v in labels_id.items() }



#plt.imshow(x_train[7])
#title=rec_labels[y_train[7]]
#plt.title(title)
#plt.show()

x_test = np.array(x_test).reshape(-1,32,32,1)
y_test = tf.keras.utils.to_categorical(y_test,num_classes=46)
y_train = tf.keras.utils.to_categorical(y_train,num_classes=46)

x_train = x_train.astype("float") / 255.0
x_test = x_test.astype("float") / 255.0



model = Sequential()
model.add(Conv2D(32,(5,5),strides=(2,2),padding='same',input_shape=x_train.shape[1:]))
model.add(Conv2D(32,(3,3),strides=(2,2),padding='same'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3),strides=(1,1),padding='same'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(128))
model.add(Dropout(0.4))

model.add(Activation("relu"))


model.add(Dense(46))

model.add(Activation("softmax"))


model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=['accuracy'])



model.fit(x_train,y_train,batch_size=32,epochs=10,verbose=1,validation_split=.20,callbacks=[tb])

scores  = model.evaluate(x_test,y_test,verbose=1,batch_size=32)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

model.save(devchar.model)
