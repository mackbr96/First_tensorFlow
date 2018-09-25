import tensorflow as tf
    
mnist = tf.keras.datasets.mnist
print(mnist)

(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=4)


val_loss, val_acc = model.evaluate(x_test, y_test)

print(val_loss, val_acc)


model.save('save.model')
new_model = tf.keras.models.load_model('save.model')

predic = new_model.predict([x_test])

import numpy as np
import matplotlib.pyplot as plt


#print(np.argmax(predic[1]))

#plt.imshow(x_test[1])


predictions = model.predict(x_test)
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.subplots_adjust(hspace=.5)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_test[i], cmap=plt.cm.binary)
    plt.xlabel(np.argmax(predictions[i]))


plt.show()
