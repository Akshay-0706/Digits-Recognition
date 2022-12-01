from sklearn import metrics
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint

print("Handwritten digits recognition\n")

print("Getting data...")

# Get the data and pre-process it
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# show data present at index i
def show_digit(i):
    plt.imshow(x_train[i], cmap='binary')
    plt.title(y_train[i])
    plt.show()


show_digit(0)

# pre-process the images
x_train = x_train.astype(np.float32) / 255
x_test = x_test.astype(np.float32) / 255


# reshape the dimensions of images to (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)


# convert classes to one hot vector
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)


# build model
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPool2D(2, 2))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D(2, 2))

model.add(Flatten())

model.add(Dropout(0.25))

model.add(Dense(10, activation="softmax"))


print("Model summary")
model.summary()


# compile model
model.compile(optimizer="adam",
              loss=keras.losses.categorical_crossentropy, metrics=["accuracy"])


# earlystopping
es = EarlyStopping(monitor="val_accuracy",
                   min_delta=0.01, patience=19, verbose=1)


# model check point
mc = ModelCheckpoint("bestmodel.h5", monitor="val_accuracy",
                     verbose=1, save_best_only=True)


# callbacks
cb = [es, mc]


print("Training model")
# train the model
his = model.fit(x_train, y_train, epochs=20,
                validation_split=0.3)

# Plot accuracy
plt.plot(his.history["accuracy"])
plt.plot(his.history["val_accuracy"])
plt.title("Model accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train", "Validation_acc"], loc="upper left")
plt.show()

# Plot loss
plt.plot(his.history["loss"])
plt.plot(his.history["val_loss"])
plt.title("Model loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Train", "Validation_loss"], loc="upper right")
plt.show()

# save trained model
model.save("bestmodel.h5")

# get model
bestmodel = keras.models.load_model("bestmodel.h5")


# evaluate model
score = bestmodel.evaluate(x_test, y_test)
print(f"The model accuracy is {score[1]}")
print(f"The model loss is {score[0]}")
