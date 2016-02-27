import keras.layers.core as core
import keras.models as kmodels
import keras.optimizers as optm
import keras.utils.np_utils as kutils
import numpy as np

X = [[0, 0],
     [0, 1],
     [1, 0],
     [1, 1]]

X = np.array(X)

numInputs = 2

y = [0, 1, 1, 0]

yClasses = kutils.to_categorical(y)

# Params
epochs = 10000
numX = len(X)
numClasses = 2 # Force binary classification

model = kmodels.Sequential()
model.add(core.Dense(4, input_shape=(numInputs,), activation="sigmoid", ))
model.add(core.Dense(numClasses, activation="sigmoid"))

model.summary()

model.compile(optimizer="adam", loss="mse")
model.fit(X, yClasses, batch_size=numX, nb_epoch=epochs, show_accuracy=True, )

yPreds = model.predict_classes(X, batch_size=numX)
print(yPreds)