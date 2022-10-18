import tensorflow as tf
from tensorflow import keras
from keras import activations
import numpy as np

# fashion_mnist is a collection of 60.000 train data with labels
# and 10.000 test data also with labels(labels provided if we want to check results)
fashion_mnist = keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# x_train = x_train[:1000]
# y_train = y_train[:1000]
# x_test = x_test[:1000]
# y_test = y_test[:1000]

# Just for clarification. If we get:
# 0 -> T-shirt/top
# 1 -> Trouser
# 2 -> Pullover
# ...
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Normalize data. From range 0 to 255 to -> 0 to 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Make model(Multilayer Perceptron). Consist of 3 layers:
# - Input layer Flatten. They will convert 2d image to 1d vector. Expect image 28px * 28px
# - Hidden layer is the main. Use relu because they are most common activation function
# - Output layer is softmax activation function because softmax is good then we have to do non binary choise. 10 units because we have only 10 variants of clothes
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), # present image as 1 dimension vector. Images are presented as 28px * 28px
    keras.layers.Dense(128, activation=activations.tanh),
    keras.layers.Dense(10, activation=activations.softmax) # Output layer. Softmax because we do classification(from 0 to 9)
])

optimizer = keras.optimizers.SGD() # SGD is good for image recognition
loss = 'sparse_categorical_crossentropy' # good then we have non binary choise
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# Train model
model.fit(x_train, y_train, epochs=10) # give train data and retry 10 times


# Try to use model
PREDICT_INDEX = 10 # What index of image we want to test?
predictions = model.predict(x_test)

chosen_class = np.argmax(predictions[PREDICT_INDEX]) # For image get results. Get index of maximum class chance
y_test_current = y_test[PREDICT_INDEX]
print('Expected:', y_test_current, class_names[y_test_current], 'Got:', chosen_class, class_names[chosen_class])
