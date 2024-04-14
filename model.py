import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Normalization
from tensorflow.keras.losses import BinaryCrossentropy

tf.random.set_seed(1234)

# Load data
data = pd.read_csv('coffee_data.csv')
X = data[['Temperature', 'Duration']].values
Y = data['Good Roast'].values
Y = Y.reshape(-1, 1)

norm_layer = Normalization(axis=-1)
norm_layer.adapt(X)


model = Sequential([
    norm_layer,  # Include the normalization layer
    Dense(3, activation='sigmoid', name='layer1'),  # First hidden layer
    Dense(1, activation='sigmoid', name='layer2')   # Output layer
])

# Compile the model
model.compile(
    loss=BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
)

X_tiled = np.tile(X, (1000, 1))
Y_tiled = np.tile(Y, (1000, 1))


model.fit(X_tiled, Y_tiled, epochs=10)
model.save('coffee_roasting_model.keras')
