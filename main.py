import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

norm_layer = tf.keras.layers.Normalization(axis=-1)

def main():
    model = tf.keras.models.load_model('coffee_roasting_model.keras')

    print("Welcome to the Coffee Roasting Predictor! ")
    x1 = int(input("Enter a temperature (C): "))
    x2 = int(input("Enter a duration (minutes): "))

    x_new = np.array([[x1, x2]])
    x_new_normalized = norm_layer(x_new)

    y_pred = model.predict(x_new_normalized)
    print(y_pred)
    if y_pred >= 0.5:
        y_hat = 1
    else:
        y_hat = 0
    if y_hat == 1:
        print("This will make good coffee!")
    else:
        print("This will make bad coffee :( ")

main()