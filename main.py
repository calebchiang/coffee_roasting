import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

model = tf.keras.models.load_model('coffee_roasting_model.keras')
def main():

    print("Welcome to the Coffee Roasting Predictor! ")
    x1 = float(input("Enter a temperature (C): "))
    x2 = float(input("Enter a duration (minutes): "))

    x_new = np.array([[x1, x2]])
    y_pred = model.predict(x_new)
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