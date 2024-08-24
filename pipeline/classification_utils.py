# classification_utils.py

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import keras
from tensorflow.keras.preprocessing import image
import os


def warmup_fd(food_model,class_food):
    # Step 2: Prepare input data (you may need to adjust this based on your specific use case)
    food_img_path = r"C:\Users\Vinh\Downloads\ViT_codes\warmup\food\củ tím\img_0306.jpg"
    img_fd = image.load_img(food_img_path, target_size=(224, 224))
    img_array1 = image.img_to_array(img_fd)
    img_array1 = np.expand_dims(img_array1, axis=0)
    #img_array1 /= 255.0  # Assuming your model expects input values in the range [0, 1]

    # Step 3: Warm-up the model with a few forward passes
    num_warmup_steps = 3  # You can adjust this based on your needs

    for _ in range(num_warmup_steps):
        predictions1 = food_model.predict(img_array1)
    # Print the model predictions for the last pass
    predicted1 = np.argmax(predictions1[0])
    print("predicted: ", class_food[predicted1])

    print("Food model warmed up successfully!")
def warmup_dr(drink_model,class_drink):
    drink_img_path = r"C:\Users\Vinh\Downloads\ViT_codes\warmup\drink\pepsi\framf0313.jpg"
    img_dr = image.load_img(drink_img_path, target_size=(224, 224))
    img_array2 = image.img_to_array(img_dr)
    img_array2 = np.expand_dims(img_array2, axis=0)
    #img_array2 /= 255.0

    # Step 3: Warm-up the model with a few forward passes
    num_warmup_steps = 3  # You can adjust this based on your needs

    for _ in range(num_warmup_steps):
        predictions2 = drink_model.predict(img_array2)
    # Print the model predictions for the last pass
    predicted2 = np.argmax(predictions2[0])
    print("predicted: ", class_drink[predicted2])

    print("Drink model warmed up successfully!")

def classify_image(model, labels, image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)

    pred = np.argmax(prediction[0])
    print("predicted: ", labels[pred])
    return labels[pred]
