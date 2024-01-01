from tensorflow import keras

model = keras.models.load_model('asl_model')
# model.summary()

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing import image as image_utils
import numpy as np

def predict_letter(file_path):
    # Show image
    image = mpimg.imread(file_path)
    plt.imshow(image, cmap='gray')
    # Load and scale image
    image = image_utils.load_img(file_path, color_mode="grayscale", target_size=(28,28))
    # Convert to array
    image = image_utils.img_to_array(image)
    # Reshape image
    image = image.reshape(1,28,28,1) 
    # Normalize image
    image = image / 255
    # Make prediction
    prediction = model.predict(image)
    # Convert prediction to letter
    alphabet = "abcdefghiklmnopqrstuvwxy"
    predicted_letter = alphabet[np.argmax(prediction)]
    # Return prediction
    return predicted_letter   