import pandas as pd
train_df = pd.read_csv("data/asl_data/sign_mnist_train.csv")
valid_df = pd.read_csv("data/asl_data/sign_mnist_valid.csv")

# train_df.head()

y_train = train_df['label']
y_valid = valid_df['label']
del train_df['label']
del valid_df['label']

x_train = train_df.values
x_valid = valid_df.values

# x_train.shape
# y_train.shape
# x_valid.shape
# y_valid.shape

# import matplotlib.pyplot as plt
# plt.figure(figsize=(40,40))

# num_images = 20
# for i in range(num_images):
#     row = x_train[i]
#     label = y_train[i]
    
#     image = row.reshape(28,28)
#     plt.subplot(1, num_images, i+1)
#     plt.title(label, fontdict={'fontsize': 30})
#     plt.axis('off')
#     plt.imshow(image, cmap='gray')

# x_train.min() 
# x_train.max()

# TODO: Normalize x_train and x_valid.
x_train = x_train / 255
x_valid = x_valid / 255

import tensorflow.keras as keras
num_classes = 24

# TODO: Categorically encode y_train and y_valid.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_valid = keras.utils.to_categorical(y_valid, num_classes)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# TODO: build a model following the guidelines above.
model = Sequential()
model.add(Dense(units=512, activation='relu', input_shape=(784,)))
model.add(Dense(units = 512, activation='relu'))
model.add(Dense(units = 24, activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
# TODO: Train the model for 20 epochs.
model.fit(
    x_train, y_train, epochs=20, verbose=1, validation_data=(x_valid, y_valid)
)