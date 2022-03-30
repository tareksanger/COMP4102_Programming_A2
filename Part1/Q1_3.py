import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from CNNModel import CNNModel

# Load CIFAR10 training and testing datasets from Karas
_, (x_test,y_test) = datasets.cifar10.load_data()

# Normalize Pixel values to be between 0 & 1
x_test = x_test/255

img = np.array(x_test[1020])
img_label = y_test[1020]
print(img_label)

images_list = []
images_list.append(np.array(img))
x = np.asarray(images_list)


m_id = 5 # Model id
model_dir = os.path.dirname(os.path.abspath(__file__)) +  f'/Models/{m_id}/final_weights.hdf5'

model = CNNModel(id=m_id)()
model.load_weights(model_dir)

prediction = model.predict(x)

print("Predicted class is:",np.argmax(prediction))