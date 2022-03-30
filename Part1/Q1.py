import os
import tensorflow as tf
from tensorflow.keras import datasets
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from CNNModel import CNNModel


# Load CIFAR10 training and testing datasets from Karas
(x_train, y_train), (x_test,y_test) = datasets.cifar10.load_data()

# Normalize Pixel values to be between 0 & 1
x_train, x_test = x_train/255, x_test/255

# # Normalize Pixel values to be between 0 & 1
y_train, y_test = y_train.flatten(), y_test.flatten()

print(y_test[0])

# print(y_train)

print("X Train: ", x_train.shape)
print("Y Train: ", y_train.shape)
print("X Test: ", x_test.shape)
print("Y Test: ", y_test.shape)

'''
    Display some random training images in a 25x25 grid.

'''
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i])
    # The CIFAR labels happen to be arrays,
    # which is why you need the extra index
    plt.xlabel(class_names[y_train[i]])


save_dir = os.path.dirname(os.path.abspath(__file__)) + '/Models'

for m_id in range(1,6):

    print(f'# Model {m_id} ', "-" * 60)

    # Initialize the model using the CNNModel Class
    model = CNNModel(id=m_id)()

    # Display the Summary
    model.summary()

    # Plot the Model and save it to the Models directory where all saves can be found
    plot_model(model=model, to_file=os.path.join(save_dir, f'{m_id}/model.png'))

    # Train the model for 15 epochs
    history = model.fit(x_train, y_train, batch_size=128, validation_data=(x_test, y_test), epochs=15)

    # Plot Accuracy and Loss
    CNNModel.plot_history(history, save_dir, m_id, show=False)

    # Save the History and the Model
    CNNModel.save_model(model, history, save_dir)

    model.evaluate(x_test,y_test)
    tf.keras.backend.clear_session()

plt.show()