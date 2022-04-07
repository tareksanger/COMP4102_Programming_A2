from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
import os
import pandas as pd
import matplotlib.pyplot as plt

# https://datascience.stackexchange.com/questions/55545/in-cnn-why-do-we-increase-the-number-of-filters-in-deeper-convolution-layers-fo

class CNNModel:
    def __init__(self, id):
        self.id = id
        
    def __call__(self):

        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        model.add(MaxPooling2D((2, 2)))

        if self.id == 1:
            
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D((2, 2)))
        
        elif self.id == 2:
            model.add(Conv2D(64, (3, 3), kernel_regularizer=regularizers.l2(l=0.01), activation='relu'))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D((2, 2)))

        elif self.id == 3:
            model.add(Conv2D(64, (3, 3), kernel_regularizer=regularizers.l2(l=0.01), activation='relu'))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D((3, 3)))

        elif self.id == 4:
            model.add(Conv2D(64, (3, 3), kernel_regularizer=regularizers.l2(l=0.01), activation='relu'))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D((2, 2)))

            model.add(Conv2D(128, (3, 3), kernel_regularizer=regularizers.l2(l=0.01), padding='same', activation='relu'))
            model.add(Conv2D(128, (3, 3), activation='relu'))
            model.add(MaxPooling2D((2, 2)))


        elif self.id == 5:
            model.add(Conv2D(64, (3, 3), kernel_regularizer=regularizers.l2(l=0.01), padding='same', activation='relu'))
            model.add(Conv2D(128, (3, 3), activation='relu'))
            model.add(MaxPooling2D((3, 3)))

            model.add(Conv2D(64, (3, 3), kernel_regularizer=regularizers.l2(l=0.01), padding='same', activation='relu'))
            model.add(Conv2D(128, (3, 3), activation='relu'))
            model.add(MaxPooling2D((2, 2)))
 
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        # model.add(Dropout(0.25))
        model.add(Dense(10, activation='softmax'))
    
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        return model
    
    @staticmethod
    def save_model(model, history, dir, m_id):
        pd.DataFrame(history.history).to_hdf(os.path.join(dir, f"{m_id}/history.h5"), 'history')
        model.save(os.path.join(dir, f'{m_id}/model.h5'))
        model.save_weights(os.path.join(dir, f'{m_id}/final_weights.hdf5'), overwrite=True)

    @staticmethod
    def plot_history(history, save_dir, m_id, show=True):
        plt.figure(figsize=(15,4), constrained_layout=True)
        plt.tight_layout(pad=20)
        plt.subplot(1, 2, 1)
        plt.plot(history.history["loss"], label="loss")
        plt.plot(history.history["val_loss"], label="val_loss")
        plt.xlabel("number of epochs")
        plt.ylabel("loss")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(history.history["accuracy"], label="accuracy")
        plt.plot(history.history["val_accuracy"], label="val_accuracy")
        plt.xlabel("number of epochs")
        plt.ylabel("accuracy")
        plt.legend()
        plt.suptitle(f"Model {m_id}")
        plt.savefig(os.path.join(save_dir, f"{m_id}/training_history.png"))
        if show:
            plt.show()

    @staticmethod
    def categories(): 
        return ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']