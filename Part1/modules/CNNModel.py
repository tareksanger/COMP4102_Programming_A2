from abc import abstractclassmethod
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

# Custom activation function
from keras.layers import Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects



class CNNModel:
    def __init__(self, id):
        self.id = id
        
    def __call__(self):
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(3, 3),padding='valid', activation='relu', input_shape=(32, 32, 3)))
        model.add(MaxPooling2D((2, 2), strides=2,padding='valid'))

        if self.id == 1:
            model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
            model.add(MaxPooling2D((2, 2), strides=2,padding='valid'))
        
        elif self.id == 2:
            model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
            model.add(MaxPooling2D((3, 3), strides=2,padding='valid'))

        elif self.id == 3:
            model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
            model.add(MaxPooling2D((2, 2), strides=2,padding='valid'))

            model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
            model.add(MaxPooling2D((2, 2), strides=2,padding='valid'))

        elif self.id == 4:
            model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
            model.add(MaxPooling2D((3, 3), strides=2,padding='valid'))

            model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
            model.add(MaxPooling2D((3, 3), strides=2,padding='valid'))


        elif self.id == 5:
            model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
            model.add(MaxPooling2D((2, 2), strides=2,padding='valid'))

            model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
            model.add(MaxPooling2D((3, 3), strides=2,padding='valid'))
 
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(10, activation='softmax'))
    
        model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

        return model




class CNNModel_5:
    def __init__(self):
        pass
    def __call__(self):
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(3, 3),padding='valid', activation='relu', input_shape=(32, 32, 3)))
        model.add(MaxPooling2D((2, 2), strides=2,padding='valid'))

        # -----------------------------------------------------------

        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D((3, 3), strides=2,padding='valid'))

        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=2,padding='valid'))

        # -----------------------------------------------------------
        
        model.add(Flatten())
        
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(10, activation='softmax'))
        
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        return model