from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K

class SimpleNet:
    @staticmethod
    def build(width, height, depth, classes, reg):
        model = Sequential()
        inputShape = (height, width, depth)
        
        if K.image_data_format() == 'channels_first':
            inputShape = (depth, height, width)
            
        #CONV => RELU => POOL Layers
        model.add(Conv2D(64, (11, 11), input_shape = inputShape, padding='same', kernel_regularizer=reg))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(128, (5, 5), padding="same", kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
  
        model.add(Conv2D(256, (3, 3), padding="same", kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
  
        # FC => RELU Layer
        model.add(Flatten())
        model.add(Dense(512, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))
  
        if classes == 1:
            model.add(Dense(1))
            model.add(Activation('sigmoid')) 
        else:
            model.add(Dense(classes))
            model.add(Activation('softmax'))
        
        return model