import keras
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout,Flatten,Conv2D,MaxPooling2D,ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam,SGD
from keras.regularizers import l2
import numpy as np

np.random.seed(100)

class AlexNet:
    @staticmethod
    def build():
        model = Sequential()


        #layer 1
        model.add(Conv2D(filters=96,kernel_size=(11,11),input_shape=(224,224,3),strides=(4,4),padding='same',kernel_regularizer=l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        #maxpooling1
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same'))


        #layer 2
        model.add(Conv2D(filters=256,kernel_size=(11,11),strides=(1,1),padding='same',kernel_regularizer = l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        #maxpooling2
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same'))


        #layer3

        model.add(Conv2D(filters=384,kernel_size=(3,3),strides=(1,1),padding='same',kernel_regularizer=l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        #layer4

        model.add(Conv2D(filters=384,kernel_size=(3,3),strides=(1,1),padding='same',kernel_regularizer=l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        #layer5

        model.add(Conv2D(filters=256,kernel_size=(3,3),strides=(1,1),padding='same',kernel_regularizer=l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

        #fc layers

        model.add(Flatten())
        model.add(Dense(3072))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        # Layer 7
        model.add(Dense(4096))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        # Layer 8
        model.add(Dense(3))
        model.add(BatchNormalization())
        model.add(Activation('softmax'))

        model.summary()
        opt = SGD(lr =0.1,momentum=0.5,decay=1e-6)

        model.compile(loss=keras.losses.categorical_crossentropy,optimizer=opt,metrics=['accuracy'])

        return model




