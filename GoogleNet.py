import keras
from keras.models import Model
from keras.layers import Dense,Activation,Dropout,Flatten,Conv2D,MaxPooling2D,ZeroPadding2D,Input,Concatenate,AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam,SGD
from keras.regularizers import l2
import numpy as np




class GoogleNet():
    
    
   
        
        
    
    
    @staticmethod
    def build():
        
        
        
        
        
        input_ = Input(shape=(224,224,3))
        
        
        
        
        #conv1 layer
        conv1_7x7 = Conv2D(64,(7,7),padding='same',activation='relu',strides=(2,2),name='conv1_7x7',kernel_regularizer=l2(0.0002))(input_)
        conv1_pad = ZeroPadding2D(padding=(1,1))(conv1_7x7)
        pool_1 = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='valid',name='pool_1_3x3')(conv1_pad)
        lrn_1 = BatchNormalization()(pool_1)
        
        #conv2 layer
        conv2_3x3 = Conv2D(192,(3,3),padding='same',activation='relu',name='conv2_3x3',kernel_regularizer=l2(0.0002))(lrn_1)
        lrn_2 = BatchNormalization()(conv2_3x3)
        conv2_zeropad = ZeroPadding2D(padding=(1,1))(lrn_2)
        pool_2 = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='valid',name='pool_2_3x3')(conv2_zeropad)
        
        #1x1: 3a
        incept_1x1 = Conv2D(64,(1,1),padding='same',activation='relu',kernel_regularizer=l2(0.0002),name='inception_1x1')(pool_2)
        
        #3x3: 3a
        
        incept_3x3_reduce = Conv2D(96,(1,1),padding='same',activation='relu',kernel_regularizer=l2(0.0002),name='inception_3x3_reduce')(pool_2)
        incept_3x3_reduce_pad = ZeroPadding2D(padding=(1,1))(incept_3x3_reduce)
        incept_3x3 = Conv2D(128,(3,3),padding='valid',activation='relu',kernel_regularizer=l2(0.0002),name='inception_3x3')(incept_3x3_reduce_pad)
        
        
        #5x5:3a
        incept_5x5_reduce = Conv2D(16,(1,1),padding='same',activation='relu',kernel_regularizer=l2(0.0002),name='inception_5x5_reduce')(pool_2)
        incept_5x5_pad = ZeroPadding2D(padding=(1,1))(incept_5x5_reduce)
        incept_5x5 = Conv2D(32,(5,5),padding='same',activation='relu',kernel_regularizer=l2(0.0002),name='inception_5x5')(incept_5x5_reduce)
        
        #pool:3a
        incept_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name='incept_pool')(pool_2)
        incept_pool_conv = Conv2D(32,(1,1),padding='same',activation='relu',kernel_regularizer=l2(0.0002),name='inception_pool_conv')(incept_pool)
        
        
        print('3x3:',incept_3x3.shape)
        print('5x5:',incept_5x5.shape)
        print('POOL',incept_pool_conv.shape)
        #Concatenate:3a
        concantate_3a = Concatenate(axis=-1,name='inception_output',)([incept_1x1,incept_3x3,incept_5x5,incept_pool_conv])
      
        
        #3B
                         
        #1x1
        incept_3b_1x1 = Conv2D(128,(1,1),padding='same',activation='relu',kernel_regularizer=l2(0.0002),name='3b/1x1')(concantate_3a)
        
        #3x3
        incept_3b_3x3_reduce = Conv2D(128,(1,1),padding='same',activation='relu',kernel_regularizer=l2(0.0002),name='3b/reduce_3x3')(concantate_3a)
        incept_3b_3x3_pad = ZeroPadding2D(padding=(1,1))(incept_3b_3x3_reduce)
        incept_3b_3x3 = Conv2D(192,(3,3),padding='valid',activation='relu',kernel_regularizer=l2(0.0002),name='3b/3x3')(incept_3b_3x3_pad)
        
        #5x5
        incept_3b_5x5_reduce = Conv2D(32,(1,1),padding='same',activation='relu',kernel_regularizer=l2(0.0002),name='3b/reduce_5x5')(concantate_3a)
        incept_3b_5x5_pad = ZeroPadding2D(padding=(2,2))(incept_3b_5x5_reduce)
        incept_3b_5x5  = Conv2D(96,(5,5),padding='valid',activation='relu',kernel_regularizer=l2(0.0002),name='3b/5x5')(incept_3b_5x5_pad)
        
        #maxpool
        incept_3b_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name='incept_3b_pool')(concantate_3a)
        incept_3b_pool_conv = Conv2D(64,(1,1),padding='same',activation='relu',kernel_regularizer=l2(0.0002),name='3b/pool')(incept_3b_pool)
        
        #concantate                        
        concantate_3b = Concatenate(axis=-1,name='incept_3b_output')([incept_3b_1x1,incept_3b_3x3,incept_3b_5x5,incept_3b_pool_conv])        
        
        
        
        
        
        
        
       
        
        
        
        average_pool = AveragePooling2D(pool_size=(7,7),strides=(1,1),name='average_pool')(concantate_3b)
        flatten = Flatten()(average_pool)
        dropout= Dropout(rate=0.3)(flatten)
        dense = Dense(1000,name='dense1',kernel_regularizer=l2(0.0002))(dropout)
        softmax = Activation('softmax',name='output_layer')(dense)
        
        
        googlenet = Model(inputs=input_,outputs=[softmax])
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        googlenet.compile(optimizer=sgd,loss='binary_crossentropy')
        print(googlenet.summary())
        
                    
        
        
        
        
        
        
        
        
                      
                                                    
        
        
        
        
                                                                                                       
        
        
        
        
        
        
        