from imglib.AlexNet import AlexNet
from imglib.resnet import ResNet
from imglib import config
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from keras.applications import mobilenet_v2
from sklearn.model_selection import train_test_split
from keras.models import Sequential,Model
from keras.layers import Dense,Activation,Dropout,Flatten,GlobalAveragePooling2D
from sklearn.svm import SVC
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from sklearn.metrics import classification_report
from imutils import paths
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
os.environ['KERAS_BACKEND'] = 'tensorflow'





#model = AlexNet.build()


EPOCHS = 30
INIT_LR = 0.001
BS = 30


def poly_decay(epoch):
    maxEpochs = EPOCHS
    baseLR = INIT_LR
    power  = 1.0

    alpha = baseLR *(1-(epoch/float(maxEpochs)))**power

    return alpha

print("[]INFO Loading Images...")
data = []
labels = []
test_data = []
test_labels = []

imagePaths = sorted(list(paths.list_images(config.TRAIN_PATH)))
testPaths = sorted(list(paths.list_images(config.TEST_PATH)))

random.seed(42)
random.shuffle(imagePaths)
random.shuffle(testPaths)
                    


for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = cv2.resize(image,(224,224))
    image  = img_to_array(image)
    data.append(image)
    
    label = imagePath.split(os.path.sep)[-2]
    
    
    
    labels.append(label)
    
    
    
for imagePath in testPaths:
    image = cv2.imread(imagePath)
    image = cv2.resize(image,(224,224))
    image  = img_to_array(image)
    test_data.append(image)
        
    label = imagePath.split(os.path.sep)[-2]
        
    
        
        
    test_labels.append(label)
    
data = np.array(data,dtype="float")/255.0
labels = np.array(labels,dtype="int")

test_data = np.array(test_data,dtype="float")/255.0
test_labels = np.array(test_labels,dtype="int")


labelsY = to_categorical(labels,num_classes=3)
testY = to_categorical(test_labels,num_classes=3)





totalTrain = len(list(paths.list_images(config.TRAIN_PATH)))



#model = AlexNet.build()

base = mobilenet_v2.MobileNetV2(weights='imagenet',include_top=False)

x = base.output
x =GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x)
x=Dense(1024,activation='relu')(x)
x=Dense(512,activation='relu')(x)
preds=Dense(5,activation='softmax')(x)


model=Model(inputs=base.input,outputs=preds)

model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])







    
    
trainAug = ImageDataGenerator(rescale=1/255.0,rotation_range=30,zoom_range=0.2,
                              width_shift_range=0.1,height_shift_range=0.1,
                              shear_range=0.1,horizontal_flip=True,
                              fill_mode="nearest")

print("[INFsO] Training the model...")
callbacks = [LearningRateScheduler(poly_decay)]
H = model.fit_generator(trainAug.flow(data,labelsY,batch_size=BS),validation_data=(test_data,testY),
                        steps_per_epoch=totalTrain//BS,epochs=EPOCHS,verbose=1,callbacks=callbacks
                        )


N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")





















































































































