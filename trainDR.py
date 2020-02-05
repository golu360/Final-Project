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



EPOCHS = 30
LR = 0.0001
BS = 32

def poly_decay(epoch):
    maxEpochs = EPOCHS
    baseLR = INIT_LR
    power  = 1.0
    alpha = baseLR *(1-(epoch/float(maxEpochs)))**power
    return alpha


print("[]LOADING IMAGES...")


data= []
labels = []
test_data = []
test_labels = []


imagePaths = sorted(list(paths.list_images(config.TRAIN_PATH)))
testPaths = sorted(list(paths.list_images(config.TEST_PATH)))

random.seed(42)
random.shuffle(imagePaths)
random.shuffle(testPaths)

for image in imagePaths:
    img= cv2.imread(image)
    img = cv2.resize(img,(224,224))
    img = img_to_array(img)
    data.append(img)
    
    label = image.split(os.path.sep)[-2]
    labels.append(label)

print(labels)    
    
    