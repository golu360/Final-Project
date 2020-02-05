from imutils import paths
from pathlib import Path
import numpy as np
import cv2
import os

path = os.getcwd()+"\\resized_train"




imagePath = sorted(list(paths.list_images(path)))




print(len(imagePath))

for im in imagePath:
    print("reading: " + im)
    image = cv2.imread(im)
    
    image = cv2.addWeighted(image,4, cv2.GaussianBlur(image,(0,0), 7), -4, 128)   
    cv2.imwrite(im,image)
    print(im+" resized")
           
    
