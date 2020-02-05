import pandas as pd
from imutils import paths
import pandas as pd
import shutil
import random
import os


path = os.getcwd() + '\\train'

base = 'D://project//split//'


folders  = ['0','1','2','3','4']

data = pd.read_csv('trainLabels_cropped.csv')




imagePath = sorted(list(paths.list_images(path)))
print(int(len(imagePath)))
print(len(data))



for i in range(int(len(imagePath))):
    
    image_name = str(imagePath[i])
    image_name = image_name.split("\\")
    image_name = image_name[3].split('.')
    name = path+"\\" +data['image'][i] + '.jpeg'
    i_path = base +'\\' +str(data['level'][i])
    shutil.copy2(name,i_path)
    print(str(data['image'][i]) + 'moved to' + str(data['level'][i]))
    
    
  






