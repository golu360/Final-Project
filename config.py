import os

ORIG_PATH = os.getcwd()+'\\split'


TRAIN_PATH = os.path.sep.join([ORIG_PATH,'training'])
VAL_PATH = os.path.sep.join([ORIG_PATH,'validation'])
TEST_PATH = os.path.sep.join([ORIG_PATH,'test'])

TRAIN_SPLIT = 0.8
VAL_SPLIT= 0.1
