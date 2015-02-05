
'''
Group:
Chris Purta
Andrew McCall
Stefan Bostain
Casey Schadewitz
'''

from skimage.io import imread
from skimage.transform import resize
import os
import glob
import numpy as np
import pandas as pd
from sklearn import svm

# get the classnames from the directory structure
directory_names = list(set(glob.glob(os.path.join("competition_data","train", "*"))\
 ).difference(set(glob.glob(os.path.join("competition_data","train","*.*")))))

print(directory_names)
