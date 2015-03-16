'''
    Group:
    Chris Purta
    Andrew McCall
    Stefan Bostain
    Casey Schadewitz
    
    Description: Will use half of the training data for training purposes and will use the 
    other half of the for a test set.
    '''

from skimage.io import imread
from skimage.transform import resize
import os
import numpy as np
from sklearn import svm, metrics

target_names = os.listdir(os.getcwd() + "/competition_data/train")
#Why Apple, Why you do this?
target_names.remove(".DS_Store")

#Chdir into the directory that contains all the folders of each type of plankton
os.chdir(os.getcwd() + "/competition_data/train")
#Read in the images in each folder and assign each picture to a target
#Resize and threshold the image and add to an array of images
maxPixel = 30
imageSize = maxPixel * maxPixel
training_target = []
training_data = []
test_target = []
test_data = []
i = 0
for dir in target_names:
    os.chdir(dir)
    print "Reading images from " + dir
    for file in os.listdir(".")[:len(os.listdir("."))/2]:
        image = imread(file, as_grey=True)
        image = resize(image, (maxPixel, maxPixel))
        training_data.append(image)
        training_target.append(i)
    
    for file in os.listdir(".")[len(os.listdir("."))/2:]:
        image = imread(file, as_grey=True)
        image = resize(image, (maxPixel, maxPixel))
        test_data.append(image)
        test_target.append(i)
    
    i+=1
    os.chdir("..")

print "There are a total of " + str(len(training_data)) + " samples in the training set and " + str(len(test_data)) + " test samples."
training_data = np.reshape(training_data, (len(training_data), -1))
test_data = np.reshape(test_data, (len(test_data), -1))

print "Building classifier..."
classifier = svm.SVC(gamma=0.001)
print "Fitting data..."
classifier.fit(training_data, training_target)

expected = test_target
predicted = classifier.predict(test_data)

print "Classification report for classifier %s:\n%s\n" % (
                                                          classifier, metrics.classification_report(expected, predicted))
print "Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted)