from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn import preprocessing
import numpy as np
import pandas as pd
from collections import Counter

#download the mnist training set
dataset = pd.read_csv('C:/Users/Downloads/d/sandbox/mnist-train.csv')
features = dataset.iloc[:,1:].values
labels = dataset[[0]].values.ravel()

list_hog_fd = []
for feature in features:
	fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
	list_hog_fd.append(fd)

hog_features = np.array(list_hog_fd, 'float64')

print "Count of digits in dataset", Counter(labels)

clf = LinearSVC()
clf.fit(hog_features, labels)
joblib.dump(clf, "digits_cls.pkl", compress=3)
