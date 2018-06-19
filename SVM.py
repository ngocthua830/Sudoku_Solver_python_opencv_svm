import numpy as np
from sklearn import svm
from imutils import paths
import cv2
from skimage import feature
import os
from sklearn.externals import joblib

data=[]
labels=[]

for imagePath in paths.list_images("/home/thua/NMTGMT-DoAn/Sudoku/train-data"):
	make = imagePath.split("/")[-2]
	print (imagePath)
	image = cv2.imread(imagePath,0)
	image = cv2.resize(image, (200, 200))

	H = feature.hog(image, orientations = 9, pixels_per_cell=(10, 10), cells_per_block=(2,2), transform_sqrt=True)
	data.append(H)
	if make == "1":
		labels.append(1)
	elif make == "2":
		labels.append(2)
	elif make == "3":
		labels.append(3)
	elif make == "4":
		labels.append(4)
	elif make == "5":
		labels.append(5)
	elif make == "6":
		labels.append(6)
	elif make == "7":
		labels.append(7)
	elif make == "8":
		labels.append(8)
	elif make == "9":
		labels.append(9)
	elif make == "0":
		labels.append(0)

clf = svm.SVC(gamma=0.001)
clf.fit(data, np.array(labels))
joblib.dump(clf, "/home/thua/NMTGMT-DoAn/Sudoku/SVM-HOG-digit-NewDataset.sav")
#clf = joblib.load(open("/home/thua/NMTGMT-DoAn/Sudoku/SVM-HOG-digit.sav"))
for filename in os.listdir("/home/thua/NMTGMT-DoAn/Sudoku/test-data"):
	image1 = cv2.imread(os.path.join("/home/thua/NMTGMT-DoAn/Sudoku/test-data",filename), 0)
	image = cv2.resize(image1, (200, 200))
        #cv2.imwrite("output/aaa"+str(filename)+".jpg", image)
	(H, hogImage) = feature.hog(image, orientations=9, pixels_per_cell=(10,10), cells_per_block=(2,2), transform_sqrt=True, visualise=True)
	pred = clf.predict(H.reshape(1, -1))[0]
	print(pred)
	cv2.putText(image1, str(pred), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
	cv2.imwrite("output/test_"+str(filename)+"-"+str(pred)+".jpg", image1)
	
	
