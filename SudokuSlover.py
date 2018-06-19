# -*- coding: utf-8 -*-
import numpy as np 
import cv2
import matplotlib.pyplot as plt
from sklearn import svm
from skimage import feature
from sklearn.externals import joblib

#################################################
"""
Created on Sun May 20 02:02:50 2018

@author: Minh Thuan
"""
def findNextShellToFill(grid,i,j):
    for x in range(i,9):
        for y in range(j,9):
            if grid[x][y] == 0:
                return x,y
    for x in range(0,9):
        for y in range(0,9):
            if grid[x][y] == 0:
                return x,y 
    return -1,-1
    

def isValid(grid,i,j,k):
    checkRow = all(k != grid[x][j] for x in range(9))
    if checkRow:
        checkColumn = all(k != grid[i][y] for y in range(9))
        if checkColumn:
            #kiểm tra trong ô
            secTopX, secTopY = 3 *(i//3), 3 *(j//3) 
            for x in range(secTopX, secTopX+3):
                for y in range(secTopY, secTopY+3):
                    if grid[x][y] == k:
                        return False
            return True
    return False        

def solveSudoku(grid, i, j):
    i, j = findNextShellToFill(grid,i,j)
    if i == -1:
        return True
    for k in range(1,10):
            if isValid(grid,i,j,k):
                grid[i][j] = k
                if solveSudoku(grid,i,j):
                    return True
            grid[i][j] = 0
    return False

####################################################
clf = svm.SVC(gamma=0.001)
clf = joblib.load(open("/home/thua/NMTGMT-DoAn/Sudoku/SVM-HOG-digit-NewDataset.sav"))
cv2.imshow("lastest result", np.zeros(shape=(5,2)))
def process(im):
        #im = cv2.imread('sudoku_web1.jpg')
        #im = cv2.imread('WP_20180619_007.jpg')
        #im = cv2.imread('DSC_0049.JPG')
        im_height, im_width, _ = im.shape 
        fig, axes = plt.subplots(2,3)
        #####FIND SUDOKU BOX##########
        imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        #ret, thresh = cv2.threshold(imgray, 127, 255, 0)
        ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_OTSU)
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(contours, key=lambda ctr: cv2.contourArea(ctr), reverse=True)
        if len(cnts) > 1:
                sudoku_box_img = cv2.drawContours(im.copy(), cnts, 1, (0, 255, 0), 2)
                if im_width > 1000 or im_height > 1000:
                        cv2.imshow('sudoku_box_img', cv2.resize(sudoku_box_img, (im_width/4, im_height/4)))
                else:
                        cv2.imshow('sudoku_box_img', sudoku_box_img)
                axes[0,0].imshow(sudoku_box_img)         
        
                x, y, w, h = cv2.boundingRect(cnts[1])
                sudoku_box = im[y:y+h, x:x+w]
                sudoku_box_coor = (x, y) #Toa do khung sudoku dung de ve lai
                sudoku_box_size = (w, h)
                cv2.imwrite("im.jpg", sudoku_box)
                ################################
                height, width, channels = sudoku_box.shape 
                print(height, width)
                #if (height > 700 and width > 700):
                #        sudoku_box = cv2.resize(sudoku_box, (600, 600))
                #################
                sudoku_box1 = sudoku_box.copy()
                ###
                ###Lam day duong net (dilation)
                #kernel = np.ones((5,5),np.uint8)
                #sudoku_box_gray =  cv2.dilate(cv2.bitwise_not(sudoku_box_gray), kernel, iterations=1)
                #sudoku_box_gray = cv2.bitwise_not(sudoku_box_gray)
                #cv2.imshow("sudoku_box_gray_dilation", sudoku_box_gray)
                kernel = np.ones((1,20), np.uint8)  # note this is a horizontal kernel
                sudoku_box2 = cv2.dilate(sudoku_box1, kernel, iterations=7)
                sudoku_box3 = cv2.erode(sudoku_box2, kernel, iterations=1) 
                sudoku_box1 = cv2.addWeighted(sudoku_box1, 1.5, sudoku_box3, -0.5, 0)
                kernel = np.ones((20,1), np.uint8)  
                sudoku_box2= cv2.dilate(sudoku_box1, kernel, iterations=7)
                sudoku_box3 = cv2.erode(sudoku_box2, kernel, iterations=1) 
                sudoku_box1 = cv2.addWeighted(sudoku_box1, 1.5, sudoku_box3, -0.5, 0)
                ####Gaussian
                #sudoku_box1 = cv2.GaussianBlur(sudoku_box1,(3,3),0)
                #sudoku_box_gray = cv2.addWeighted(sudoku_box_gray, 1.5, sudoku_box_gray1, -0.5, 0)
                #####cv2.imshow("sudoku_box1", sudoku_box1)
                ####
                #####Hough Tranform
                sudoku_box2 = sudoku_box1.copy()
                gray = cv2.cvtColor(sudoku_box2, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150, apertureSize=3)
                lines = cv2.HoughLinesP(edges,1,np.pi/180, 10, 0, 0)
                if not (lines is None):
                        for line in lines:
                                for x1, y1, x2, y2 in line:
                                        cv2.line(sudoku_box2, (x1, y1), (x2, y2), (255, 0, 0), 1)
                if im_width > 1000 or im_height > 1000:
                        cv2.imshow("hough tranform", cv2.resize(sudoku_box2, (width/4, height/4)))
                else:
                        cv2.imshow("hough tranform", sudoku_box2)
                ######
                sudoku_box_gray = cv2.cvtColor(sudoku_box2, cv2.COLOR_BGR2GRAY)
                ###############
                ret, thresh = cv2.threshold(sudoku_box_gray, 127, 255, 0)
                #thresh = cv2.adaptiveThreshold(sudoku_box_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
                #ret, thresh = cv2.threshold(sudoku_box_gray, 0, 255, cv2.THRESH_OTSU)
                ##############
                if im_width > 1000 or im_height > 1000:
                        height, width = thresh.shape 
                        cv2.imshow('thresh', cv2.resize(thresh, (width/4, height/4)))
                else:
                        cv2.imshow('thresh', thresh)
                #axes[0,2].imshow(thresh)
                _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cnts2 = sorted(contours, key=lambda ctr: (cv2.boundingRect(ctr)[0]*10000) + cv2.boundingRect(ctr)[1])
                number_boxs_img = sudoku_box.copy()
                listnumber_box = [] #danh sach cac o trong sudoku
                i = 0
                for cnt in cnts2:
                        i = i+1
                        x, y, w, h = cv2.boundingRect(cnt)
                        area = w*h
                        #print(area, sudoku_box_size[0], sudoku_box_size[1])
                        if area > sudoku_box_size[0]*sudoku_box_size[1]/81 - sudoku_box_size[0]*sudoku_box_size[1]*0.01 and area < sudoku_box_size[0]*sudoku_box_size[1]/81 + sudoku_box_size[0]*sudoku_box_size[1]*0.01 and abs((h*1.0)/(w*1.0))>0.85 and abs((h*1.0)/(w*1.0))<1.15:
                                cv2.rectangle(number_boxs_img, (x, y), (x+w, y+h), (0, 255, 0), 5) 
                                number_box = sudoku_box[y:y+h, x:x+w]
                                #cv2.imwrite("/home/thua/NMTGMT-DoAn/Sudoku/output/"+str(i)+".jpg", number_box)
                                listnumber_box.append((number_box, x, y, h, w))
                if im_width > 1000 or im_height > 1000:
                        height, width, _ = number_boxs_img.shape 
                        cv2.imshow('number_boxs_img', cv2.resize(number_boxs_img, (width/4, height/4)))
                else:
                        cv2.imshow('number_boxs_img', number_boxs_img)
                #axes[0,1].imshow(number_boxs_img)

                ################################
                box_num = 0
                listcolumn=[] #danh sach cac cot cua sudoku
                listTmp=[]
                listnumber_box_tmp=[]
                for number_box in listnumber_box:
                        box_num = box_num+1
                        if box_num < 9:
                                listTmp.append(number_box)
                        else:
                                listTmp.append(number_box)
                                listcolumn.append(listTmp)
                                listTmp=[]
                                box_num=0
                for column in listcolumn:
                        column.sort(key=lambda x: x[2])
                for column in listcolumn:
                        for number_box in column:
                                listnumber_box_tmp.append(number_box)
                listnumber_box = listnumber_box_tmp
                ################################
                print(len(listnumber_box))
                if len(listnumber_box) == 81:
                        i = 0

                        sudoku_values = np.zeros((9,9))
                        su_x = -1
                        su_y = 0
                        for number_box in listnumber_box:
                                su_x = su_x + 1
                                if su_x > 8:
                                        su_y = su_y+1
                                        su_x = 0
                                imgray = cv2.cvtColor(number_box[0], cv2.COLOR_BGR2GRAY)
                                ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_OTSU)
                                axes[1,0].imshow(thresh)
                                _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                                cnts = sorted(contours, key=lambda ctr: cv2.contourArea(ctr), reverse=True)
                                x, y, w, h = cv2.boundingRect(cnts[0])
                                if w*h > 20 and len(cnts) > 1:
                                        x1,y1,w1,h1 = cv2.boundingRect(cnts[1])
                                        print(x, y, w, h, w1*h1, w*h*0.05)
                                        if w1*h1 > w*h*0.05:
                                                k1 = 0.15
                                                k2 = 0.25
                                                img = number_box[0][int(y+h*k1):int(y+h-h*k1), int(x+w*k2):+int(x+w-w*k2)]
                                                image = cv2.cvtColor(cv2.resize(img, (200, 200)), cv2.COLOR_RGB2GRAY)
                                                (H, hogImage) = feature.hog(image, orientations=9, pixels_per_cell=(10,10), cells_per_block=(2,2), transform_sqrt=True, visualise=True)
                                                pred = clf.predict(H.reshape(1, -1))[0]
                                                #cv2.imwrite("output/test_"+str(i)+"-"+str(pred)+".jpg", img)
                                                sudoku_values[su_x][su_y] = str(pred)
                                i = i+1
                        print(sudoku_values)
                        #slove = raw_input('Do you want to slove 1.yes: ') 
                        #if int(slove) == 1:
                        if True:
                                input=sudoku_values.copy()
                                solveSudoku(input,0,0)
                                print (input)
                                
                                
                        #####################
                        #Ve ket qua vao anh goc
                        #####################
                        sloved = im
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        column_num = 0
                        box_num = -1
                        s = True
                        for i in range(0, 9):
                                for j in range(0, 9):
                                        if int(input[j][i]) == 0:
                                                s = False
                        if s == True:
                                for i in range(0, 9):
                                        for j in range(0, 9):
                                                number_box = listnumber_box[i*9+j]
                                                num = int(input[j][i])
                                                if (int(sudoku_values[j][i]) == 0):
                                                        cv2.putText(sloved,str(num),(int(sudoku_box_coor[0]+number_box[1]+number_box[4]/2.5),int(sudoku_box_coor[1]+number_box[2]+number_box[3]/1.5)), font, 0.7,(0,0,255),2,cv2.LINE_AA)
                                if im_width > 1000 or im_height > 1000:
                                        cv2.imshow('sudoku_box_img', cv2.resize(sloved, (im_width/4, im_height/4)))
                                        cv2.imshow('lastest result', cv2.resize(sloved, (im_width/4, im_height/4)))
                                else:
                                        cv2.imshow("lastest result",sloved)
                                        cv2.imshow("sudoku_box_img",sloved)
                                cv2.imwrite("1!sloved.jpg", sloved)

cap = cv2.VideoCapture(0)
i = 0
while True:
#if True:
        ret, im = cap.read()
        if ret == True:
                i = i + 1
                if i == 2:
                        i = 0
                        process(im)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                #cv2.waitKey(0)
cap.release()
#plt.show()



