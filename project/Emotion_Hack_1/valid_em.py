from time import sleep
import csv
import cv2
import os
import dlib
import glob
import math
import pickle
#import pandas
import numpy as np
def get_landmarks(image):
    detections = detector(image, 1)
    for k,d in enumerate(detections): #For all detected face instances individually
        shape = predictor(image, d) #Draw Facial Landmarks with the predictor class
        xlist = []
        ylist = []
        for i in range(0,68): #Store X and Y coordinates in two lists
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
        meannp = np.asarray((shape.part(30).x, shape.part(30).y))
        x30 = [(x - shape.part(30).x) for x in xlist]  # Calculate distance centre <-> other points in both axes
        y30 = [(y - shape.part(30).y) for y in ylist]
        landmarks_vectorised = []
        for x, y, w, z in zip(x30, y30, xlist, ylist):
            #landmarks_vectorised.append(w)
            #landmarks_vectorised.append(z)
            coornp = np.asarray((z, w))
            dist = np.linalg.norm(coornp - meannp)
            landmarks_vectorised.append(dist)
        landmarks_vectorised[:]=landmarks_vectorised[:]/landmarks_vectorised[27]
            #landmarks_vectorised.append((math.atan2(y, x) * 360) / (2 * math.pi))
    if len(detections) == 0:
        xlist=0
        ylist=0
        meannp=np.asarray((0, 0))
        landmarks_vectorised=0
    return xlist,ylist,meannp,landmarks_vectorised
f=open('svn_dat.dat','rb')
clf=pickle.load(f)
f.close
predictor_path = 'shape_predictor_68_face_landmarks.dat'
faces_folder_path = 'val'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
win = dlib.image_window()
im_par=[]

for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
    #print("Processing file: {}".format(f))
    img = dlib.load_rgb_image(f)
    l_mx,l_my,mn,l_mv  = get_landmarks(img)
    if l_mx!=0:
        for x,y in zip(l_mx,l_my) :  # There are 68 landmark points on each face
            cv2.circle(img, (int(x),int(y)), 1, (255, 0, 0), 1)
            #cv2.line (img,(int(mn[0]),int(mn[1])),(int(x),int(y)),(0,255,0),1)
        cv2.circle(img, (int(mn[0]),int(mn[1])), 1, (0, 0, 255), 1)
        dat=[]
        dat.append(l_mv)
        em_n=clf.predict(dat)
        print(em_n, f)
    win.clear_overlay()
    win.set_image(img)
    sleep(1)  

