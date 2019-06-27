from time import sleep
import csv
import cv2
import os
import dlib
import glob
import math
#import pandas
import numpy as np
def get_landmarks(image):
    detections = detector(image, 1)
    for k,d in enumerate(detections): #For all detected face instances individually
        shape = predictor(image, d) #Draw Facial Landmarks with the predictor class
        face_descriptor = facerec.compute_face_descriptor(image, shape)
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
        face_descriptor=0
    return xlist,ylist,meannp,landmarks_vectorised,face_descriptor
nn=0
predictor_path = 'shape_predictor_68_face_landmarks.dat'
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
faces_folder_path = 'Em_1'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
#win = dlib.image_window()
im_par=[]
im_par1=[]
for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
    #print("Processing file: {}".format(f))
    img = dlib.load_rgb_image(f)
    l_mx,l_my,mn,l_mv, f_d  = get_landmarks(img)
    if l_mx!=0:
        im_par.append(l_mv)
        im_par1.append(f_d)
        nn+=1
        """
        cv2.circle(img, (int(mn[0]),int(mn[1])), 1, (0, 0, 255), 2)
        for x,y in zip(l_mx,l_my) :  # There are 68 landmark points on each face
            cv2.circle(img, (int(x),int(y)), 1, (255, 0, 0), 2)
            cv2.line (img,(int(mn[0]),int(mn[1])),(int(x),int(y)),(0,255,0),1)
    win.clear_overlay()
    win.set_image(img)
    #print(l_mx,l_my)
    #sleep(0.5)  
    """
f = open(faces_folder_path +'.txt', 'w', newline='')
writer = csv.writer(f, delimiter=',')
for it in im_par:
    #if isinstance(it, list):
    writer.writerow(it)
f.close()
f = open(faces_folder_path +'_128.txt', 'w', newline='')
writer = csv.writer(f, delimiter=',')
for it in im_par1:
    #if isinstance(it, list):
    writer.writerow(it)
f.close()
print(nn)

'''
for op in it:
        f.write("%f," % op)
    f.write("\n")
 '''