import dlib
import cv2
from skimage import io
from scipy.spatial import distance

img =cv2.imread('DataSet/Em_1/EM_1_1560854518406443776.jpg')
#sp = dlib.shape_predictor('Data/shape_predictor_68_face_landmarks.dat')
#facerec = dlib.face_recognition_model_v1('Data/dlib_face_recognition_resnet_model_v1.dat')
#detector = dlib.get_frontal_face_detector()
cv2.imshow('example', img) # выводим картинку с камеры
cv2.waitKey(10000)