#Импорт нужных библиотек
from time import sleep
import csv
import cv2
import os
import dlib
import glob
import math
import numpy as np
def get_landmarks(image): #Функция, рассчитывающая два вида дискрипторов лица
    detections = detector(image, 1) #Функция выделяет лицо в прямоугольник
    for k,d in enumerate(detections): #Цикл по всем найденым на изображении лицам
        shape = predictor(image, d) #Рисуем лицевые ориентиры с помощью класса предиктора(Возврат координат точек на лице)
        face_descriptor = facerec.compute_face_descriptor(image, shape) #Получаем дискрипторы лица
        xlist = [] #Список для размещения координат точек лица по оси X
        ylist = [] #Список для размещения координат точек лица по оси Y
        for i in range(0,68): #Сохраняем координаты X и Y в двух списках
            xlist.append(float(shape.part(i).x)) #Список для X
            ylist.append(float(shape.part(i).y)) #Список для Y
        meannp = np.asarray((shape.part(30).x, shape.part(30).y)) #Берем 30 точку на носу как центральную на лице
        landmarks_vectorised = [] #Создаем список для записи дискрипторов1
        for w, z in zip(xlist, ylist): #Расчитываем дискрипторы 1
            coornp = np.asarray((z, w)) #Создаем масив из координат векторов расстояений 
            dist = np.linalg.norm(coornp - meannp) #Рассчитываем расстояние от центральной точки до данной
            landmarks_vectorised.append(dist) #Добавляем в список дискрипторы1
        landmarks_vectorised[:]=landmarks_vectorised[:]/landmarks_vectorised[27] #Масштабируем параметры изображения
    if len(detections) == 0: #Добавляем значения переменных, если нет выделенных лиц на экране
        xlist=0
        ylist=0
        meannp=np.asarray((0, 0))
        landmarks_vectorised=0
        face_descriptor=0
    return xlist,ylist,meannp,landmarks_vectorised,face_descriptor #Возвращаем дискрипторы1, количество выделенных лиц
nn=0 #Счетчик для количества файлов с распознанными лицами
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat') #Извлекаем 128 дискрипторов
faces_folder_path = 'Em_1'  #Расположение изображений с заданной эмоцией 
detector = dlib.get_frontal_face_detector() #Создаем объект который может выделять лица в прямоугольник 
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') #Загрузка данных для извлечения 68 точек лица
im_par=[] #Список для записи 68 дискрипторов
im_par1=[] #Список для записи 128 дискрипторов
for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")): #Просмотр всего каталога, составление списка файлов в этом каталоге и начало его прохождения
    img = dlib.load_rgb_image(f)
    l_mx,l_my,mn,l_mv, f_d  = get_landmarks(img)
    if l_mx!=0:
        im_par.append(l_mv)
        im_par1.append(f_d)
        nn+=1
f = open(faces_folder_path +'.txt', 'w', newline='') #Открываем текстовый файл 68 дискрипторов на запись
writer = csv.writer(f, delimiter=',') #Создаем объект который работает с csv файлами 
for it in im_par:
    writer.writerow(it) #Запись каждого из 68 дискрипторов
f.close()
f = open(faces_folder_path +'_128.txt', 'w', newline='') #Открываем текстовый файл 128 дискрипторов на запись
writer = csv.writer(f, delimiter=',') #Создаем объект который работает с csv файлами 
for it in im_par1:
    writer.writerow(it) #Запись каждого из 128 дискрипторов
f.close()
print(nn)