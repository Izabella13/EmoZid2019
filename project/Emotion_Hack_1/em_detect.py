# импорт нужных нам нам библиотек
import cv2
import dlib
import pickle
import numpy as np
from time import sleep
from PIL import ImageFont, ImageDraw, Image
dat_path='Data/'
detector = dlib.get_frontal_face_detector() # создаем объект, который выделяет лицо прямоугольником
predictor = dlib.shape_predictor(dat_path+'shape_predictor_68_face_landmarks.dat') # загрузка(шаблона) данных обучения для точек на лице
facerec = dlib.face_recognition_model_v1(dat_path+'dlib_face_recognition_resnet_model_v1.dat') # загрузка данных обучения нейросети resnet
def get_landmarks(image): # ф-ция, расчитывающая два вида дискрипторов лица
    detections = detector(image, 1) # ф-ция выделяет лицо в прямоугольник
    for k,d in enumerate(detections): # цикл по всем найденным на изображении лицам
        shape = predictor(image, d) #возвращает координаты точек на лице
        face_descriptor = facerec.compute_face_descriptor(image, shape) # получаем 128 дискрипторов лица(дискриптор2)
        xlist = [] # списки для размещения координат точек лица
        ylist = []
        for i in range(68): # добавляем координаты точек лица в списки xlist, ylist
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
        meannp = np.asarray((shape.part(30).x, shape.part(30).y)) # берем 30 точку на носу как центральную на лице
        landmarks_vectorised = [] # создаем список для записи дискрипторов1
        for w, z in zip(xlist, ylist): # рассчитываем 68 дискрипторов(дискрипторы1)
            coornp = np.asarray((z, w)) # создаем массив из кординат векторов расстояний
            dist = np.linalg.norm(coornp - meannp) # рассчитываем расстояние от центральной точки до данной
            landmarks_vectorised.append(dist) # добавляем в список дискрипторы1
        landmarks_vectorised[:]=landmarks_vectorised[:]/landmarks_vectorised[27] # масштабируем расстояние изображения лица
    if len(detections) == 0: # добавляем значения переменных, если нет выделенных лиц лиц на экране
        landmarks_vectorised=0
        detections=0
        face_descriptor=0
    return landmarks_vectorised, detections, face_descriptor # возвращаем дискрипторы1, кол-во выделенных лиц, дискрипторы2
def em_7(em_ind):
    nn=10 # кол-во кадров для обработки для определения эмоций
    emo=['Радость', 'Удивление', 'Грусть', 'Злость', 'Отвращение', 'Презрение', 'Страх'] # эмоции для распознования
    '''
    font= ImageFont.truetype("DejaVuSans.ttf", 32) # добавляем файл шрифта ttf с поддержкой кирилицы
    f=open(dat_path+'svm_dat.dat','rb') # открываем файл svm_dat.dat для чтения как двоичный
    clf=pickle.load(f)# загружаем обученный метод svm 
    f.close # закрываем файл
    video_capture = cv2.VideoCapture(0) # подключение камеры
    video_capture.set(3, 380) # задаем размеры кадра камеры
    video_capture.set(4, 240)  
    '''
    emo_sm=0 # сглаженное значение эмоции
    alfa=0.9 # степень сглаживания
    en_count=0 # кол-во правильно воспроизведенных эмоций
    for i in range(1,nn): # цикл обработки nn кадров
        ret, frame = video_capture.read()# запуск камеры
        l_mv, det,f_n = get_landmarks(frame) # вызываем ф-цию, расчитывающую дискрипторы по изображению
        if l_mv!=0: # если есть лицо на кадре
            dat=[] # создаем список для дискрипторов
            dat.append(f_n) # записываем 128 дискрипторов для каждого лица
            em_n=clf.predict(dat) # определяем индекс эмоции в кадре с помощью метода svm
            emo_sm=alfa*(em_n-1)+(1-alfa)*emo_sm # рассчитываем сглаженный индекс эмоции
            cv2.rectangle(frame, (det[0].left(), det[0].top()), (det[0].right(), det[0].bottom()), (0, 0, 255), 2) # рисуем прямоугольник вокруг лица
            img_pil = Image.fromarray(frame) # передаем изображение для обработки в библиотеку pillow
            draw = ImageDraw.Draw(img_pil) # создаем объект, содержащий изображение
            draw.text( (det[0].left(), det[0].top()-40),  emo[round(float(emo_sm))], font=font, fill=(0,255,0)) # выводим текст кирилицей
            frame = np.array(img_pil) # возвращаем изображение с названием эмоции
            print(em_n)
            if em_n==em_ind: # если эмоция показана верна, то переходим к другой эмоции
                en_count+=1
        cv2.waitKey(1) # задержка изображений
        cv2.imshow('camera', frame) # выводим картинку с камеры
        cv2.moveWindow('camera', 600,300)
    en_count=en_count/nn # считаем долю правильно воспроизведенных эмоций
    if em_ind==7: # если все эмоции показаны правильно
        video_capture.release() # закрываем камеру
        cv2.destroyAllWindows() # закрываем все окна, открытые во время работы программы
    return en_count # возвращаем долю правильно воспроизведенных эмоций