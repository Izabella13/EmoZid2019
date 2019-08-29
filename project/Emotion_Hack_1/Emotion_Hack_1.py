from time import sleep
from pygame import mixer
from em_detect import em_7
import glob, cv2, dlib
from PIL import ImageFont, ImageDraw, Image
import pickle
import numpy as np
nn=20 # кол-во кадров для обработки для определения эмоций
P_em=0.5
#Функция отвечающая за вывод окна с картинками
mp3_pic_path='mp3_pic/'
dat_path='Data/'
detector = dlib.get_frontal_face_detector() # создаем объект, который выделяет лицо прямоугольником
predictor = dlib.shape_predictor(dat_path+'shape_predictor_68_face_landmarks.dat') # загрузка(шаблона) данных обучения для точек на лице
facerec = dlib.face_recognition_model_v1(dat_path+'dlib_face_recognition_resnet_model_v1.dat') # загрузка данных обучения нейросети resnet
font= ImageFont.truetype("DejaVuSans.ttf", 21) # добавляем файл шрифта ttf с поддержкой кирилицы
emotions=['Радость', 'Удивление', 'Грусть', 'Злость', 'Отвращение', 'Презрение', 'Страх'] #Список эмоций, которые будут выводиться в консоль
sound=['1.mp3', '2.mp3', '3.mp3', '4.mp3', '5.mp3', '6.mp3', '7.mp3'] #Звуковые дорожки, содержащие обращение к каждой из эмоций
pictures=['1.jpg','2.jpg','3.jpg','4.jpg','5.jpg','6.jpg','7.jpg'] #Изображения, содержащие подсказки к каждой из эмоций
f=open(dat_path+'svm_dat.dat','rb') # открываем файл svm_dat.dat для чтения как двоичный
clf=pickle.load(f)# загружаем обученный метод svm 
f.close # закрываем файл
mixer.init() #Инициализация mixer из pygame
video_capture = cv2.VideoCapture(0) # подключение камеры
video_capture.set(3, 380) # задаем размеры кадра камеры   380
video_capture.set(4, 240) 

for i, em in enumerate(emotions): #Начало цикла, проходящему по всему списку emotions, содержащий и индекс i, и его содержание
    emotion_result=0 #Переменная, содержащия в себе процент того, насколько оператор показывает похожую эмоцию
    while emotion_result<P_em: #Начало цикла, который будет просить показать эмоцию, попробовать еще раз, илил перейти к следующей в зависимости от схожести с примером
        img =cv2.imread(mp3_pic_path+pictures[i])
        cv2.waitKey(10) # задержка изображений
        cv2.imshow('example', img) # выводим картинку с камеры
        cv2.moveWindow('example', 100,500)
        mixer.music.load(mp3_pic_path+sound[i]) #Загрузка звуковой дорожки, содержащихся в списке sound, с последующим ее проигрыванием
        mixer.music.play()
        print("Покажите нам "+ em) #Выводит в консоль просьбу показать эмоцию из списка emotions
        for j in range(nn-1): # цикл обработки nn кадров
            ret, frame = video_capture.read()# запуск камеры
            detections = detector(frame, 1) # ф-ция выделяет лицо в прямоугольник
            if len(detections) == 0: # добавляем значения переменных, если нет выделенных лиц лиц на экране
                    em_n=0
                    print(em_n)
            for k,d in enumerate(detections): # цикл по всем найденным на изображении лицам
                shape = predictor(frame, d) #возвращает координаты точек на лице
                face_descriptor = facerec.compute_face_descriptor(frame, shape) # получаем 128 дискрипторов лица
                dat=[] # создаем список для дискрипторов
                dat.append(face_descriptor) # записываем 128 дискрипторов для каждого лица
                em_n=clf.predict(dat)
                cv2.rectangle(frame, (detections[0].left(), detections[0].top()), (detections[0].right(), detections[0].bottom()), (0, 0, 255), 2) # рисуем прямоугольник вокруг лица
                for k in range(0,17): #   
                    cv2.circle(frame, (shape.part(k).x, shape.part(k).y), 1, (255,0,255), 1)
                for k in range(18,26): #  
                    cv2.circle(frame, (shape.part(k).x, shape.part(k).y), 1, (255,255,255), 1)
                for k in range(27,36): #    
                    cv2.circle(frame, (shape.part(k).x, shape.part(k).y), 1, (0,255,0), 1)
                for k in range(36,48): #    
                    cv2.circle(frame, (shape.part(k).x, shape.part(k).y), 1, (255,0,0), 1)
                for k in range(49,68): #
                    cv2.circle(frame, (shape.part(k).x, shape.part(k).y), 1, (0,0,255), 1)
                cv2.circle(frame, (shape.part(30).x, shape.part(30).y), 1, (0,255,255), 1)    
                print(em_n)
                if em_n==i+1: # если эмоция показана верна, то переходим к другой эмоции
                    emotion_result+=1
                    img_pil = Image.fromarray(frame) # передаем изображение для обработки в библиотеку pillow
                    draw = ImageDraw.Draw(img_pil) # создаем объект, содержащий изображение
                    draw.text( (detections[0].left(), detections[0].top()-25),  emotions[int(em_n)-1], font=font, fill=(0,255,0)) # выводим текст кирилицей
                    frame = np.array(img_pil) # возвращаем изображение с названием эмоции
                else:
                    img_pil = Image.fromarray(frame) # передаем изображение для обработки в библиотеку pillow
                    draw = ImageDraw.Draw(img_pil) # создаем объект, содержащий изображение
                    draw.text( (detections[0].left(), detections[0].top()-25),  emotions[int(em_n)-1], font=font, fill=(0,0,255)) # выводим текст кирилицей
                    frame = np.array(img_pil) # возвращаем изображение с названием эмоции
            cv2.waitKey(10) # задержка изображений
            cv2.imshow('camera', frame) # выводим картинку с камеры
            cv2.moveWindow('camera', 600,500)
        emotion_result=emotion_result/nn # считаем долю правильно воспроизведенных эмоций
        if emotion_result>=P_em: #Условие, если проецнт совпадения с эмоцией оператора больше или равен P_em процентам
            mixer.music.load(mp3_pic_path+'8.mp3') #Запускает звуковую дорожку, содержащую в себе похвалу оператора
            mixer.music.play()
            print("Молодец") #Выводит в консоль "Молодец"
            sleep(2)
        else:
            mixer.music.load(mp3_pic_path+'9.mp3') #Запускает звуковую дорожку, содержащую в себе прозьбу оператора попробовать еще
            mixer.music.play()
            print("Попробуй еще") #Выводит в консоль "Попробуй еще"
            sleep(2)
video_capture.release() # закрываем камеру
cv2.destroyAllWindows() # закрываем все окна, открытые во время работы программы            