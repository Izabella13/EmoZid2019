import dlib, cv2
from scipy.spatial import distance
dat_path='Data/'
detector = dlib.get_frontal_face_detector() # создаем объект, который выделяет лицо прямоугольником
predictor = dlib.shape_predictor(dat_path+'shape_predictor_68_face_landmarks.dat') # загрузка(шаблона) данных обучения для точек на лице
facerec = dlib.face_recognition_model_v1(dat_path+'dlib_face_recognition_resnet_model_v1.dat') # загрузка данных обучения нейросети resnet
video_capture = cv2.VideoCapture(0) # подключение камеры
video_capture.set(3, 360) # задаем размеры кадра камеры   160x120
video_capture.set(4, 240) 
img =cv2.imread('Nikita.jpg')
detections = detector(img, 1) # ф-ция выделяет лицо в прямоугольник
for k,d in enumerate(detections): # цикл по всем найденным на изображении лицам
    shape = predictor(img, d) #возвращает координаты точек на лице
    face_descriptor_img = facerec.compute_face_descriptor(img, shape) # получаем 128 дискрипторов лица
    cv2.rectangle(img, (detections[0].left(), detections[0].top()), (detections[0].right(), detections[0].bottom()), (0, 0, 255), 1) # рисуем прямоугольник вокруг лица
    for k in range(0,17): #   
        cv2.circle(img, (shape.part(k).x, shape.part(k).y), 1, (255,0,255), 1)
    for k in range(18,26): #  
        cv2.circle(img, (shape.part(k).x, shape.part(k).y), 1, (255,255,255), 1)
    for k in range(27,36): #    
        cv2.circle(img, (shape.part(k).x, shape.part(k).y), 1, (0,255,0), 1)
    for k in range(36,48): #    
        cv2.circle(img, (shape.part(k).x, shape.part(k).y), 1, (255,0,0), 1)
    for k in range(49,68): #
        cv2.circle(img, (shape.part(k).x, shape.part(k).y), 1, (0,0,255), 1)
    cv2.circle(img, (shape.part(30).x, shape.part(30).y), 1, (0,255,255), 1)    
cv2.imshow('example', img) # выводим картинку
cv2.moveWindow('example', 100,400)
cv2.waitKey(10) # задержка изображений
while(1): # цикл обработки nn кадров
            ret, frame = video_capture.read()# запуск камеры
            detections = detector(frame, 1) # ф-ция выделяет лицо в прямоугольник
            for k,d in enumerate(detections): # цикл по всем найденным на изображении лицам
                shape = predictor(frame, d) #возвращает координаты точек на лице
                face_descriptor_frame = facerec.compute_face_descriptor(frame, shape) # получаем 128 дискрипторов лица
                cv2.rectangle(frame, (detections[0].left(), detections[0].top()), (detections[0].right(), detections[0].bottom()), (0, 0, 255), 1) # рисуем прямоугольник вокруг лица
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
            cv2.imshow('camera', frame) # выводим картинку с камеры
            cv2.moveWindow('camera', 600,400)
            cv2.waitKey(10) # задержка изображений
            a = distance.euclidean(face_descriptor_img,face_descriptor_frame)
            if a<0.6:
                print('Привет Никита!')
            else:
                 print('Ты не Никита!')