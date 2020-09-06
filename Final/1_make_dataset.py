import cv2
import time
dat_path='Data/'
cam = cv2.VideoCapture(0) # захватываем изображение
cam.set(3, 800) # устанавливаем высоту кадра
cam.set(4, 600) # устанавливаем ширину кадра
face_detector = cv2.CascadeClassifier(dat_path+'haarcascade_frontalface_default.xml') # Установим классификатор лица, предоставляемый библиотекой OpenCV
face_id = input('\n Введите номер эмоции и нажмите enter ==>  ') # Для каждой эмоции, устанавливаем свой id
print("\t [INFO] Инициализация лица. Смотрите в камеру и ждите...")
count = 0
while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Переводим изображение в оттенки серого
    '''faces = face_detector.detectMultiScale(gray, 1.3, 5) # Выделяем лицо в прямоугольную область
    for (x,y,w,h) in faces:
        count+=1
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)'''
    cv2.imwrite("Img/page"+str(count), gray) # Сохраняем захваченное изображение в папку dataset "dataset/Em_{}/EM_{}_{}.jpg".format(face_id, face_id, int(time.time()*1e9)), gray[y:y+h,x:x+w]
    cv2.imshow('image', img)
    if cv2.waitKey(300) & 0xff == 27:
        break # Для завершения работы программы нажмите «ESC»
    elif count >= 30:
        break # Выполнение программы заканчивается после того, как сделает определённое количество кадров (захватит изображений лица)
    print(count)

print("\n [INFO] Закрытие программы")
cam.release()
cv2.destroyAllWindows() # Заканчиваем видеопоток и закрываем окна
