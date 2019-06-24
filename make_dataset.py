import cv2
import os
import time
cam = cv2.VideoCapture(0)
cam.set(3, 800) # устанавливаем высоту окна
cam.set(4, 600) # устанавливаем ширину окна

# Установим классификатор лица, предоставляемый библиотекой OpenCV
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Для каждого лица, устанавливаем свой id
face_id = input('\n enter EMOTION id end press <return> ==>  ')

print("\testn [INFO] Initializing face capture. Look the camera and wait ...")
# Инициализация индивидуальной выборки лица
count = 0

while(True):

    ret, img = cam.read()
    # Переводим изображение в оттенки серого
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    # Выделяем лицо в прямоугольную область
    for (x,y,w,h) in faces:
        count+=1
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        # Сохраняем захваченное изображение в папку dataset
        cv2.imwrite("datasetV/EM_{}_{}.jpg".format(face_id, int(time.time()*1e9)), gray[y:y+h,x:x+w])
        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff # Для завершения работы программы нажмите «ESC»
    if k == 27:
        break
    elif count >= 300: # Выполнение программы заканчивается после того, как сделает определённое количество кадров (захватит изображений лица)
        break
    print(count)

# Заканчиваем видеопоток и закрываем окна
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
6