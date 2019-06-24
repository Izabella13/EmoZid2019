import cv2
import numpy as np
from PIL import Image
import os

# Директория, в которой храниться dataset
path = 'dataset'

# Задаём распознаватель (LBPH)
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Установим классификатор лица, предоставляемый библиотекой OpenCV
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Функция для получения изображений и меток данных
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faceSamples=[]
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L') # Перевод изображения в градации серого
        img_numpy = np.array(PIL_img,'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples,ids

print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# Сохраняем обученную модель в trainer/trainer.yml
recognizer.write('trainer/trainer.yml')

# Печатаем количество лиц, которым обучен распознователь
print("\n [INFO] {} faces trained. Exiting Program".format(len(np.unique(ids))))
