from sklearn import svm
import csv
import pickle
from sklearn.utils import shuffle
import pandas as pd
dis_path='../'
dat_path='Data/'
dat=[] #Список, содержащий дискрипторы
emotion_number=[] #Список, содержащий номера эмоций
file_names=['data_from_json+.txt','data_from_json-.txt'] #Список, содержащий документы, в которых находятся параметры каждой эмоции в формате cvs   'data_from_json_unreal_happy.txt','data_from_json_real_happy.txt'
for i, file_name in enumerate(file_names): #Запуск цикла, пробегающегося по списку file_names, доставая и его индекс i, и то, что в нем находится
    f=open(dis_path+file_name,'r') #открываем каждый из этих файлов с разрешением на чтение
    reader = csv.reader(f) #Считываем параметры в формате CVS
    for row in reader: #Пробегаемся по считанным данным
        dat.append(row) #Добавляем в список считанные данные
        emotion_number.append(i) #Добавляем в список номера всех эмоций
    f.close() #Закрываем файлы
pp=int(len(dat)*0.8) #Индекс в списке, соответствующий 80-ти процентам(0.8 - доля от общей выборки на обучаение) 
dat, emotion_number = shuffle(dat, emotion_number, random_state=0) #Смешивает полученные данные одинаковым способ, сохраняя соответствие номера эмоций и их параметров
learning_dat,learning_emotion_number=dat[:pp], emotion_number[:pp] #Обучаящая выборка
test_dat,test_emotion_number=dat[pp+1:], emotion_number[pp+1:] #Тестирующая выборка
clf = svm.SVC(C=2.0, gamma=2.0 , kernel='poly', degree=3) #Задаем параметры для метода обучения Support Vector Machine
clf = svm.SVC()
clf.fit(dat,emotion_number) #Запускаем метод обучения Support Vector Machine  learning_dat,learning_emotion_number
succesfull_prediction =clf.score(test_dat,test_emotion_number) #Доля правильно определенных эмоций на фотографиях из базы
print (succesfull_prediction) #Выводим полученную долю в консоль
f=open(dat_path+'svm_dat1.dat','wb') #Сохроняем полученные данные в файл svm_dat
pickle.dump(clf,f) #Запись на диск
f.close
