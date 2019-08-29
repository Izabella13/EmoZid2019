from sklearn import svm
import csv
import pickle
from sklearn.utils import shuffle
dat=[] #Список, содержащий дискрипторы
emotion_number=[] #Список, содержащий номера эмоций
file_names=['Em_1_128.txt','Em_2_128.txt','Em_3_128.txt','Em_4_128.txt','Em_5_128.txt','Em_6_128.txt','Em_7_128.txt'] #Список, содержащий документы, в которых находятся параметры каждой эмоции в формате cvs
for i, file_name in enumerate(file_names): #Запуск цикла, пробегающегося по списку file_names, доставая и его индекс i, и то, что в нем находится
    f=open(file_name,'r') #открываем каждый из этих файлов с разрешением на чтение
    reader = csv.reader(f) #Считываем параметры в формате CVS
    for row in reader: #Пробегаемся по считанным данным
        dat.append(row) #Добавляем в список считанные данные
        emotion_numbers.append(i+1) #Добавляем в список номера всех эмоций
    f.close #Закрываем файлы
pp=int(len(dat)*0.8) #Индекс в списке, соответствующий 80-ти процентам(0.8 - доля от общей выборки на обучаение) 
dat, emotion_number = shuffle(dat, emotion_number, random_state=0) #Смешивает полученные данные одинаковым способ, сохраняя соответствие номера эмоций и их параметров
learning_dat,learning_emotion_number=dat[:pp], y[:pp] #Обучаящая выборка
test_dat,test_emotion_number=dat[pp+1:], emotion_number[pp+1:] #Тестирующая выборка
clf = svm.SVC(C=2.0, gamma=2.0 , kernel='poly', degree=3) #Задаем параметры для метода обучения Support Vector Machine
clf.fit(l_dat,l_y) #Запускаем метод обучения Support Vector Machine
succesfull_prediction =clf.score(t_dat,t_emotion_number) #Доля правильно определенных эмоций на фотографиях из базы
print (succesfull_prediction) #Выводим полученную долю в консоль
f=open('svm_dat.dat','wb') #Сохроняем полученные данные в файл svm_dat
pickle.dump(clf,f) #Запись на диск
f.close
i=1
j=1
while i<3:
    while j<3:
        clf = svm.SVC(C=j, gamma=i , kernel='poly', degree=3)
        clf.fit(l_dat,l_y)
        pred_lin =clf.score(t_dat,t_y)
        print (str(pred_lin)+'   '+'C = '+str(j)+'  gamma = '+str(i))
        f=open('svm_dat.dat','wb')
        pickle.dump(clf,f)
        f.close
        j+=0.1
    i+=0.1
    j=1.4


