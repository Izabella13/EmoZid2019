#!/usr/bin/env python
#coding=utf8
from numpy import array, arange, abs as np_abs
from numpy.fft import rfft, rfftfreq, irfft
from numpy.random import uniform
from math import sin, pi
import matplotlib.pyplot as plt
import matplotlib as mpl
import pylab


import json


num = input('Input your emotion: ')

def parse(num,detector):
    with open(num+'+.JSON') as jf:  #"JsonData/"
        data = json.load(jf)
        secs = data[6]['DATABLOCK']
        i=0
        j=0
        dat = []
        while len(secs)>i:
            if i > 3000 and i<5001:
                dat.append(secs[i]['data'][detector])
            i += 1
        return dat
        #print(dat)

# Вывод на экран текущей версии библиотеки matplotlib
print ('Current version on matplotlib library is', mpl.__version__)

# а можно импортировать numpy и писать: numpy.fft.rfft
FD = 250 # частота дискретизации, отсчётов в секунду  22050
# а это значит, что в дискретном сигнале представлены частоты от нуля до 11025 Гц (это и есть теорема Котельникова)
N = 2000 # длина входного массива, 0.091 секунд при такой частоте дискретизации
# сгенерируем сигнал с частотой 440 Гц длиной N
pure_sig = array([6.*sin(2.*pi*440.0*t/FD) for t in range(N)])
print(pure_sig)
# сгенерируем шум, тоже длиной N (это важно!)
noise = uniform(-50.,50., N)
detectors = [3,4,5]
# суммируем их и добавим постоянную составляющую 2 мВ (допустим, не очень хороший микрофон попался. Или звуковая карта или АЦП)
for detector in detectors:
    sig = array(parse(num, detector), float)  # в numpy так перегружена функция сложения   pure_sig + noise + 2.0
    print(sig)
    bug = arange(N)/float(FD)
    # вычисляем преобразование Фурье. Сигнал действительный, поэтому надо использовать rfft, это быстрее, чем fft
    spectrum = rfft(sig) #X
    #Модифицировать спектрум в Все выщше 50 элемента = 0

    # нарисуем всё это, используя matplotlib
    # Сначала сигнал зашумлённый и тон отдельно
    pylab.figure (1)
    plt.plot(arange(N)/float(FD), sig) # по оси времени секунды!
    plt.plot(arange(N)/float(FD), pure_sig, 'r') # чистый сигнал будет нарисован красным
    plt.xlabel(u'Время, c') # это всё запускалось в Python 2.7, поэтому юникодовские строки
    plt.ylabel(u'Напряжение, мВ')
    plt.title(u'Зашумлённый сигнал и тон 440 Гц')
    plt.grid(True)
    #plt.show()
    # когда закроется этот график, откроется следующий
    # Потом спектр
    pylab.figure(2)
    zzz = rfftfreq(N, 1./FD)
    print (len(zzz))
    debug = np_abs(spectrum)/N
    for j,d in enumerate(debug):
        if j <9 or j >13:
            spectrum[j] = 0;
    sigv = irfft(spectrum)



    plt.plot(zzz, np_abs(spectrum)/N)
    # rfftfreq сделает всю работу по преобразованию номеров элементов массива в герцы
    # нас интересует только спектр амплитуд, поэтому используем abs из numpy (действует на массивы поэлементно)
    # делим на число элементов, чтобы амплитуды были в милливольтах, а не в суммах Фурье. Проверить просто — постоянные составляющие должны совпадать в сгенерированном сигнале и в спектре
    plt.xlabel(u'Частота, Гц')
    plt.ylabel(u'Напряжение, мВ')
    plt.title(u'Спектр')
    plt.grid(True)
    #Все выщше 50 элемента = 0
    pylab.figure (3)
    plt.plot(arange(N)/float(FD), sigv) # по оси времени секунды!
    #plt.plot(arange(N)/float(FD), pure_sig, 'r') # чистый сигнал будет нарисован красным
    plt.xlabel(u'Время, c') # это всё запускалось в Python 2.7, поэтому юникодовские строки
    plt.ylabel(u'Напряжение, мВ')
    plt.title(u'Зашумлённый сигнал и тон 250 Гц')
    plt.grid(True)

    plt.show()
