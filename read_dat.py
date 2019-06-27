from sklearn import svm
#import numpy as np 
import csv
import pickle
from sklearn.utils import shuffle
dat=[]
y=[]
f_ns=['Em_1_128.txt','Em_2_128.txt','Em_3_128.txt','Em_4_128.txt','Em_5_128.txt','Em_6_128.txt','Em_7_128.txt']
for i, f_n in enumerate(f_ns):
    f=open(f_n,'r')
    reader = csv.reader(f)
    for row in reader:
        dat.append(row)
        y.append(i+1)
    f.close
#np.array(dat,dtype=float)
#print(dat[0])
dat, y = shuffle(dat, y, random_state=0)
l_dat,l_y=dat[:int(len(dat)*0.8)], y[:int(len(y)*0.8)]
t_dat,t_y=dat[int(len(dat)*0.8):], y[int(len(y)*0.8):]

clf = svm.SVC(C=2, gamma=2, kernel='poly', degree=3) 
#decision_function_shape='ovo'

clf.fit(l_dat,l_y)
pred_lin =clf.score(t_dat,t_y)
print (pred_lin)
f=open('svm_dat.dat','wb')
pickle.dump(clf,f)
f.close


