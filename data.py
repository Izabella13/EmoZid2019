from sklearn import svm
import csv
dat=[]
i=0
y=[]
f_ns=['Em_1.txt','Em_2.txt','Em_3.txt', 'Em_4.txt', 'Em_6.txt']
for f_n in f_ns:
    i+=1
    f=open(f_n,'r')
    reader = scv.reader(f)
    for row in reader:
        dat1.append(row)
        if i<5:
            y.append(i)
        else:
            y.append(6)
    f.close
clf= svm.SVC(gamma='scale', kernel='cubic')
clf.fit(dat, y)
f=open('svn_dat.dat','wb')