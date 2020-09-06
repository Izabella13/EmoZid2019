import json


with open("1.1-.JSON") as jf:
    data = json.load(jf)
    file = open("data_from_json+.txt", 'a')
    secs = data[6]['DATABLOCK']
    i=0
    j=0
    while len(secs)>i:
        dat = secs[i]['data']
        #print(dat)
        for d in dat:
            #print(d)
            if (j==0):
                file.write(str(d))
            file.write(','+ str(d))
            j += 1
        file.write('\n')
        j = 0
        i += 1
file.close()