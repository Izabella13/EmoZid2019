import json


num = input('Input your emotion: ')
with open(num+"-.JSON") as jf:  #"JsonData/"
    data = json.load(jf)
    secs = data[6]['DATABLOCK']
    i=0
    j=0
    dat = []
    while len(secs)>i:
        if i > 3000 and i<5001:
            dat.append(secs[i]['data'][3])
            dat.append(secs[i]['data'][4])
            dat.append(secs[i]['data'][5])
        i += 1
    print(dat)
        
