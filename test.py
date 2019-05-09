import pandas as pd
import numpy

def pointsize(x,size):
    result = []
    for k in x:
        a = k*(10**size)
        result.append(int(a)/(10**size))
    return result

a=pd.read_csv('data/val_csv/reslut001.csv')
k = a.to_dict()
del k['frame']
b = []
for item in k:
    b.append(k[item][0])
a = numpy.array(b)
print(10**2)
print(b)
print(a.shape)
print(pointsize(b,2))