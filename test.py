import pandas as pd

a=pd.read_csv('data/val_csv/reslut001.csv')
k = a.to_dict()
del k['frame']
b = {}
for item in k:
    b[item] = k[item][0]
print(b)