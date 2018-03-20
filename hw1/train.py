# coding: utf-8
import numpy as np
import pandas as pd
import sys
import random
import math as math

input_path = sys.argv[1]
model_path = sys.argv[2]

#load the train data
input_data = pd.read_csv(input_path , encoding="big5").as_matrix()
data = input_data[ : , 3: ]
data[data == 'NR'] = 0
data.astype(float)

#cocate 

temp = np.split(data , range(18,data.shape[0],18) , axis=0)
All_attri = np.concatenate(temp , axis=1)
#print(All_attri.shape) #18*5760

Attribute_PM = All_attri[9]
Attribute_PM10 = All_attri[8]
Attribute_O3 = All_attri[7]
Attribute_SO2 = All_attri[12]
#print(Attribute_PM.shape)

Attribute_PM = np.array(Attribute_PM,dtype=float)
Attribute_PM10 = np.array(Attribute_PM10,dtype=float)
Attribute_O3 = np.array(Attribute_O3)
Attribute_SO2 = np.array(Attribute_SO2)
#print(Attribute_PM[3359])  #2880~3359 is july
#print(Attribute_PM10)

for i in range(Attribute_PM.shape[0]):
    if Attribute_PM[i] <= 0 :
        Attribute_PM[i] = (Attribute_PM[i-1] + Attribute_PM[i+1])/2
for i in range(Attribute_PM10.shape[0]):
    if Attribute_PM10[i] <= 0 :
        Attribute_PM10[i] = (Attribute_PM10[i-1] + Attribute_PM10[i+1])/2

#Attribute_PM = np.append(Attribute_PM[ :3360] , Attribute_PM[3840: ])
#Attribute_PM10 = np.append(Attribute_PM10[ :3360] , Attribute_PM10[3840: ])

train_x = []
train_y = []
Attribute_PM = np.append(Attribute_PM[ :4800] , Attribute_PM[5280:])
Attribute_PM10 = np.append(Attribute_PM10[ :4800] , Attribute_PM10[5280:])
print(Attribute_PM.shape)
month = All_attri[ : , 4800 : 5280 ]

for i in range(Attribute_PM.shape[0]-480-9):
    #new_feature = []
    mean_PM2 = 0
    mean_PM10 = 0
    std_PM2 = 0
    std_PM10 = 0
    flag = False
    fea_2 = np.array(Attribute_PM[i+4:i+9])
    fea_10 = np.array(Attribute_PM10[i+4:i+9])
    fea_O3 = np.array(Attribute_O3[i+4:i+9])
    fea_SO2 = np.array(Attribute_SO2[i+4:i+9])

    new_feature = np.append(fea_2,fea_10)
    
    train_x.append(np.append(new_feature,1.0))
    if Attribute_PM[i+9] <= 0 :
        train_y.append((Attribute_PM[i+8]+Attribute_PM[i+10])/2)
        #print("!!!!")
    else:
        train_y.append(Attribute_PM[i+9])

train_x = np.array(train_x,dtype=float)
#train_x = (train_x - np.mean(train_x))/np.std(train_x)
#print(train_x.shape)
train_y = np.array(train_y,dtype=float)

weight = [[1.0]*11]
weight = np.array(weight)
weight = weight.flatten()
#print(weight.dtype)

lr = 0.3
adasum = 0.0

for i in range(0,1000):
    loss_sum = 0
    guess_y = np.dot(train_x,weight)
    loss = train_y - guess_y
    for i in range(len(loss)):
        loss_sum += loss[i]
    print(loss_sum)

    gradient = -2* np.dot(train_x.T,loss)
    adasum += gradient**2
    weight -= lr*gradient/np.sqrt(adasum)

valid_x = []
valid_y = []

for i in range(471):
    PM2 = np.array(month[9, i+4:i+9])
    PM10 = np.array(month[8, i+4:i+9])
    O3 = np.array(month[7, i+4:i+9])
    SO2 = np.array(month[12, i+4:i+9])

    new_feature = np.append(PM2,PM10)
    valid_x.append(np.append(new_feature,1.0))
    valid_y.append(month[9,i+9])

valid_x = np.array(valid_x , dtype=float)
valid_y = np.array(valid_y , dtype=float)
guess_valid_y = np.dot(valid_x,weight)

valid_errsum = 0
loss_valid = valid_y - guess_valid_y
#print(loss_valid)

for i in range(valid_x.shape[0]):
    valid_errsum += loss_valid[i]**2
print(np.sqrt(valid_errsum)) 

f = open(sys.argv[2], 'w')
#f.write('id,value\n')
for i in range(0, weight.shape[0]):
    f.write(str(i)+','+str(weight[i])+'\n')
f.close()












