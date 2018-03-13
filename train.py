# coding: utf-8
import numpy as np
import pandas as pd
import sys
import random

input_path = sys.argv[1]
output_path = sys.argv[2]

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
#print(Attribute_PM)
Attribute_PM = np.array(Attribute_PM)
Attribute_PM10 = np.array(Attribute_PM10)
#print(Attribute_PM)
#print(Attribute_PM10)

train_x = []
train_y = []
#print(Attribute_PM.shape)
#split_feature = np.split(Attribute_PM, range(480, Attribute_PM.shape[0], 480), axis=0)
#split_feature_PM10 = np.split(Attribute_PM10, range(480,Attribute_PM10.shape[0], 480), axis=0)
#print(split_feature)
#print(split_feature_PM10)
#print(split_feature)

for i in range(Attribute_PM.shape[0]-9):
    fea_2 = np.array(Attribute_PM[i+3:i+9])
    fea_10 = np.array(Attribute_PM10[i+4:i+9])
    new_feature = np.append(fea_2,fea_10)
    train_x.append(np.append(new_feature,1.0))
    train_y.append(Attribute_PM[i+9])

train_x = np.array(train_x,dtype=float)
#train_x = (train_x - np.mean(train_x))/np.std(train_x)
#print(train_x)
train_y = np.array(train_y,dtype=float)

#for i in range(train_x.shape[0]):
    #print(train_x[i])
    #print("mean: ",np.mean(train_x[i]))
    #print("normalization: ",(train_x[i]-np.mean(train_x[i]))/np.std(train_x[i]))
    #print("std: ",np.std(train_x[i]))
    #if((np.std(train_x[i]))!= 0.0):
    #    train_x[i] = (train_x[i]-np.mean(train_x[i]))/np.std(train_x[i])
    #else:
    #    train_x[i] = (train_x[i]-np.mean(train_x[i]))
    #print("after: ",train_x[i])
    #print("mean after: ",np.mean(train_x[i]))
    #print("std after: ",np.std(train_x[i]))
    #print("mean: ",np.mean(train_x[i]))
    #print("std: ",np.std(train_x[i]))

#print(train_x)
#print(train_y.shape)

weight = [[1.0]*12]
weight = np.array(weight)
weight = weight.flatten()
#print(weight.dtype)

lr = 0.3
adasum = 0.0


for i in range(0,5000):
    loss_sum = 0
    guess_y = np.dot(train_x,weight)
    loss = train_y - guess_y
    for i in range(len(loss)):
        loss_sum += loss[i]
    print(loss_sum)

    gradient = -2* np.dot(train_x.T,loss)
    adasum += gradient**2
    weight -= lr*gradient/np.sqrt(adasum)

print(weight.shape)
output_data = pd.read_csv(output_path , encoding= 'big5' , header = None ).as_matrix()
data_test = output_data[ : , 2: ]
data_test = np.array(data_test)

#print(data_test)
test_x = []
for i in range(9,data_test.shape[0],18):
    PM2  = np.array(data_test[i, 3: ])
    PM10 = np.array(data_test[i-1,4:])
    new_feature = np.append(PM2 , PM10)
    test_x.append(np.append(new_feature, 1.0))
    #test_x.append(new_feature)

test_x = np.array(test_x , dtype=float)
print(test_x.shape)
#print(Attribute_PM_test)
#print(Attribute_PM_test.shape) 2340

test_y = np.dot(test_x,weight)
#print(test_y[0])

f = open(sys.argv[3], 'w')
f.write('id,value\n')
for i in range(0, test_y.shape[0]):
    f.write('id_' + str(i) + ',' + str(test_y[i]) + '\n')
f.close()














