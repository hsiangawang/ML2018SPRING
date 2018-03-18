# coding: utf-8
import numpy as np
import pandas as pd
import sys
import random
import math as math

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
#print(month_12.shape)
#print(Attribute_PM.shape)
#split_feature = np.split(Attribute_PM, range(480, Attribute_PM.shape[0], 480), axis=0)
#split_feature_PM10 = np.split(Attribute_PM10, range(480,Attribute_PM10.shape[0], 480), axis=0)
#print(split_feature)
#print(split_feature_PM10)
#print(split_feature)

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

    #fea_2[fea_2.shape[0]-1] = float(fea_2[fea_2.shape[0]-1])**2
    #fea_2[fea_2.shape[0]-2] = float(fea_2[fea_2.shape[0]-2])**2
    #for j in range(fea_2.shape[0]):
    #    if fea_2[j] == 0:
    #        flag = True
    #if flag == True:
    #    break


    #print(np.mean(fea_2))
    #print(np.mean(fea_10))
    #for k in range(fea_10.shape[0]):
    #    fea_10[k] = float(fea_10[k])**3
    #for j in range(9):
    #    new_feature.append(float(fea_2[j]) * float(fea_O3[j]) * float(fea_SO2[j]))
    #new_feature = np.append(fea_2,fea_2)
    new_feature = np.append(fea_2,fea_10)
    #mean_feature = np.mean(new_feature)
    #std_feature = np.std(new_feature)
    #new_feature = (new_feature - mean_feature)/std_feature
    #new_feature = np.append(new_feature,fea_O3)
    #new_feature = np.append(new_feature,fea_SO2)
    train_x.append(np.append(new_feature,1.0))
    if Attribute_PM[i+9] <= 0 :
        train_y.append((Attribute_PM[i+8]+Attribute_PM[i+10])/2)
        #print("!!!!")
    else:
        train_y.append(Attribute_PM[i+9])

#for i in range(11):
#    for j in range(471):
#        fea_2 = np.array(Attribute_PM[480*i+j+4:480*i+j+9])
#        fea_10 = np.array(Attribute_PM10[480*i+j+4:480*i+j+9])
#        new_feature = np.append(fea_2,fea_10)
        #train_x.append(np.append(new_feature,1.0))
#        train_x.append(np.append(new_feature,1.0))
#        train_y.append(Attribute_PM[480*i+j+9])

train_x = np.array(train_x,dtype=float)
#train_x = (train_x - np.mean(train_x))/np.std(train_x)
#print(train_x.shape)
train_y = np.array(train_y,dtype=float)

#mean_trainy = np.mean(train_y)
#std_trainy = np.std(train_y)
#train_y = (train_y - mean_trainy)/std_trainy

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
#print(train_y)

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

#print(weight.shape)
output_data = pd.read_csv(output_path , encoding= 'big5' , header = None ).as_matrix()
data_test = output_data[ : , 2: ]
data_test = np.array(data_test)

#print(data_test.shape)
test_x = []
for i in range(9,data_test.shape[0],18):
    PM2  = np.array(data_test[i , 4: ])
    PM10 = np.array(data_test[i-1 , 4: ])
    O3 = np.array(data_test[i-2 , 4: ])
    SO2 = np.array(data_test[i+3 , 4: ])

    #PM2[PM2.shape[0]-1] = float(PM2[PM2.shape[0]-1])**2
    #PM10[PM10.shape[0]-1] = float(PM10[PM10.shape[0]-1])**2
    PM2 = np.array(PM2,dtype=float)
    PM10 = np.array(PM10,dtype=float)

    for i in range(PM2.shape[0]):
        if PM2[i] <= 0 :
            if i != 4:
                PM2[i] = (PM2[i-1] + PM2[i+1])/2
            else:
                PM2[i] = PM2[i-1]
    for i in range(PM10.shape[0]):
        if PM10[i] <= 0 :
            if i != 4:
                PM10[i] = (PM10[i-1] + PM10[i+1])/2
            else:
                PM10[i] = PM10[i-1]

    #for j in range(PM2.shape[0]):
    #    PM2[j] = math.pow
    #for k in range(PM10.shape[0]):
    #    PM10[k] = float(PM10[k])**2
    #new_feature = np.append(PM2,PM2)
    new_feature = np.append(PM2,PM10)
    #new_feature = np.append(new_feature,O3)
    #new_feature = np.append(new_feature,SO2)
    test_x.append(np.append(new_feature,1.0))

test_x = np.array(test_x , dtype=float)
#print(test_x)
#print(Attribute_PM_test)
#print(Attribute_PM_test.shape) 2340

test_y = np.dot(test_x,weight)
#print(test_y[0])
valid_x = []
valid_y = []

for i in range(471):
    PM2 = np.array(month[9, i+4:i+9])
    PM10 = np.array(month[8, i+4:i+9])
    O3 = np.array(month[7, i+4:i+9])
    SO2 = np.array(month[12, i+4:i+9])

    #for j in range(PM2.shape[0]):
    #    PM2[j] = float(PM2[j])**2
    #for k in range(PM10.shape[0]):
    #    PM10[k] = float(PM10[k])**2
    #PM2[PM2.shape[0]-1] = float(PM2[PM2.shape[0]-1])**2
    #PM2[PM2.shape[0]-2] = float(PM2[PM2.shape[0]-2])**2
    
    #new_feature = np.append(PM2,PM2)
    new_feature = np.append(PM2,PM10)
    #new_feature = np.append(new_feature,O3)
    #new_feature = np.append(new_feature,SO2)
    valid_x.append(np.append(new_feature,1.0))
    valid_y.append(month[9,i+9])

valid_x = np.array(valid_x , dtype=float)
valid_y = np.array(valid_y , dtype=float)
guess_valid_y = np.dot(valid_x,weight)
#print("valid_y: "+'\n', valid_y)
#print("guess_valid_y: "+'\n',guess_valid_y)
#print(guess_valid_y)

valid_errsum = 0
loss_valid = valid_y - guess_valid_y
#print(loss_valid)

for i in range(valid_x.shape[0]):
    valid_errsum += loss_valid[i]**2
print(np.sqrt(valid_errsum)) 

f = open(sys.argv[3], 'w')
f.write('id,value\n')
for i in range(0, test_y.shape[0]):
    f.write('id_' + str(i) + ',' + str(test_y[i]) + '\n')
f.close()














