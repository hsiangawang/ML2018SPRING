import numpy as np
import pandas as pd
import sys
import random
import math as math

#input_path = sys.argv[1]
output_path = sys.argv[1]
model_path = sys.argv[3]

model = pd.read_csv(model_path , encoding='big5' , header = None).as_matrix()
weight = model[ : ,1]
weight = np.array(weight)

output_data = pd.read_csv(output_path , encoding= 'big5' , header = None ).as_matrix()
data_test = output_data[ : , 2: ]
data_test = np.array(data_test)

#print(data_test.shape)
test_x = []
for i in range(9,data_test.shape[0],18):
    PM2  = np.array(data_test[i , 4: ])
    PM10 = np.array(data_test[i-1 , 4: ])
    O3 = np.array(data_test[i-2 , : ])
    SO2 = np.array(data_test[i+3 , 4: ])
    NO = np.array(data_test[i-5 , : ])
    NO2 = np.array(data_test[i-4 , : ])
    NOX = np.array(data_test[i-3 , : ])

    #PM2[PM2.shape[0]-1] = float(PM2[PM2.shape[0]-1])**2
    #PM10[PM10.shape[0]-1] = float(PM10[PM10.shape[0]-1])**2
    PM2 = np.array(PM2,dtype=float)
    PM10 = np.array(PM10,dtype=float)
    O3 = np.array(O3,dtype=float)
    NO = np.array(NO,dtype=float)
    NO2 = np.array(NO2,dtype=float)
    NOX = np.array(NOX,dtype=float)

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

    new_feature = np.append(PM2,PM10)
    
    
    test_x.append(np.append(new_feature,1.0))

test_x = np.array(test_x , dtype=float)
#print(test_x)
#print(Attribute_PM_test)
#print(Attribute_PM_test.shape) 2340

test_y = np.dot(test_x,weight)

f = open(sys.argv[2], 'w')
f.write('id,value\n')
for i in range(0, test_y.shape[0]):
    f.write('id_' + str(i) + ',' + str(test_y[i]) + '\n')
f.close()

