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
print(All_attri.shape) #18*5760


Attribute_PM = All_attri[9]
print(Attribute_PM)