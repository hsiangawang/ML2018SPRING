import sys
import pandas as pd
import numpy as np
import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam


testdata_path = sys.argv[1]
predict_path = sys.argv[2]


def normalize(X):
	mean = sum(X)/X.shape[0]
	sigma = np.std(X, axis=0)
	#print(mean.shape)
	#for i in range(len(sigma)):
	#	if sigma[i] == 0:
			#print("!!")
	#		sigma[i] = (sigma[i-1]+sigma[i+1])/2
	        #print("!!!!")

	for i in range(len(X)):
		X[i]=(X[i]-mean)/sigma
	return X


temp = []
test_X_temp = []

test_data = pd.read_csv(testdata_path)
#test_X = np.array([row.split(" ") for row in test_data["feature"].tolist()],dtype = np.float32)
for i in range(test_data["feature"].shape[0]):
	temp = test_data["feature"][i].split(' ')
	test_X_temp.append(temp)
test_X = np.array(test_X_temp,dtype=np.float32)

test_X = normalize(test_X)
test_X = test_X.reshape(-1,48,48,1)


print("load model ...")
model = load_model("model_0.69545")
print("prediting...")
p = model.predict(test_X)

print("predict over")
print(p)
print(p.shape)

pred_y = np.argmax(p,axis=1)
print(pred_y)
print(pred_y.shape)

pred_y = np.array(pred_y,dtype=np.float32)
print(pred_y.shape)

f = open(sys.argv[2], 'w')
f.write('id,label\n')
for i in range(0, pred_y.shape[0]):
	f.write(str(i)+','+str(int(pred_y[i]))+'\n')
f.close()




