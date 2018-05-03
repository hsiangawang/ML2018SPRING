import sys
import pandas as pd
import numpy as np
import keras
from keras.models import load_model , Model
from keras.utils import np_utils
from keras.layers import Input , Average
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier

testdata_path = sys.argv[1]
predict_path = sys.argv[2]
train_path = sys.argv[3]

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

def ensemble(models , model_input):

	outputs = [model(model_input) for model in models]
	y = Average()(outputs)

	model = Model(model_input , y , name = 'ensemble')

	return model

train_data = pd.read_csv(train_path)
temp = []
train_X_temp = []
train_Y_temp = []

for i in range(train_data["feature"].shape[0]):
	temp = train_data["feature"][i].split(' ')
	train_X_temp.append(temp)
train_X = np.array(train_X_temp,dtype=np.float32)
print(train_X.shape)

for i in range(train_data["label"].shape[0]):
	temp = train_data["label"][i]
	train_Y_temp.append(temp)
train_Y = np.array(train_Y_temp,dtype=np.float32)

train_X = normalize(train_X)
#print(train_X.shape)
#print(train_Y.shape)
valid_size = 2000

train_X = train_X[ : -valid_size]
valid_X = train_X[-valid_size : ]
train_Y = train_Y[ : -valid_size]
valid_Y = train_Y[-valid_size : ] 

#print(train_X.shape)
#print(valid_X.shape)

train_X = train_X.reshape(-1,48,48,1)
valid_X = valid_X.reshape(-1,48,48,1)
train_Y = np_utils.to_categorical(train_Y , num_classes=7)
valid_Y = np_utils.to_categorical(valid_Y , num_classes=7)

model_input = Input(shape = (48,48,1))

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

estimators = []

model1 = load_model("model_1.hdf5")
model2 = load_model("model_2.hdf5")
model3 = load_model("model_3.hdf5")

estimators = [model1 , model2 , model3]

ensemble_model = ensemble(estimators , model_input)
ensemble_model.save("modelEns.h5")
modelEns = load_model("modelEns.h5")
modelEns.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

print('\nTesting Valid')
loss , accuracy = modelEns.evaluate(valid_X,valid_Y)

print('\nvalid loss: ', loss)
print('\nvalid acc: ', accuracy)

p = ensemble_model.predict(test_X)


print("predict over")
#print(p)
#print(p.shape)

pred_y = np.argmax(p,axis=1)
#print(pred_y)
#print(pred_y.shape)

pred_y = np.array(pred_y,dtype=np.float32)
print(pred_y.shape)

f = open(sys.argv[2], 'w')
f.write('id,label\n')
for i in range(0, pred_y.shape[0]):
	f.write(str(i)+','+str(int(pred_y[i]))+'\n')
f.close()





