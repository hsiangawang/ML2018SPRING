import numpy as np
import pandas as pd
import sys


def sigmoid(z):
	res = 1/(1+np.exp(-z))
	return np.clip(res,1e-8,1-(1e-8))

def normalize(X):
	mean = sum(X)/X.shape[0]
	sigma = np.std(X, axis=0)
	#print(mean.shape)
	#for i in range(len(sigma)):
	#	if sigma[i] == 0:
	#		sigma[i] = (sigma[i-1]+sigma[i+1])/2

	for i in range(len(X)):
		X[i]=(X[i]-mean)/sigma
	return X[0:len(X)]

def sel_feature(X):
	index = X[:,3:5]
	occupation = X[:,50:64]
	Exec_managerial = X[:,50:51]
	Prof_specialty = X[:,62:63]
	age = X[:,0:2]
	education_type = X[:,11:26]
	hours = X[:,80:81]
	Race = X[:,71:75]
	captital_g_l = X[:,78:80]
	fnlwgt = X[:,10:11]
	feature = np.concatenate((age,index,hours,captital_g_l,fnlwgt),axis=1)
	#feature = normalize(feature)
	#print(feature.shape)
	#square = [0,1,3,4,5]
	#cubic = [0,1,3,4,5]
	#print(X[:,index])
	X = np.concatenate((X,feature,feature**2,feature**3), axis=1)
	#print(X.shape)
	return X

def logistic_regression(X,Y):
	lr = 0.1
	epoch = 3000
	#landa = 0.1

	w = np.random.randn(X.shape[1],1)
	ada_grad = np.zeros((X.shape[1],1))
	for i in range(1,epoch+1):
		y_pred = sigmoid(np.dot(X,w))
		diff = y_pred - Y
		loss = -np.mean(Y*np.log(y_pred) + (1-Y)*np.log(1-y_pred))
		grad = np.dot(X.T,diff)
		#grad += landa*w
		ada_grad += grad**2
		w -= lr*grad/np.sqrt(ada_grad)
		acc = acc_count(y_pred,Y)
		#if i % 200 == 0:
		#	print('epoch : %d | cost : %f | acc : %f' %(i,loss,acc))

	#print(w.shape)
			
		
	return w , acc


def acc_count(y_pred,y):
	
	y_pred[y_pred>=0.5] = 1
	y_pred[y_pred<0.5] = 0
	return np.mean(1-np.abs(y-y_pred))

def test_predict(X, w):

	y = sigmoid(np.dot(X,w))
	y[y>=0.5] = 1
	y[y< 0.5] = 0    
	#print(y.shape[0])
	f = open(sys.argv[6], 'w')
	f.write('id,label\n')
	for i in range(0, y.shape[0]):
		f.write(str(i+1)+','+str(int(y[i][0]))+'\n')
	f.close()
	return y




np.random.seed(0)
X = pd.read_csv(sys.argv[3]).as_matrix().astype('float')
Y = pd.read_csv(sys.argv[4],header=None).as_matrix().astype('float')
test_X = pd.read_csv(sys.argv[5]).as_matrix().astype('float')

temp = np.concatenate((X,test_X),axis = 0)
#print(temp.shape)
temp = sel_feature(temp)
#print(temp.shape)
temp = normalize(temp)
#print(temp.shape)
temp = np.concatenate((np.ones((temp.shape[0],1)),temp),axis = 1)
#print(temp.shape)
X = temp[:X.shape[0],:]
X_test = temp[X.shape[0]:,:]

w , acc= logistic_regression(X,Y)

#print(w)
#print(w.shape)
#print(X_test.shape)

ans = test_predict(X_test, w)


