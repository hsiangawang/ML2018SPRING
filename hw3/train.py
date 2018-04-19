import sys
import pandas as pd
import numpy as np
import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.regularizers import l1,l2
#import os
from keras.preprocessing.image import ImageDataGenerator
from keras import initializers

#os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"


data_path = sys.argv[1]

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

train_data = pd.read_csv(data_path)
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
#print(train_Y)
#print(train_Y.shape)

gen = ImageDataGenerator(featurewise_center=False,
	                     samplewise_center=False,
	                     rotation_range=20,
	                     width_shift_range=0.2,
	                     shear_range=0.3,
	                     height_shift_range=0.2,
	                     zoom_range=0.08,
	                     data_format="channels_last"
	                      )
gen.fit(train_X)
train_generator = gen.flow(train_X,train_Y,batch_size=128)

model = Sequential()

model.add(Convolution2D(400,3,3,input_shape=(48,48,1),kernel_initializer=initializers.glorot_normal(seed=None)))
model.add(LeakyReLU(alpha = 0.03))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))

model.add(Convolution2D(400,3,3,kernel_initializer=initializers.glorot_normal(seed=None)))
model.add(LeakyReLU(alpha = 0.03))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.25))

model.add(Convolution2D(400,3,3,kernel_initializer=initializers.glorot_normal(seed=None)))
model.add(LeakyReLU(alpha = 0.03))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))

model.add(Convolution2D(400,3,3,kernel_initializer=initializers.glorot_normal(seed=None)))
model.add(LeakyReLU(alpha = 0.03))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.4))


model.add(Flatten())

model.add(Dense(output_dim=500,kernel_initializer=initializers.glorot_normal(seed=None)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(output_dim=500,kernel_initializer=initializers.glorot_normal(seed=None)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(output_dim=7))
model.add(Activation('softmax'))

#adam = Adam(lr=1e-4)

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
print('Training OWO')
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, verbose=0, mode='auto')
learning_rate_funtion = ReduceLROnPlateau(monitor='val_loss',patience=3,verbose=0,factor=0.5,min_lr=0.000001)

#model.fit(train_X,train_Y,batch_size=200,epochs=100,callbacks=[early_stopping],validation_split=0.1,shuffle=True)
model.fit_generator(train_generator,steps_per_epoch=(train_X.shape[0]//128),epochs=300,callbacks=[early_stopping,learning_rate_funtion],validation_data=(valid_X,valid_Y))


print('\nTesting Valid')
loss , accuracy = model.evaluate(valid_X,valid_Y)

print('\nvalid loss: ', loss)
print('\nvalid acc: ', accuracy)

model.save("model_0.69545")
print("LUL")


#acc:0.69545













#acc:0.69545









