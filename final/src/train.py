import numpy as np
from keras.models import Sequential
from keras.models import Model
from keras.models import load_model
from keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau,CSVLogger
import sys
print('read in training data')
x_train_0 = np.load('./data/x_train_log10.npy')
x_train_1 = np.load('./data/x_train_shift_1.npy')
x_train_2 = np.load('./data/x_train_shift_2.npy')
x_train_3 = np.load('./data/x_train_shift_3.npy')
x_train_4 = np.load('./data/x_train_noise_1.npy')
x_train_5 = np.load('./data/x_train_noise_2.npy')
x_train_6 = np.load('./data/x_train_noise_3.npy')
y_train = np.load('./data/y_train.npy')

cut = 800

def new_order(x, y, length):
	order =  np.random.permutation(length)
	print(length)
	new_x = []
	new_y = []

	for i in range(length):
		new_x.append(x[order[i]])
		new_y.append(y[order[i]])
	
	new_x = np.array(new_x)
	new_y = np.array(new_y)
	return new_x, new_y		

with open('./data/train.csv','r') as f:
	n_row = 0
	verified = []
	for row in f:
		if n_row != 0:
			r = row.strip().split(',')
			verified.append(int(r[2]))
		n_row += 1


x_verify = []
x_no_verify = []
y_verify = []
y_no_verify = []
count = 0
for i in range(len(verified)):
	if verified[i] == 1:
		if count < cut:
			count += 1
			x_verify.append(x_train_0[i])
			y_verify.append(y_train[i])
		else:
			x_verify.append(x_train_0[i])
			x_verify.append(x_train_1[i])
			x_verify.append(x_train_2[i])
			x_verify.append(x_train_3[i])
			x_verify.append(x_train_4[i])
			x_verify.append(x_train_5[i])
			x_verify.append(x_train_6[i])
			y_verify.append(y_train[i])
			y_verify.append(y_train[i])
			y_verify.append(y_train[i])
			y_verify.append(y_train[i])
			y_verify.append(y_train[i])
			y_verify.append(y_train[i])
			y_verify.append(y_train[i])
	else:
		x_no_verify.append(x_train_0[i])
		x_no_verify.append(x_train_1[i])
		x_no_verify.append(x_train_2[i])
		x_no_verify.append(x_train_3[i])
		x_no_verify.append(x_train_4[i])
		x_no_verify.append(x_train_5[i])
		x_no_verify.append(x_train_6[i])
		y_no_verify.append(y_train[i])
		y_no_verify.append(y_train[i])
		y_no_verify.append(y_train[i])
		y_no_verify.append(y_train[i])
		y_no_verify.append(y_train[i])
		y_no_verify.append(y_train[i])
		y_no_verify.append(y_train[i])


x_verify = np.array(x_verify)
y_verify = np.array(y_verify)
x_no_verify = np.array(x_no_verify)
y_no_verify = np.array(y_no_verify)

#x_train = np.concatenate((x_verify[cut:],x_no_verify),axis=0)
#y_train = np.concatenate((y_verify[cut:],y_no_verify),axis=0)
x_valid = x_verify[:cut]
y_valid = y_verify[:cut]

x_verify = x_verify[cut:]
y_verify = y_verify[cut:]



x_verify, y_verify = new_order(x_verify,y_verify,x_verify.shape[0])
x_no_verify, y_no_verify = new_order(x_no_verify,y_no_verify,x_no_verify.shape[0])





a, b, c, d = x_verify.shape
print(x_verify.shape)
print('training')
if sys.argv[1] == 'train' or sys.argv[1] == 'all':
	print("-------------------")
	print("--Let's train-------")
	print("-------------------")
	model = Sequential()

	model.add(Conv2D(32, (5,10), input_shape=(b, c, 1), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2,3)))
	model.add(Dropout(0.5))

	model.add(Conv2D(64, (5,10), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2,2)))
	model.add(Dropout(0.5))

	model.add(Conv2D(128,(3,6), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2,2)))
	model.add(Dropout(0.5))

	model.add(Flatten())

	model.add(Dense(625, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))

	model.add(Dense(625, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))

	model.add(Dense(41,activation='softmax'))
	#model = load_model('model/model_0.76.h5')

	earlystopping = EarlyStopping(monitor='val_acc', patience = 8, verbose=0, mode='auto')
	lr_reducer = ReduceLROnPlateau(factor=0.2, cooldown=0, patience=4, min_lr=0.5e-6)
	checkpoint = ModelCheckpoint(filepath='./model/model_all_verify.h5', 
								 verbose=1,
								 save_best_only=True,
								 monitor='val_acc',
								 mode='max' )
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'] )

	model.fit(x_verify, y_verify,validation_data=(x_valid,y_valid), epochs=1000, batch_size=60,callbacks=[checkpoint,earlystopping])
	#score = model.evaluate(x_train,y_train)
	#print('Train Acc: ',score[1])
	print('model saved')


if sys.argv[1] == 'semi' or sys.argv[1] == 'all':
	print("-------------------")
	print("--Let's semi-------")
	print("-------------------")
	model = load_model('./model/model_all_verify.h5')
	x = x_verify
	y = y_verify
	x_test = x_no_verify[:]
	y_test = y_no_verify[:]

	a,b,c,d = x_test.shape

	for loop in range(10):
		print("\nIt is %d th loop"%(loop))

		x_test = np.reshape(x_test,(-1,b,c,1))
		y_test = np.reshape(y_test,(-1,41))

		prediction = model.predict(x_test)
		prediction_label = np.argmax(prediction, axis=1)
		prediction_candidate = list(np_utils.to_categorical(prediction_label,num_classes=41))
		prediction_candidate = np.array(prediction_candidate)
		prediction_candidate = np.reshape(prediction_candidate,(-1,1,41))

		y_label = np.argmax(y_test,axis=1)
		
		x_test = np.reshape(x_test,(-1,1,b,c,1))
		y_test = np.reshape(y_test,(-1,1,41))

		x_ok = []
		y_ok = []

		x_tmp = []
		y_tmp = []

		count1 = 0
		count2 = 0
		print("There are %d items" %prediction.shape[0])
		for i in range(prediction.shape[0]):
			print("\rsemi item : " + repr(i), end="", flush=True)
			
			if prediction_label[i] == y_label[i] and prediction[i][prediction_label[i]] > 0.9:
				#x_tmp.append(x_test[i])
				#y_tmp.append(y_test[i])
				count1 += 1
				x_ok.append(x_test[i])
				y_ok.append(y_test[i])
			elif prediction[i][prediction_label[i]] > 0.95:
				count2 += 1
				x_ok.append(x_test[i])
				y_ok.append(prediction_candidate[i])
				#print(prediction_candidate[i])
			else:
				x_tmp.append(x_test[i])
				y_tmp.append(y_test[i])
		print('\n')
		x_ok = np.array(x_ok)
		y_ok = np.array(y_ok)
		x_ok = np.reshape(x_ok,(-1,b,c,1))
		y_ok = np.reshape(y_ok,(-1,41))

		x = np.concatenate((x,x_ok),axis=0)	
		y = np.concatenate((y,y_ok),axis=0)

		x_test = np.array(x_tmp)
		y_test = np.array(y_tmp)

		earlystopping = EarlyStopping(monitor='val_acc', patience = 6, verbose=0, mode='auto')
		lr_reducer = ReduceLROnPlateau(factor=0.2, cooldown=0, patience=4, min_lr=0.5e-6)
		checkpoint = ModelCheckpoint(filepath='./model/model_all_'+str(loop)+'.h5', 
								 verbose=1,
								 save_best_only=True,
								 monitor='val_acc',
								 mode='max' )

		print("\nWe add %d data into training %d %d"%(count1+count2,count1,count2))

		model.fit(x, y,validation_data=(x_valid,y_valid), epochs=1000, batch_size=10,callbacks=[checkpoint,earlystopping])
		model = load_model('./model/model_all_'+str(loop)+'.h5')
