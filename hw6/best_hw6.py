import sys
import numpy as np
import csv 
import os
import keras
from itertools import islice
from keras.models import Model, Sequential, load_model
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from keras.regularizers import l2


from sklearn.model_selection import train_test_split
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"

train_path = sys.argv[1]

def trian_parser(train_path):
	with open(train_path, 'r') as f:
		for line in islice(f, 1 ,None):
		    lines = csv.reader(f)
		    train_data = []
		    User_id = []
		    Movie_id = []
		    Rating = []
		    All_User = []
		    All_Movie = []
		    All_Rating = []
		    for row in lines :
			    row[0] = int(row[0])
			    row[1] = int(row[1])
			    row[2] = int(row[2])
			    row[3] = int(row[3])
			    All_User.append(row[1])
			    All_Movie.append(row[2])
			    All_Rating.append(row[3])
			    if row[1] not in User_id:
			    	User_id.append(row[1])
			    if row[2] not in Movie_id:
			    	Movie_id.append(row[2])
			    if row[3] not in Rating:
			    	Rating.append(row[3])
			    train_data.append(row)

	train_data = np.array(train_data)
	User_id = np.array(User_id)
	Movie_id = np.array(Movie_id)
	Rating = np.array(Rating)
	All_User = np.array(All_User)
	All_Movie = np.array(All_Movie)
	All_Rating = np.array(All_Rating)
	print(train_data.shape[0])
	print(User_id.shape)
	print(Movie_id.shape[0])
	print(Rating.shape)

	return train_data, User_id, Movie_id, Rating, All_User, All_Movie, All_Rating


def main():
	data, User_id, Movie_id, Rating, All_User, All_Movie, All_Rating = trian_parser(train_path)
	#User_train, User_valid, Movie_train, Movie_valid, Rating_train, Rating_valid = train_test_split(User_id, Movie_id, Rating, test_size=0.1)
	User_valid = []
	Movie_valid = []
	Rating_valid = []
	User_train = []
	Movie_train = []
	Rating_train = []
	for i in range(data.shape[0]):
		if i%10==1.0:
			User_valid.append(All_User[i])
			Movie_valid.append(All_Movie[i])
			Rating_valid.append(All_Rating[i])
		else:
			User_train.append(All_User[i])
			Movie_train.append(All_Movie[i])
			Rating_train.append(All_Rating[i])

	User_valid = np.array(User_valid)
	Movie_valid = np.array(Movie_valid)
	Rating_valid = np.array(Rating_valid)
	User_train = np.array(User_train)
	Movie_train = np.array(Movie_train)
	Rating_train = np.array(Rating_train)

	print("User_id shape: ",User_id.shape)
	print("Movie_id shape: ",Movie_id.shape)

	user_input = Input(shape=[1])
	item_input = Input(shape=[1])
	user_vec = Embedding(input_dim=User_id.shape[0], output_dim=64, embeddings_initializer='random_normal',embeddings_regularizer=l2(0.00001))(user_input)
	user_vec = Flatten()(user_vec)
	user_vec = Dropout(0.5)(user_vec)
	item_vec = Embedding(input_dim=3952, output_dim=64, embeddings_initializer='random_normal',embeddings_regularizer=l2(0.00001))(item_input)
	item_vec = Flatten()(item_vec)
	item_vec = Dropout(0.5)(item_vec)
	user_bias = Embedding(input_dim=User_id.shape[0], output_dim=1, embeddings_initializer='zeros',embeddings_regularizer=l2(0.00001))(user_input)
	user_bias = Flatten()(user_bias)
	item_bias = Embedding(input_dim=3952, output_dim=1, embeddings_initializer='zeros',embeddings_regularizer=l2(0.00001))(item_input)
	item_bias = Flatten()(item_bias)
	r_hat = Dot(axes=1)([user_vec, item_vec])
	r_hat = Add()([r_hat,user_bias,item_bias])
	r_hat = Dense(1, bias_initializer='ones', activation='relu')(r_hat)
	#r_hat = Dense(1, bias_initializer='ones', activation='linear')(r_hat)
	adam = Adam()
	model = keras.models.Model([user_input, item_input], r_hat)
	model.compile(loss='mse', optimizer='sgd')
	print(model.summary())
	checkpointer = ModelCheckpoint(filepath='MF_best.h5',
                                       monitor='val_loss',save_best_only=True,
                                       verbose=1, mode='min')
	earlystopping = EarlyStopping(monitor='val_loss', patience = 8, verbose=1, mode='min')
	model.fit([User_train, Movie_train], Rating_train,
            batch_size=256, epochs=5000,
            validation_data=([User_valid, Movie_valid], Rating_valid),
            callbacks=[checkpointer,earlystopping],
            verbose=1)



if __name__ == '__main__':
    main()
