import sys
import numpy as np
import csv 
import os
import keras
from itertools import islice
from keras.models import Model, Sequential, load_model

from sklearn.model_selection import train_test_split
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

test_path = sys.argv[1]
movie_path = sys.argv[3]
user_path = sys.argv[4]

model = load_model('MF_best.h5')
with open(test_path, 'r') as f:
	f.readline()
	test_user = []
	test_movie = []
	for line in f:
		line = line.strip('\n').split(',')
		test_user.append(int(line[1]))
		test_movie.append(int(line[2]))
test_user = np.array(test_user)
test_movie = np.array(test_movie)
result = np.squeeze(model.predict([test_user, test_movie], verbose=1))


with open(sys.argv[2],'w') as f:
	f.write('TestDataID,Rating\n')
	for i, r in enumerate(result):
		f.write('%d,%f\n' %(i+1, r))



