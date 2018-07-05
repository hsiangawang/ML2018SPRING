import csv,sys
import numpy as np
from keras.models import Sequential
from keras.models import Model
from keras.models import load_model
from keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.layers.embeddings import Embedding
from keras.optimizers import SGD, Adam
from keras.utils import np_utils

# label dictionary
dict = ['Acoustic_guitar', 'Applause', 'Bark', 'Bass_drum', 'Burping_or_eructation',
		'Bus', 'Cello', 'Chime', 'Clarinet', 'Cowbell', 'Computer_keyboard', 'Cough', 
		'Double_bass', 'Drawer_open_or_close', 'Electric_piano', 'Fart', 'Finger_snapping',
		'Fireworks', 'Flute', 'Glockenspiel', 'Gong', 'Gunshot_or_gunfire', 'Harmonica',
		'Hi-hat', 'Keys_jangling', 'Knock', 'Laughter', 'Meow', 'Microwave_oven', 'Oboe',
		'Saxophone', 'Scissors', 'Shatter', 'Snare_drum', 'Squeak', 'Tambourine', 'Tearing',
		'Telephone', 'Trumpet', 'Writing', 'Violin_or_fiddle']

print('read sample submission.csv')
with open('./data/sample_submission.csv','r') as f:
	n_row = 0
	filename = []
	for row in f:
		if n_row != 0:
			r = row.strip().split(',')
			filename.append(r[0])
		n_row += 1

print('read x_test.npy')
x_test = np.load('./data/x_test_log10.npy')

print('load model')
model = load_model('./model/final_model.h5')
print(model.summary())

print('predict')
prediction = model.predict(x_test, verbose=1)
prediction1 = np.argmax(prediction, axis=1)

for i in range(prediction.shape[0]):
	prediction[i,prediction1[i]] = 0

prediction2 = np.argmax(prediction, axis=1)
for i in range(prediction.shape[0]):
	prediction[i,prediction2[i]] = 0

prediction3 = np.argmax(prediction, axis=1)

print('write')
ans = []
for i in range(len(x_test)):
	ans.append([filename[i]])
	ans[i].append(str(dict[prediction1[i]]+' '+dict[prediction2[i]] + ' '+dict[prediction3[i]]))



predict_csv = open(sys.argv[1], "w+")
s = csv.writer(predict_csv,delimiter=',',lineterminator='\n')
s.writerow(["fname","label"])
for i in range(len(ans)):
	s.writerow(ans[i]) 
predict_csv.close()
print('done')
