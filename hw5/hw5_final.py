import sys, argparse, os
import keras
import _pickle as pk
import readline
import numpy as np
import sys

from keras import regularizers
from keras.models import Model
from keras.layers import Input, GRU, LSTM, Dense, Dropout, Bidirectional, BatchNormalization, Masking
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import string
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from itertools import islice

import keras.backend.tensorflow_backend as K
import tensorflow as tf

from utils.util import DataManager



os.environ["CUDA_VISIBLE_DEVICES"] = "2"

parser = argparse.ArgumentParser(description='Sentiment classification')
parser.add_argument('--model')
parser.add_argument('--action', choices=['train','test','semi'])

# training argument
parser.add_argument('--batch_size', default=512, type=float)
parser.add_argument('--nb_epoch', default=20, type=int)
parser.add_argument('--val_ratio', default=0.1, type=float)
parser.add_argument('--gpu_fraction', default=0.3, type=float)
parser.add_argument('--vocab_size', default=20000, type=int)
parser.add_argument('--max_length', default=40,type=int)

# model parameter
parser.add_argument('--loss_function', default='binary_crossentropy')
parser.add_argument('--cell', default='LSTM', choices=['LSTM','GRU'])
parser.add_argument('-emb_dim', '--embedding_dim', default=256, type=int)
parser.add_argument('-hid_siz', '--hidden_size', default=512, type=int)
parser.add_argument('--dropout_rate', default=0.4, type=float)
parser.add_argument('-lr','--learning_rate', default=0.001,type=float)
parser.add_argument('--threshold', default=0.2,type=float)

# output path for your prediction
parser.add_argument('--result_path', default='result.csv',)

# put model in the same directory
parser.add_argument('--load_model', default = None)
parser.add_argument('--save_dir', default = 'model/')
parser.add_argument('--train_path')
parser.add_argument('--test_path')
parser.add_argument('--semi_path')
parser.add_argument('--pred_path')
parser.add_argument('--mode')

args = parser.parse_args()

train_path = args.train_path
test_path = args.test_path
semi_path = args.semi_path
pred_path = args.pred_path

class DataManager:
    def __init__(self):
        self.data = {}  #means dictionary ~
    # Read data from data_path
    #  name       : string, name of data
    #  with_label : bool, read data with label or without label
    def add_data(self,name, data_path, with_label=True):
        print ('read data from %s...'%data_path)
        X, Y = [], []
        with open(data_path,'r') as f:
            for line in f:
                if with_label:
                    lines = line.strip().split(' +++$+++ ')
                    X.append(lines[1])
                    Y.append(int(lines[0]))
                else:
                    
                    table = str.maketrans({key: None for key in string.punctuation})
                    new_line = line.translate(table)
                    #print(new_line)  
                    new_line = line.strip()
                    X.append(new_line)

        if with_label:
            self.data[name] = [X,Y]
        else:
            self.data[name] = [X]
    def add_test_data(self,name, data_path):
        print ('read data from %s...'%data_path)
        ID, Text = [], []
        with open(data_path,'r') as f:
            for line in islice(f,1,None):
                #line = line.replace()
                comma_index = line.find(',')
                lines = line[comma_index+1:-1]
                table = str.maketrans({key: None for key in string.punctuation})
                new_line = lines.translate(table) 
                new_line = lines.strip()

                #print(lines)
                #ID.append(lines[0])
                Text.append(new_line)
                #print('ID',lines[0])
                #print('Text',lines)

        self.data[name] = [Text]

    # Build dictionary
    #  vocab_size : maximum number of word in yout dictionary
    def tokenize(self, vocab_size):
        print ('create new tokenizer')
        self.tokenizer = Tokenizer(num_words=vocab_size, filters='|"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True, split=" ",char_level=False)
        for key in self.data:
            print ('tokenizing %s'%key)
            texts = self.data[key][0]
            self.tokenizer.fit_on_texts(texts)
        
    # Save tokenizer to specified path
    def save_tokenizer(self, path):
        print ('save tokenizer to %s'%path)
        pk.dump(self.tokenizer, open(path, 'wb'))
            
    # Load tokenizer from specified path
    def load_tokenizer(self,path):
        print ('Load tokenizer from %s'%path)
        self.tokenizer = pk.load(open(path, 'rb'))


    # Convert words in data to index and pad to equal size
    #  maxlen : max length after padding
    def to_sequence(self, maxlen):
        self.maxlen = maxlen
        for key in self.data:
            print ('Converting %s to sequences'%key)
            tmp = self.tokenizer.texts_to_sequences(self.data[key][0])
            self.data[key][0] = np.array(pad_sequences(tmp, maxlen=maxlen))
    
    # Convert texts in data to BOW feature
    def to_bow(self):
        for key in self.data:
            print ('Converting %s to tfidf'%key)
            self.data[key][0] = self.tokenizer.texts_to_matrix(self.data[key][0],mode='count')
    
    # Convert label to category type, call this function if use categorical loss
    def to_category(self):
        for key in self.data:
            if len(self.data[key]) == 2:
                self.data[key][1] = np.array(to_categorical(self.data[key][1]))


    def get_semi_data(self,name,label,threshold,loss_function) : 
        # if th==0.3, will pick label>0.7 and label<0.3
        label = np.squeeze(label)
        index = (label>1-threshold) + (label<threshold)
        semi_X = self.data[name][0]
        semi_Y = np.greater(label, 0.5).astype(np.int32)
        if loss_function=='binary_crossentropy':
            return semi_X[index,:], semi_Y[index]
        elif loss_function=='categorical_crossentropy':
            return semi_X[index,:], to_categorical(semi_Y[index])
        else :
            raise Exception('Unknown loss function : %s'%loss_function)


    # get data by name
    def get_data(self,name):
        return self.data[name]

    # split data to two part by a specified ratio
    #  name  : string, same as add_data
    #  ratio : float, ratio to split
    def split_data(self, name, ratio):
        data = self.data[name]
        X = data[0]
        Y = data[1]
        data_size = len(X)
        val_size = int(data_size * ratio)
        return (X[val_size:],Y[val_size:]),(X[:val_size],Y[:val_size])


# build model
def simpleRNN(args):
    inputs = Input(shape=(args.max_length,))

    # Embedding layer
    embedding_inputs = Embedding(args.vocab_size,              #embedding layer = vocab_size * embedding_dim
                                 args.embedding_dim, 
                                 trainable=True)(inputs)
    print(type(embedding_inputs))
   
    # RNN 
    return_sequence = False
    dropout_rate = args.dropout_rate
    if args.cell == 'GRU':
        RNN_cell = GRU(args.hidden_size, 
                       return_sequences=True, 
                       dropout=dropout_rate)(embedding_inputs)
        RNN_cell = GRU(args.hidden_size, 
                       return_sequences=True, 
                       dropout=dropout_rate)(embedding_inputs)

        RNN_output = GRU(args.hidden_size, 
                       return_sequences=False, 
                       dropout=dropout_rate)(RNN_cell)

    elif args.cell == 'LSTM':
        RNN_output = Bidirectional(LSTM(args.hidden_size, 
                        return_sequences=True, 
                        kernel_initializer='glorot_uniform',
                        recurrent_initializer='orthogonal',
                        dropout=dropout_rate))(embedding_inputs)
        #print(type(RNN_LSTM1))
        print(RNN_output.shape)
        RNN_output = Bidirectional(LSTM(args.hidden_size, 
                        return_sequences=False, 
                        dropout=dropout_rate))(RNN_output)
      

    #RNN_output = RNN_cell(embedding_inputs)
    outputs = BatchNormalization()(RNN_output)
    
    # DNN layer
    
    outputs = Dense(args.hidden_size//2, 
                    activation='relu',
                    kernel_regularizer=regularizers.l2(0.1))(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Dropout(dropout_rate)(outputs)
    #outputs = Dense(args.hidden_size//4, 
    #                activation='relu',
    #                kernel_regularizer=regularizers.l2(0.1))(outputs)
    #outputs = BatchNormalization()(outputs)
    #outputs = Dropout(dropout_rate)(outputs)

    outputs = Dense(1, activation='sigmoid')(outputs)
        
    model =  Model(inputs=inputs,outputs=outputs)

    # optimizer
    adam = Adam()
    print ('compile model...')

    # compile model
    model.compile( loss=args.loss_function, optimizer=adam, metrics=[ 'accuracy',])
    
    return model

def main():
    # limit gpu memory usage
    def get_session(gpu_fraction):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))  
    K.set_session(get_session(args.gpu_fraction))
    
    save_path = os.path.join(args.save_dir,args.model)
    if args.load_model is not None:
        load_path = os.path.join(args.save_dir,args.load_model)



 #####read data#####
    dm = DataManager()
    print ('Loading data...')
    if args.action == 'train':
        dm.add_data('train_data', train_path, True)
    elif args.action == 'semi':
        dm.add_data('train_data', train_path, True)
        dm.add_data('semi_data', semi_path, False)
    elif args.action == 'test':
        #dm.add_data('train_data', train_path, False)
        dm.add_test_data('test_data', test_path)
    else:
        raise Exception ('Implement your testing parser LUL')
            
    # prepare tokenizer
    print ('get Tokenizer...')
    if args.load_model is not None:
        # read exist tokenizer
        print("read exist tokenizer")
        dm.load_tokenizer(os.path.join(load_path,'token.pk'))
    else:
        # create tokenizer on new data
        print("create tokenizer on new data")
        dm.tokenize(args.vocab_size)
                            
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    if not os.path.exists(os.path.join(save_path,'token.pk')):
        dm.save_tokenizer(os.path.join(save_path,'token.pk')) 

    # convert to sequences
    dm.to_sequence(args.max_length)
    #dm.to_bow()

# initial model
    print ('initial model...')
    model = simpleRNN(args)    
    print (model.summary())

    if args.load_model is not None:
        if args.action == 'train':
            print ('Warning : load a exist model and keep training')
        path = os.path.join(load_path,'model.h5')
        if os.path.exists(path):
            print ('load model from %s' % path)
            model.load_weights(path)
        else:
            raise ValueError("Can't find the file %s" %path)
    elif args.action == 'test':
        print ('load model for testing')
        path = os.path.join(load_path,'model.h5')
        if os.path.exists(path):
            print ('load model from %s' % path)
            model.load_weights(path)
        else:
            raise ValueError("Can't find the file %s" %path)


# training
    if args.action == 'train':
        (X,Y),(X_val,Y_val) = dm.split_data('train_data', args.val_ratio)
        earlystopping = EarlyStopping(monitor='val_acc', patience = 5, verbose=1, mode='max')

        save_path = os.path.join(save_path,'model.h5')
        checkpoint = ModelCheckpoint(filepath=save_path, 
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     monitor='val_acc',
                                     mode='max' )

        learning_rate_funtion = ReduceLROnPlateau(monitor='val_acc',patience=2,verbose=1,factor=0.5,min_lr=0.000001)


        history = model.fit(X, Y, 
                            validation_data=(X_val, Y_val),
                            epochs=args.nb_epoch, 
                            batch_size=args.batch_size,
                            callbacks=[checkpoint, earlystopping] )

# testing
    elif args.action == 'test' :
        #raise Exception ('Implement your testing function')
        [test_X] = dm.get_data('test_data')
        #[labeled_data] = dm.get_data('train_data')
        print('test_X')
        print('len for test_X: ',len(test_X))
        print(test_X)
        test_pred = model.predict(test_X,batch_size=1024, verbose=True)
        print("test_pred")
        print(test_pred)

        test_pred[test_pred>=0.5]=1
        test_pred[test_pred<0.5]=0
        print(test_pred)

        f = open(pred_path, 'w')
        f.write('id,label\n')
        for i in range(0, len(test_pred)):
            f.write(str(i)+','+str(int(test_pred[i][0]))+'\n')
        f.close()
        

 # semi-supervised training
    elif args.action == 'semi':
        (X,Y),(X_val,Y_val) = dm.split_data('train_data', args.val_ratio)

        [semi_all_X] = dm.get_data('semi_data')
        earlystopping = EarlyStopping(monitor='val_acc', patience = 5, verbose=1, mode='max')

        save_path = os.path.join(save_path,'model.h5') ##!!!!
        checkpoint = ModelCheckpoint(filepath=save_path, 
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     monitor='val_acc',
                                     mode='max' )
        # repeat 10 times
        for i in range(5):
            # label the semi-data
            semi_pred = model.predict(semi_all_X, batch_size=1024, verbose=True)
            print("semi_pred")
            print(semi_pred)
            semi_X, semi_Y = dm.get_semi_data('semi_data', semi_pred, args.threshold, args.loss_function)
            semi_X = np.concatenate((semi_X, X))
            semi_Y = np.concatenate((semi_Y, Y))
            print ('-- iteration %d  semi_data size: %d' %(i+1,len(semi_X)))
            # train
            history = model.fit(semi_X, semi_Y, 
                                validation_data=(X_val, Y_val),
                                epochs=2, 
                                batch_size=args.batch_size,
                                callbacks=[checkpoint, earlystopping] )

            if os.path.exists(save_path):
                print ('load model from %s' % save_path)
                model.load_weights(save_path)
            else:
                raise ValueError("Can't find the file %s" %path)


if __name__ == '__main__':
        main()

   

