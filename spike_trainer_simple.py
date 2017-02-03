# Imports
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
import numpy

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

from keras.models import load_model

import ipdb
import sys

# 1 set input

# set output

# Create Network

class spike_trainer():
    """ create a spike training object. """

    def __init__(self,input_signal,output_spikes,exp_name='default_exp',load_weights=True,load_network=True,nb_epoch=10,batch_size=128,seq_length = 100):
        
        self.input = input_signal
        self.output = output_spikes

        self.model = Sequential()
        self.nb_epoch=nb_epoch
        self.load_weights = load_weights
        self.load_network = load_network
        self.exp_name = exp_name
        
        self.model_filename = self.exp_name + "_model.h5"
        self.weights_filename = self.exp_name + "_weights.h5"
        
        self.batch_size=batch_size
        self.seq_length = 100

        filepath=self.exp_name + "-{epoch:02d}-{loss:.4f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        self.callbacks_list = [checkpoint]
      


        # Assign X and Y
        self.X = []
        self.y = []

        self._process_data()
        self._create_network()

    def _save_weights(self):
        print "Saving weights..."
        self.model.save_weights(self.weights_filename)
        print "Saved to %s" % self.weights_filename

    def _load_weights(self):
        print "Loading weights..."
        self.model.load_weights(self.weights_filename)
        print "Loaded from %s" % self.weights_filename

    def _process_data(self):
        # Get number of output states (chars)(0/1)
        output_classes = 2#len(set(self.output))

        # summarize the loaded data
        n_chars = self.output.shape[0]
        n_vocab = 2 #len(output_classes) # Hard Coded Spike Output

        # prepare the dataset of input to output pairs encoded as integers
        self.seq_length = 100
        dataX = []
        dataY = []
        X = numpy.zeros((n_chars-self.seq_length,self.seq_length,self.input.shape[1]))
        y = numpy.zeros((n_chars-self.seq_length,1,self.input.shape[1]))
        # for every char in the input string
        for i in range(0, n_chars - self.seq_length, 1):        
            # create the input sequence
            seq_in = self.input[i:i + self.seq_length]
            # create the output result
            seq_out = self.output[i + self.seq_length]
            # append the float version of the input vecot to dataX
            dataX.append(seq_in)
            X[i,:,:] = seq_in
            y[i,:,:] = seq_out
            # append the float version of the output to dataY
            dataY.append(seq_out[0])

        # Calculate the number of patterns
        self.n_patterns = len(dataX)

        # normalize
        self.X = X #/ float(n_vocab)
        # one hot encode the output variablec
        self.y = np_utils.to_categorical(dataY)

    def _create_network(self):

        def create_new_network(dims):
            #self.model.add(LSTM(256,input_shape=(self.X.shape[1], self.X.shape[2])))
            model = Sequential()
            model.add(LSTM(256, return_sequences=True,input_shape=(dims[1], dims[2])))
            model.add(LSTM(256))
            model.add(Dropout(0.2))
            model.add(Dense(self.y.shape[1], activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adam')

            # define the checkpoint
            return model


        if self.load_network:
            try:
                print  "Loading Model"
                self.model = load_model(self.model_filename)
                print  "Loaded from %s" % self.model_filename
            except:
                print "Failed to load model"
                ipdb.set_trace()
                self.model = create_new_network(self.X.shape)
                self.model.save(self.model_filename)
                print "Created new model"
        else:
                self.model  = create_new_network(self.X.shape)
                print "Created new model"
                self.model.save(self.model_filename)
        if self.load_weights:
            try:
                self._load_weights()
            except:
                print "Failed to load weights"

    def fit_model(self):
        self.model.fit(self.X, self.y, nb_epoch=self.nb_epoch, batch_size=self.batch_size, callbacks=self.callbacks_list)
        self._save_weights()

    def generate(self):
        """ from a random input, generate stereotypical output """
        # pick a random seed
        #pdb.set_trace()
        start = numpy.random.randint(0, self.X.shape[0]-1)
        pattern = self.X[start,:,:]
        pattern = numpy.reshape(self.X[start],(1,self.X.shape[1],self.X.shape[2]))
        output = []
        # generate characters
        for i in range(100):
            x = pattern
            prediction = self.model.predict(x, verbose=0)
            index = numpy.argmax(prediction)
            sys.stdout.write(str(index))
            output.append(index)
            pattern = numpy.reshape(self.X[start],(1,self.X.shape[1],self.X.shape[2]))
            pattern[0,self.seq_length-1,0] = index
    
        print "\nGeneration Complete."
        return output

    def compute(self,input_signal,seq_length=100):
        """ from a set of input input, generate predicted output """
        # pick a random seed
        #pdb.set_trace()
        print "\nComputing Model"
        output = []
        # generate characters
        for i in range(len(input_signal)-seq_length):
            x = input_signal[i:seq_length+i]
            x = numpy.reshape(x,(1,self.X.shape[1],self.X.shape[2]))
            prediction = self.model.predict(x, verbose=0)
            index = numpy.argmax(prediction)
            #sys.stdout.write(str(index))
            output.append(index)
    
        print "\nCompute Complete."
        return output
  


def main():
    # Create Input
    input_len = 10000
    input_signal = numpy.zeros((input_len,1))

    input_signal[::3,0] = 1


    # Create Output
    output_signal = numpy.zeros((input_len,1))
    output_signal[1::3,0] = 1

    st = spike_trainer(input_signal,output_signal,nb_epoch=3)
    st.fit_model()
    st.generate()

if __name__ == "__main__":
    main()
        



