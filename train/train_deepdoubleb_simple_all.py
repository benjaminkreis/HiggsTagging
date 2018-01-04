import sys
import os
from time import clock

copypath = ['../models',
            '../generator',
            '../train']
for p in reversed(copypath):
    sys.path.insert(0, p)

import keras

#keras.backend.set_image_data_format('channels_last')
import numpy as np
# fix random seed for reproducibility
seed = 42
np.random.seed(seed)
#import setGPU
#os.environ['CUDA_VISIBLE_DEVICES'] = '6'
from training_base import training_base
import sys
from DataCollection import DataCollection

class MyClass:
    """A simple example class"""
    def __init__(self):
        self.inputDataCollection = ''
        self.outputDir = ''


args = MyClass()
args.inputDataCollection = 'data/convert_20170717_ak8_deepDoubleB_simple10_train_val/dataCollection.dc'
args.outputDir = 'train_deep_simple_all/'

#also does all the parsing
#train=training_base(testrun=False,args=args)

traind=DataCollection()
traind.readFromFile(args.inputDataCollection)

if os.path.isdir(args.outputDir):
    raise Exception('output directory must not exists yet')
else: 
    os.mkdir(args.outputDir)
    

NENT = 1
features_val = [fval[::NENT] for fval in traind.getAllFeatures()]
labels_val=traind.getAllLabels()[0][::NENT,:]
#weights_val=traind.getAllWeights()[0][::NENT]
#spectators_val = traind.getAllSpectators()[0][::NENT,0,:]
print features_val[0].shape
print labels_val.shape

#X_train_val = features_val[0][:,0,:]
#y_train_val = labels_val[:,1]
#print X_train_val.shape
#print y_train_val.shape

from sklearn.model_selection import train_test_split

#X_train_val, X_test, y_train_val, y_test = train_test_split(features_val[0][:,0,0:10], labels_val[:,1], test_size=0.2, random_state=42)
X_train_val = features_val[0][:,0,0:10]
y_train_val = labels_val[:,1]
print X_train_val.shape
print y_train_val.shape
#print X_test.shape
#print y_test.shape


from models import two_layer_model

from keras.optimizers import Adam, Nadam
from keras.layers import Input

keras_model = two_layer_model(Input(shape=(10,)))

startlearningrate=0.0001
adam = Adam(lr=startlearningrate)
keras_model.compile(optimizer=adam, loss=['binary_crossentropy'], metrics=['accuracy'])

from DeepJet_callbacks import DeepJet_callbacks
        
callbacks=DeepJet_callbacks(stop_patience=1000, 
                            lr_factor=0.5,
                            lr_patience=10,
                            lr_epsilon=0.000001, 
                            lr_cooldown=2, 
                            lr_minimum=0.0000001,
                            outputDir=args.outputDir)

keras_model.fit(X_train_val, y_train_val, batch_size = 1024, epochs = 10,
                validation_split = 0.25, shuffle = True, callbacks = callbacks.callbacks)


from keras.utils import plot_model
plot_model(keras_model, to_file='model.png')

import h5py
h5File = h5py.File('train_deep_simple_all//KERAS_check_model_last_weights.h5')
biases = h5File['/dense_1/dense_1/bias:0'][()]
weights = h5File['/dense_1/dense_1/kernel:0'][()]
biases2 = h5File['/dense_2/dense_2/bias:0'][()]
weights2 = h5File['/dense_2/dense_2/kernel:0'][()]

print "biases", biases
#print "weights", weights
#print weights for HLS
print repr(weights)

print "biases2", biases2
print "weights2", weights2

#print "X_train_val[100:101]", X_train_val[100:101]

#### CHECK MATH
#############################
## NEED TO ADD SIGMOID FIXME
#layer1_out = np.dot(X_train_val[100:101],weights)+biases
#print "np.dot(X_train_val[100:101],weights)+biases", layer1_out

#relu
#for x in np.nditer(layer1_out, op_flags=['readwrite']):
#    if x<=0: 
#        x[...] = 0

#print "np.dot(layer1_out, weights2)+biases2", np.dot(layer1_out, weights2)+biases2


#### KERAS PREDICT
###########################
#print "prediction", keras_model.predict(X_train_val[100:101])
#t0 = clock()
#my_prediction = keras_model.predict(X_train_val[100:101])
#t1 = clock()
#print("%.15f seconds" % (t1-t0))

t0 = clock()
my_prediction = keras_model.predict(X_train_val[0:10000])
t1 = clock()
print("%.15f seconds" % (t1-t0))
print "prediction", my_prediction

#print "X_train_val", X_train_val[0:1000]
np.savetxt('in.dat',X_train_val[0:10000])
np.savetxt('keras.dat',my_prediction)

print "truth", y_train_val[100:101]
