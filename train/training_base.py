


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


## to call it from cammand lines
import sys
import os
from argparse import ArgumentParser
import shutil

# argument parsing and bookkeeping
#from Losses import *

class training_base(object):
    
    def __init__(self, 
                 splittrainandtest=0.8,
                 useweights=False,
                 testrun=False,
                 args=None):
        
        import matplotlib
        #if no X11 use below
        matplotlib.use('Agg')
        import imp
        try:
            imp.find_module('setGPU')
            import setGPU
        except:
            found = False
            
        
        import keras
                
        self.keras_inputs=[]
        self.keras_inputsshapes=[]
        self.keras_model=None
        self.train_data=None
        self.val_data=None
        self.startlearningrate=None
        self.trainedepoches=0
        self.compiled=False
        self.checkpointcounter=0

        if args is None:
            parser = ArgumentParser('Run the training')
            parser.add_argument('inputDataCollection')
            parser.add_argument('outputDir')
            args = parser.parse_args()

        self.inputData = os.path.abspath(args.inputDataCollection)
        self.outputDir = args.outputDir
        

        # create output dir
        
        isNewTraining=True
        if os.path.isdir(self.outputDir):
            var = raw_input('output dir exists. To recover a training, please type "yes"\n')
            if not var == 'yes':
                raise Exception('output directory must not exists yet')
            isNewTraining=False     
        else:
	        os.mkdir(self.outputDir)
        self.outputDir = os.path.abspath(self.outputDir)
        self.outputDir+='/'
        
        from DataCollection import DataCollection
        #copy configuration to output dir
        #if isNewTraining:
        #    djsource= os.environ['DEEPJET']
        #    shutil.copytree(djsource+'/modules/models', self.outputDir+'models')
        #    #shutil.copyfile(sys.argv[0],self.outputDir+sys.argv[0])

            
            
        self.train_data=DataCollection()
        self.train_data.readFromFile(self.inputData)
        self.train_data.useweights=useweights
        
        if testrun:
            self.train_data.split(0.02)
            
        self.val_data=self.train_data.split(splittrainandtest)
        
        self.train_data.writeToFile(self.outputDir+'trainsamples.dc')
        self.val_data.writeToFile(self.outputDir+'valsamples.dc')


        shapes=self.train_data.getInputShapes()
        
        self.keras_inputs=[]
        self.keras_inputsshapes=[]
        
        print(shapes)
        
        for s in shapes:
            self.keras_inputs.append(keras.layers.Input(shape=s))
            self.keras_inputsshapes.append(s)
            
        if not isNewTraining:
            self.loadModel(self.outputDir+'KERAS_check_last_model.h5')
            self.trainedepoches=sum(1 for line in open(self.outputDir+'losses.log'))
        
    def modelSet(self):
        return not self.keras_model==None
        
    def setModel(self,model,**modelargs):
        if len(self.keras_inputs)<1:
            raise Exception('setup data first') #can't happen
        self.keras_model=model(self.keras_inputs,
                               self.train_data.getNClassificationTargets(),
                               self.train_data.getNRegressionTargets(),
                               **modelargs)
    
    def saveCheckPoint(self,addstring=''):
        
        self.checkpointcounter=self.checkpointcounter+1 
        self.saveModel("KERAS_model_checkpoint_"+str(self.checkpointcounter)+"_"+addstring +".h5")    
           
        
    def loadModel(self,filename):
        #import h5py
        #f = h5py.File(filename, 'r+')
        #del f['optimizer_weights']
        from keras.models import load_model
        self.keras_model=load_model(filename, custom_objects=global_loss_list)
        self.compiled=True
        
    def compileModel(self,
                     learningrate,
                     **compileargs):
        if not self.keras_model:
            raise Exception('set model first') #can't happen
        #if self.compiled:
        #    return
        from keras.optimizers import Adam
        self.startlearningrate=learningrate
        adam = Adam(lr=self.startlearningrate)
        self.keras_model.compile(optimizer=adam,**compileargs)
        self.compiled=True
        
    def saveModel(self,outfile):
        self.keras_model.save(self.outputDir+outfile)
        import tensorflow as tf
        import keras.backend as K
        tfsession=K.get_session()
        saver = tf.train.Saver()
        tfoutpath=self.outputDir+outfile+'_tfsession/tf'
        import os
        os.system('rm -f '+tfoutpath)
        os.system('mkdir -p '+tfoutpath)
        saver.save(tfsession, tfoutpath)


        #import h5py
        #f = h5py.File(self.outputDir+outfile, 'r+')
        #del f['optimizer_weights']
        #f.close()
        
    def trainModel(self,
                   nepochs,
                   batchsize,
                   stop_patience=300, 
                   lr_factor=0.5,
                   lr_patience=2, 
                   lr_epsilon=0.003, 
                   lr_cooldown=6, 
                   lr_minimum=0.000001,
                   maxqsize=20, 
                   **trainargs):
        
        
        self.train_data.setBatchSize(batchsize)
        self.val_data.setBatchSize(batchsize)
        
        self.keras_model.save(self.outputDir+'KERAS_check_last_model.h5')
        
        from DeepJet_callbacks import DeepJet_callbacks
        
        callbacks=DeepJet_callbacks(stop_patience=stop_patience, 
                                    lr_factor=lr_factor,
                                    lr_patience=lr_patience, 
                                    lr_epsilon=lr_epsilon, 
                                    lr_cooldown=lr_cooldown, 
                                    lr_minimum=lr_minimum,
                                    outputDir=self.outputDir)
        nepochs=nepochs-self.trainedepoches
        
        self.keras_model.fit_generator(self.train_data.generator() ,
                            steps_per_epoch=self.train_data.getNBatchesPerEpoch(), 
                            epochs=nepochs,
                            callbacks=callbacks.callbacks,
                            validation_data=self.val_data.generator(),
                            validation_steps=self.val_data.getNBatchesPerEpoch(), #)#,
                            max_q_size=maxqsize,**trainargs)
        
        
        self.saveModel("KERAS_model.h5")
        
        return self.keras_model, callbacks.history, callbacks #added callbacks
        
        import copy
        #reset all file reads etc
        tmpdc=copy.deepcopy(self.train_data)
        del self.train_data
        self.train_data=tmpdc
        
        return self.keras_model, callbacks.history, callbacks #added callbacks
    
    
        
    def makeRoc(self,callbacks):
    	     
		from sklearn.metrics import roc_curve
		from sklearn.metrics import auc
		from ROCs import predictAndMakeRoc
		from root_numpy import array2root
		traind = self.train_data
		testd = self.val_data
		outputDir = self.outputDir
		model = self.keras_model
    	
    	# summarize history for loss for training and test sample
		plt.figure(1)
		plt.plot(callbacks.history.history['loss'])
		plt.plot(callbacks.history.history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.savefig(self.outputDir+'learningcurve.pdf') 
		plt.close(1)

		plt.figure(2)
		plt.plot(callbacks.history.history['acc'])
		plt.plot(callbacks.history.history['val_acc'])
		plt.title('model accuracy')
		plt.ylabel('acc')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.savefig(self.outputDir+'accuracycurve.pdf')
		plt.close(2)
    	
		features_val=self.train_data.getAllFeatures()
		labels_val=self.train_data.getAllLabels()
		weights_val=self.train_data.getAllWeights()[0]
		spectator_val=self.train_data.getAllSpectators()[0][:,0,4]
		print(spectator_val)

		predict_test = self.keras_model.predict(features_val)

		print(predict_test[:,0])

		fpr, tpr, threshold = roc_curve(labels_val[0][:,0],predict_test[:,0])
		DNNauc = auc(labels_val[0][:,0],predict_test[:,0],True)
		dfpr, dtpr, threshold1 = roc_curve(labels_val[0][:,0],spectator_val) 
		BDTauc = auc(labels_val[0][:,0],spectator_val,True)

		
		plt.figure(3)       
		plt.plot(tpr,fpr,label='deepNN auc='+str(DNNauc))
		plt.plot(dtpr,dfpr,label='double-b CMSSW auc='+str(BDTauc))
		plt.semilogy()
		plt.xlabel("Hbb efficiency")
		plt.ylabel("QCD efficiency")
		plt.ylim(0.001,1)
		plt.grid(True)
		plt.legend()
		plt.savefig(self.outputDir+"test.pdf")
		plt.close(3)
    

# 		y = np.array([0, 1, 0, 1, 1, 1, 1, 1])
# 		scores = np.array([0.1, 0.4, 0.35, 0.8, 0.3, 0.2, 0.2, 1])
# 		fpr, tpr, thresholds = roc_curve(y, scores)
# 		print(fpr)
# 		print(tpr)
# 		print(thresholds)
#     	       
# 		plt.figure(1)       
# 		plt.plot(tpr,fpr,label='testing')
# 		plt.semilogy()
# 		plt.xlabel("b efficiency")
# 		plt.ylabel("BKG efficiency")
# 		plt.ylim(0.001,1)
# 		plt.grid(True)
# 		plt.savefig(self.outputDir+"test.pdf")
# 		plt.close(1)
		
		
		
		return
    	       

        
        
        
            
    
