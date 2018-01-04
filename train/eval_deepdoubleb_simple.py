import sys
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
keras.backend.set_image_data_format('channels_first')
from keras.models import load_model, Model
from argparse import ArgumentParser
from keras import backend as K
import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from root_numpy import array2root
import pandas as pd


def makeRoc(testd, model, outputDir):
    print 'in makeRoc()'
    
    # let's use only first 1000 entries
    NENT = 10000
    features_val = [fval[:NENT] for fval in testd.getAllFeatures()]
    labels_val=testd.getAllLabels()[0][:NENT,:]
    weights_val=testd.getAllWeights()[0][:NENT]
    spectators_val = testd.getAllSpectators()[0][:NENT,0,:]
    print features_val[0].shape
    df = pd.DataFrame(spectators_val)
    df.columns = ['fj_pt',
                  'fj_eta',
                  'fj_sdmass',
                  'fj_n_sdsubjets',
                  'fj_doubleb',
                  'fj_tau21',
                  'fj_tau32',
                  'npv',
                  'npfcands',
                  'ntracks',
                  'nsv']

    print(df.iloc[:10])

        
    predict_test = model.predict(features_val[0][:,0,:])
    df['fj_isH'] = labels_val[:,1]
    df['fj_deepdoubleb'] = predict_test

    print(df.iloc[:10])

    #df['fj_doubleb'] is BDT.  replace this with text file
    #res_32_12_data = pd.read_csv("res_32-12.dat", header=None)
    res_32_12_data = pd.read_csv("res_16-10.dat", header=None)
    res_16_8_data = pd.read_csv("res_16-8.dat", header=None)
    res_16_3_data = pd.read_csv("res_16-3.dat", header=None)
    res_8_4_data = pd.read_csv("res_8-4.dat", header=None)
    res_6_4_data = pd.read_csv("res_6-4.dat", header=None)
    #print df['fj_deepdoubleb']
    #print df['fj_doubleb']


    fpr, tpr, threshold = roc_curve(df['fj_isH'],df['fj_deepdoubleb'])
    #dfpr, dtpr, threshold1 = roc_curve(df['fj_isH'],df['fj_doubleb'])
    dfpr_32_12, dtpr_32_12, threshold1_32_12 = roc_curve(df['fj_isH'],res_32_12_data)
    dfpr_16_8, dtpr_16_8, threshold1_16_8 = roc_curve(df['fj_isH'],res_16_8_data)
    dfpr_16_3, dtpr_16_3, threshold1_16_3 = roc_curve(df['fj_isH'],res_16_3_data)
    dfpr_8_4, dtpr_8_4, threshold1_8_4 = roc_curve(df['fj_isH'],res_8_4_data)
    dfpr_6_4, dtpr_6_4, threshold1_6_4 = roc_curve(df['fj_isH'],res_6_4_data)

    def find_nearest(array,value):
        idx = (np.abs(array-value)).argmin()
        return idx, array[idx]

    value = 0.01 # 1% mistag rate
    idx, val = find_nearest(fpr, value)
    deepdoublebcut = threshold[idx] # threshold for deep double-b corresponding to ~1% mistag rate
    print('deep double-b > %f coresponds to %f%% QCD mistag rate'%(deepdoublebcut,100*val))

    auc1 = auc(fpr, tpr)
    #auc2 = auc(dfpr, dtpr)
    auc_32_12 = auc(dfpr_32_12, dtpr_32_12)
    auc_16_8 = auc(dfpr_16_8, dtpr_16_8)
    auc_16_3 = auc(dfpr_16_3, dtpr_16_3)
    auc_8_4 = auc(dfpr_8_4, dtpr_8_4)
    auc_6_4 = auc(dfpr_6_4, dtpr_6_4)

    plt.figure()       
    plt.plot(tpr,fpr,label='CPU floating point, auc = %.1f%%'%(auc1*100))
    plt.plot(dtpr_32_12,dfpr_32_12,label='FPGA fixed<32,12>, auc = %.1f%%'%(auc_32_12*100))
    plt.plot(dtpr_16_8,dfpr_16_8,label='FPGA fixed<16,8>, auc = %.1f%%'%(auc_16_8*100))
    #plt.plot(dtpr_16_3,dfpr_16_3,label='FPGA fixed<16,3>, auc = %.1f%%'%(auc_16_3*100))
    #plt.plot(dtpr_8_4,dfpr_8_4,label='FPGA fixed<8,4>, auc = %.1f%%'%(auc_8_4*100))
    #plt.plot(dtpr_6_4,dfpr_6_4,label='FPGA fixed<6,4>, auc = %.1f%%'%(auc_6_4*100))
    plt.semilogy()
    plt.xlabel("H(bb) efficiency")
    plt.ylabel("QCD mistag rate")
    plt.ylim(0.001,1)
    plt.grid(True)
    plt.legend()
    plt.savefig(outputDir+"ROC.pdf")
    
    plt.figure()
    bins = np.linspace(-1,1,70)
    plt.hist(df['fj_doubleb'], bins=bins, weights = 1-df['fj_isH'],alpha=0.5,normed=True,label='QCD')
    plt.hist(df['fj_doubleb'], bins=bins, weights = df['fj_isH'],alpha=0.5,normed=True,label='H(bb)')
    plt.xlabel("BDT double-b")
    plt.legend(loc='upper left')
    plt.savefig(outputDir+"doubleb.pdf")
    
    plt.figure()
    bins = np.linspace(0,1,70)
    plt.hist(df['fj_deepdoubleb'], bins=bins, weights = 1-df['fj_isH'],alpha=0.5,normed=True,label='QCD')
    plt.hist(df['fj_deepdoubleb'], bins=bins, weights = df['fj_isH'],alpha=0.5,normed=True,label='H(bb)')
    plt.xlabel("deep double-b")
    #plt.ylim(0.00001,1)
    #plt.semilogy()
    plt.legend(loc='upper left')
    plt.savefig(outputDir+"deepdoubleb.pdf")
    
    plt.figure()
    bins = np.linspace(0,2000,70)
    plt.hist(df['fj_pt'], bins=bins, weights = 1-df['fj_isH'],alpha=0.5,normed=True,label='QCD')
    plt.hist(df['fj_pt'], bins=bins, weights = df['fj_isH'],alpha=0.5,normed=True,label='H(bb)')
    plt.xlabel(r'$p_{\mathrm{T}}$')
    plt.legend(loc='upper left')
    plt.savefig(outputDir+"pt.pdf")
    
    plt.figure()
    bins = np.linspace(0,200,70)
    plt.hist(df['fj_sdmass'], bins=bins, weights = 1-df['fj_isH'],alpha=0.5,normed=True,label='QCD')
    plt.hist(df['fj_sdmass'], bins=bins, weights = df['fj_isH'],alpha=0.5,normed=True,label='H(bb)')
    plt.xlabel(r'$m_{\mathrm{SD}}$')
    plt.legend(loc='upper left')
    plt.savefig(outputDir+"msd.pdf")
    
    plt.figure()
    bins = np.linspace(0,200,70)
    df_passdoubleb = df[df.fj_doubleb > 0.9]
    plt.hist(df_passdoubleb['fj_sdmass'], bins=bins, weights = 1-df_passdoubleb['fj_isH'],alpha=0.5,normed=True,label='QCD')
    plt.hist(df_passdoubleb['fj_sdmass'], bins=bins, weights = df_passdoubleb['fj_isH'],alpha=0.5,normed=True,label='H(bb)')
    plt.xlabel(r'$m_{\mathrm{SD}}$')
    plt.legend(loc='upper left')
    plt.savefig(outputDir+"msd_passdoubleb.pdf")
    
    plt.figure()
    bins = np.linspace(0,200,70)
    df_passdeepdoubleb = df[df.fj_deepdoubleb > deepdoublebcut]
    plt.hist(df_passdeepdoubleb['fj_sdmass'], bins=bins, weights = 1-df_passdeepdoubleb['fj_isH'],alpha=0.5,normed=True,label='QCD')
    plt.hist(df_passdeepdoubleb['fj_sdmass'], bins=bins, weights = df_passdeepdoubleb['fj_isH'],alpha=0.5,normed=True,label='H(bb)')
    plt.xlabel(r'$m_{\mathrm{SD}}$')
    plt.legend(loc='upper left')
    plt.savefig(outputDir+"msd_passdeepdoubleb.pdf")
    
    return df

#os.environ['CUDA_VISIBLE_DEVICES'] = '7'

inputModel = 'train_deep_simple_all/KERAS_check_model_last.h5'
outputDir = 'out_deep_simple_all/'
# test data:
#inputDataCollection = 'data/convert_20170717_ak8_deepDoubleB_simple10_test/dataCollection.dc'
inputDataCollection = 'data/convert_20170717_ak8_deepDoubleB_simple10_train_val/dataCollection.dc'
# training data:
#inputDataCollection = '/cms-sc17/convert_deepDoubleB_simple_train_val/dataCollection.dc'

if os.path.isdir(outputDir):
    raise Exception('output directory must not exists yet')
else: 
    os.mkdir(outputDir)

model=load_model(inputModel)
    
#intermediate_output = intermediate_layer_model.predict(data)

#print(model.summary())
    
from DataCollection import DataCollection
    
testd=DataCollection()
testd.readFromFile(inputDataCollection)
    
df = makeRoc(testd, model, outputDir)


def _byteify(data, ignore_dicts = False):
    # if this is a unicode string, return its string representation
    if isinstance(data, unicode):
        return data.encode('utf-8')
    # if this is a list of values, return list of byteified values
    if isinstance(data, list):
        return [ _byteify(item, ignore_dicts=True) for item in data ]
    # if this is a dictionary, return dictionary of byteified keys and values
    # but only if we haven't already byteified it
    if isinstance(data, dict) and not ignore_dicts:
        return {
            _byteify(key, ignore_dicts=True): _byteify(value, ignore_dicts=True)
            for key, value in data.iteritems()
        }
    # if it's anything else, return it in its original form
    return data


import json

f = open('train_deep_simple_all/full_info.log')
myListOfDicts = json.load(f, object_hook=_byteify)
myDictOfLists = {}
for key, val in myListOfDicts[0].iteritems():
    myDictOfLists[key] = []
for i, myDict in enumerate(myListOfDicts):
    for key, val in myDict.iteritems():
        myDictOfLists[key].append(myDict[key])


plt.figure()
val_loss = np.asarray(myDictOfLists['val_loss'])
loss = np.asarray(myDictOfLists['loss'])
plt.plot(val_loss, label='validation')
plt.plot(loss, label='train')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig(outputDir+"loss.pdf")

plt.figure()
val_acc = np.asarray(myDictOfLists['val_acc'])
acc = np.asarray(myDictOfLists['acc'])
plt.plot(val_acc, label='validation')
plt.plot(acc, label='train')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.savefig(outputDir+"acc.pdf")
