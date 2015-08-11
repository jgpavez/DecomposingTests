#!/usr/bin/env python
__author__ = "Pavez J. <juan.pavezs@alumnos.usm.cl>"


import ROOT
import numpy as np
from sklearn import svm, linear_model
from sklearn.externals import joblib
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import GradientBoostingClassifier

import sys

import os.path
import pdb

from mlp import make_predictions, train_mlp

from utils import printMultiFrame, printFrame, saveFig, loadData

''' 
  Train each one of the classifiers on the data ./data/c1/{train|test}_i_j.dat
  each model is saved on the file ./model/c1/{model_file}_i_j.pkl
'''

def trainClassifiers(clf,c0,c1,
      workspace='workspace_DecomposingTestOfMixtureModelsClassifiers.root', 
      model_g='mlp',c1_g='',
      dir='/afs/cern.ch/user/j/jpavezse/systematics',
      model_file='adaptive'
    ):
  '''
    Train classifiers pair-wise on 
    datasets
  '''
  print 'Training classifier'

  for k,c in enumerate(c0):
    for j,c_ in enumerate(c1):
      if k==j or k > j:
        continue
      print " Training Classifier on f{0}/f{1}".format(k,j)
      if model_g == 'mlp':
        train_mlp(datatype='train',kpos=k,jpos=j,dir='{0}/data/{1}/{2}'.format(dir,'mlp',c1_g),
          save_file='{0}/model/{1}/{2}/{3}_{4}_{5}.pkl'.format(dir,model_g,c1_g,model_file,k,j))
      else:
        traindata,targetdata = loadData('train',k,j,dir=dir,c1_g=c1_g) 
        clf.fit(traindata.reshape(traindata.shape[0],traindata.shape[1])
            ,targetdata)
        joblib.dump(clf, '{0}/model/{1}/{2}/{3}_{4}_{5}.pkl'.format(dir,model_g,c1_g,model_file,k,j))
  
  print " Training Classifier on F0/F1"
  if model_g == 'mlp':
    train_mlp(datatype='train',kpos='F0',jpos='F1',dir='{0}/data/{1}/{2}'.format(dir,'mlp',c1_g), 
        save_file='{0}/model/{1}/{2}/{3}_F0_F1.pkl'.format(dir,model_g,c1_g,model_file))
  else:
    traindata,targetdata = loadData('train','F0','F1',dir=dir,c1_g=c1_g) 
    #clf = svm.NuSVC(probability=True) #Why use a SVR??
    clf.fit(traindata.reshape(traindata.shape[0],traindata.shape[1])
        ,targetdata)
    joblib.dump(clf, '{0}/model/{1}/{2}/{3}_F0_F1.pkl'.format(dir,model_g,c1_g,model_file))

def predict(filename, traindata,model_g='mlp'):
  sig = 1
  sfilename,k,j = filename.split('_')
  j = j.split('.')[0]
  sig = 1
  if k <> 'F0':
    k = int(k)
    j = int(j)
    sig = 1 if k < j else 0
    filename = '{0}_{1}_{2}.pkl'.format(sfilename,min(k,j),max(k,j))
  if model_g == 'mlp':
    return make_predictions(dataset=traindata, model_file=filename)[:,sig]
  else:
    clf = joblib.load(filename)
    if clf.__class__.__name__ == 'NuSVR':
      output = clf.predict(traindata)
      return np.clip(output,0.,1.)
    else:
      return clf.predict_proba(traindata)[:,sig]

