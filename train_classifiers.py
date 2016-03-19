#!/usr/bin/env python
__author__ = "Pavez J. <juan.pavezs@alumnos.usm.cl>"


import ROOT
import numpy as np
from sklearn import svm, linear_model
from sklearn.externals import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation
from xgboost_wrapper import XGBoostClassifier

import sys

from os import listdir
from os.path import isfile, join
import os.path

from utils import printMultiFrame, printFrame, saveFig, loadData

'''
  Train each one of the classifiers on the data ./data/c1/{train|test}_i_j.dat
  each model is saved on the file ./model/c1/{model_file}_i_j.pkl
'''


def trainClassifiers(clf, nsamples,
                     model_g='mlp', c1_g='',
                     dir='/afs/cern.ch/user/j/jpavezse/systematics',
                     model_file='adaptive',
                     dataset_names=None,
                     full_names=None,
                     data_file='train',
                     seed=1234,
                     index=None,
                     vars_names=None
                     ):
    '''
      Train classifiers pair-wise on
      datasets
    '''
    print 'Training classifier'
    for k in range(nsamples):
        for j in range(nsamples):
            if k == j or k > j:
                continue
            if dataset_names is not None:
                name_k, name_j = (dataset_names[k], dataset_names[j])
            else:
                name_k, name_j = (k, j)
            print " Training Classifier on {0}/{1}".format(name_k, name_j)
            traindata, targetdata = loadData(data_file, name_k, name_j, dir=dir, c1_g=c1_g)
            if model_g == 'mlp':
                clf.fit(traindata,
                        targetdata,
                        save_file='{0}/model/{1}/{2}/{3}_{4}_{5}.pkl'.format(dir,
                                                                             model_g,
                                                                             c1_g,
                                                                             model_file,
                                                                             k,
                                                                             j))
            else:
                rng = np.random.RandomState(seed)
                indices = rng.permutation(traindata.shape[0])
                traindata = traindata[indices]
                targetdata = targetdata[indices]
                scores = cross_validation.cross_val_score(clf, traindata.reshape(
                    traindata.shape[0], traindata.shape[1]), targetdata)
                print "Accuracy: {0} (+/- {1})".format(scores.mean(), scores.std() * 2)
                clf.fit(
                    traindata.reshape(
                        traindata.shape[0],
                        traindata.shape[1]),
                    targetdata)
                joblib.dump(clf,
                            '{0}/model/{1}/{2}/{3}_{4}_{5}.pkl'.format(dir,
                                                                       model_g,
                                                                       c1_g,
                                                                       model_file,
                                                                       k,
                                                                       j))
    print " Training Classifier on F0/F1"
    traindata, targetdata = loadData(data_file, 'F0' if full_names is None else full_names[0],
                                     'F1' if full_names is None else full_names[1], dir=dir, c1_g=c1_g)
    if model_g == 'mlp':
        clf.fit(traindata,
                targetdata,
                save_file='{0}/model/{1}/{2}/{3}_F0_F1.pkl'.format(dir,
                                                                   model_g,
                                                                   c1_g,
                                                                   model_file))
    else:
        rng = np.random.RandomState(seed)
        indices = rng.permutation(traindata.shape[0])
        traindata = traindata[indices]
        targetdata = targetdata[indices]
        scores = cross_validation.cross_val_score(clf, traindata, targetdata)
        print "Accuracy: {0} (+/- {1})".format(scores.mean(), scores.std() * 2)
        clf.fit(traindata, targetdata)
        joblib.dump(
            clf, '{0}/model/{1}/{2}/{3}_F0_F1.pkl'.format(dir, model_g, c1_g, model_file))


def predict(filename, traindata, model_g='mlp', sig=1, clf=None):
    sfilename, k, j = filename.split('/')[-1].split('_')
    sfilename = '/'.join(filename.split('/')[:-1]) + '/' + sfilename
    j = j.split('.')[0]
    sig = 1
    if k != 'F0':
        k = int(k)
        j = int(j)
        sig = 1 if k < j else 0
        filename = '{0}_{1}_{2}.pkl'.format(sfilename, min(k, j), max(k, j))
    if clf is not None:
        return clf.predict_proba(traindata, model_file=filename)[:, sig]
    else:
        clf = joblib.load(filename)
        if clf.__class__.__name__ == 'NuSVR':
            output = clf.predict(traindata)
            return np.clip(output, 0., 1.)
        else:
            return clf.predict_proba(traindata)[:, sig]
