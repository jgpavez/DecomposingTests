#!/usr/bin/env python
'''
 This python script can be used to reproduce the results on 1D distributions
 on the article Experiments using machine learning to approximate likelihood ratios
 for mixture models (ACAT 2016).
'''

__author__ = "Pavez J. <juan.pavezs@alumnos.usm.cl>"


import ROOT
import numpy as np
from sklearn import svm, linear_model
from sklearn.ensemble import GradientBoostingClassifier

import sys

import os.path

from mlp import MLPTrainer

from make_data import makeData, makeModelND, makeModelPrivateND,\
    makeModel
from utils import printMultiFrame, printFrame, saveFig, loadData,\
    makeROC, makeSigBkg, makePlotName

from train_classifiers import trainClassifiers, predict
from decomposed_test import DecomposedTest

from xgboost_wrapper import XGBoostClassifier


if __name__ == '__main__':
    # Setting the classifier to use
    model_g = None
    classifiers = {
        'svc': svm.NuSVC(
            probability=True),
        'svr': svm.NuSVR(),
        'logistic': linear_model.LogisticRegression(),
        'bdt': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=1.0,
            max_depth=5,
            random_state=0),
        'mlp': MLPTrainer(
            n_hidden=4,
            L2_reg=0),
        'xgboost': XGBoostClassifier(
            num_class=2,
            nthread=4,
            silent=1,
            num_boost_round=100,
            eta=0.5,
            max_depth=4)}
    clf = None
    if (len(sys.argv) > 1):
        model_g = sys.argv[1]
        clf = classifiers.get(sys.argv[1])
    if clf is None:
        model_g = 'logistic'
        clf = classifiers['logistic']
        print 'Not found classifier, Using logistic instead'

    # parameters of the mixture model
    c0 = np.array([.0, .3, .7])
    c1 = np.array([.1, .3, .7])
    c1_g = ''

    c0 = c0 / c0.sum()
    c1[0] = sys.argv[2]
    if c1[0] < 0.01:
        c1_g = "%.3f" % c1[0]
    else:
        c1_g = "%.2f" % c1[0]
    c1[0] = (c1[0] * (c1[1] + c1[2])) / (1. - c1[0])
    c1 = c1 / c1.sum()

    verbose_printing = True
    dir = '.'
    workspace_file = 'workspace_DecomposingTestOfMixtureModelsClassifiers.root'

    # features
    vars_g = ['x']

    ROOT.gROOT.SetBatch(ROOT.kTRUE)
    ROOT.RooAbsPdf.defaultIntegratorConfig().setEpsRel(1E-15)

    # Set this value to False if only final plots are needed
    verbose_printing = True

    if (len(sys.argv) > 3):
        print 'Setting seed: {0} '.format(sys.argv[3])
        ROOT.RooRandom.randomGenerator().SetSeed(int(sys.argv[3]))
        np.random.seed(int(sys.argv[3]))

    # Create models to sample from
    makeModel(c0=c0,c1=c1,workspace=workspace_file,dir=dir,verbose_printing=
        verbose_printing)

    # make sintetic data to train the classifiers
    makeData(vars_g=vars_g,c0=c0,c1=c1,num_train=100000,num_test=50000,
        workspace=workspace_file,dir=dir, c1_g=c1_g, model_g='mlp')

    # train the pairwise classifiers
    trainClassifiers(clf,3,dir=dir, model_g=model_g,
        c1_g=c1_g ,model_file='adaptive')

    # class which implement the decomposed method
    test = DecomposedTest(
        c0,
        c1,
        dir=dir,
        c1_g=c1_g,
        model_g=model_g,
        input_workspace=workspace_file,
        verbose_printing=verbose_printing,
        dataset_names=[
            '0',
            '1',
            '2'],
        clf=clf if model_g == 'mlp' else None)

    test.fit(data_file='test',true_dist=True,vars_g=vars_g)
    test.computeRatios(true_dist=True, vars_g=vars_g, use_log=False)
