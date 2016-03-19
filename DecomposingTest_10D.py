#!/usr/bin/env python
'''
 This python script can be used to reproduce the results on 10D distributions
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
'''
 A simple example for the work on the section
 5.4 of the paper 'Approximating generalized
 likelihood ratio test with calibrated discriminative
 classifiers' by Kyle Cranmer
'''


def plotCValues(
        test,
        c0,
        c1,
        dir='/afs/cern.ch/user/j/jpavezse/systematics',
        c1_g='',
        model_g='mlp',
        true_dist=False,
        vars_g=None,
        workspace='workspace_DecomposingTestOfMixtureModelsClassifiers.root',
        use_log=False,
        n_hist_c=100):
    if use_log:
        post = 'log'
    else:
        post = ''

    keys = ['true', 'dec']
    c1_values = dict((key, np.zeros(n_hist_c)) for key in keys)
    c2_values = dict((key, np.zeros(n_hist_c)) for key in keys)
    c1_2 = np.loadtxt('{0}/fitting_values_c1c2{1}.txt'.format(dir, post))
    c1_values['true'] = c1_2[:, 0]
    c1_values['dec'] = c1_2[:, 1]
    c2_values['true'] = c1_2[:, 2]
    c2_values['dec'] = c1_2[:, 3]

    saveFig([], [c1_values['true'], c1_values['dec']],
            makePlotName('c1c2', 'train', type='c1_hist{0}'.format(post)), type='hist',
            axis=['signal weight'], marker=True, marker_value=c1[0],
            labels=['Exact', 'Approx. Decomposed'], x_range=[0., 0.2], dir=dir,
            model_g=model_g, title='Histogram for estimated values signal weight', print_pdf=True)
    saveFig([], [c2_values['true'], c2_values['dec']],
            makePlotName('c1c2', 'train', type='c2_hist{0}'.format(post)), type='hist',
            axis=['bkg. weight'], marker=True, marker_value=c1[1],
            labels=['Exact', 'Approx. Decomposed'], x_range=[0.1, 0.4], dir=dir,
            model_g=model_g, title='Histogram for estimated values bkg. weight', print_pdf=True)


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
            n_hidden=40,
            L2_reg=0.0001),
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
    vars_g = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9']

    ROOT.gROOT.SetBatch(ROOT.kTRUE)
    ROOT.RooAbsPdf.defaultIntegratorConfig().setEpsRel(1E-15)
    ROOT.RooAbsPdf.defaultIntegratorConfig().setEpsAbs(1E-15)
    # Set this value to False if only final plots are needed
    verbose_printing = True

    if (len(sys.argv) > 3):
        print 'Setting seed: {0} '.format(sys.argv[3])
        ROOT.RooRandom.randomGenerator().SetSeed(int(sys.argv[3]))
        np.random.seed(int(sys.argv[3]))

    # make private mixture model
    makeModelPrivateND(
        vars_g=vars_g,
        c0=c0,
        c1=c1,
        workspace=workspace_file,
        dir=dir,
        model_g=model_g,
        verbose_printing=verbose_printing,
        load_cov=True)

    # make sintetic data to train the classifiers
    makeData(vars_g=vars_g, c0=c0, c1=c1, num_train=100000, num_test=50000,
             workspace=workspace_file, dir=dir, c1_g=c1_g, model_g='mlp')

    # train the pairwise classifiers
    trainClassifiers(clf, 3, dir=dir, model_g=model_g,
                     c1_g=c1_g, model_file='adaptive')

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
    test.fit(
        data_file='test',
        true_dist=True,
        vars_g=vars_g)

    test.computeRatios(true_dist=True, vars_g=vars_g, use_log=False)

    n_hist = 200
    # compute likelihood for c0[0] and c0[1] values
    test.fitCValues(
        c0,
        c1,
        dir=dir,
        c1_g=c1_g,
        model_g=model_g,
        true_dist=True,
        vars_g=vars_g,
        workspace=workspace_file,
        use_log=False,
        clf=clf if model_g == 'mlp' else None,
        n_hist_c=n_hist)

    plotCValues(test,c0,c1,dir=dir,c1_g=c1_g,model_g=model_g,true_dist=True,vars_g=vars_g,
        workspace=workspace_file,use_log=False, n_hist_c=n_hist)
