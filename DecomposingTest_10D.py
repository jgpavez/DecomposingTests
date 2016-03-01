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
from sklearn.externals import joblib
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import GradientBoostingClassifier

import sys

import os.path
import pdb

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

def evalC1C2Likelihood(test,c0,c1,dir='/afs/cern.ch/user/j/jpavezse/systematics',
            workspace='workspace_DecomposingTestOfMixtureModelsClassifiers.root',
            c1_g='',model_g='mlp',use_log=False,true_dist=False,vars_g=None,clf=None,
            verbose_printing=False):

  f = ROOT.TFile('{0}/{1}'.format(dir,workspace))
  w = f.Get('w')
  f.Close()
  if true_dist == True:
    vars = ROOT.TList()
    for var in vars_g:
      vars.Add(w.var(var))
    x = ROOT.RooArgSet(vars)
  else:
    x = None

  score = ROOT.RooArgSet(w.var('score'))
  if use_log == True:
    evaluateRatio = test.evaluateLogDecomposedRatio
    post = 'log'
  else:
    evaluateRatio = test.evaluateDecomposedRatio
    post = ''

  npoints = 25
  csarray = np.linspace(0.01,0.2,npoints)
  cs2array = np.linspace(0.1,0.4,npoints)
  testdata = np.loadtxt('{0}/data/{1}/{2}/{3}_{4}.dat'.format(dir,model_g,c1_g,'test','F1'))

  decomposedLikelihood = np.zeros((npoints,npoints))
  trueLikelihood = np.zeros((npoints,npoints))
  c1s = np.zeros(c1.shape[0])
  c0s = np.zeros(c1.shape[0])
  pre_pdf = []
  pre_dist = []
  pre_pdf.extend([[],[]])
  pre_dist.extend([[],[]])
  for k,c0_ in enumerate(c0):
    pre_pdf[0].append([])
    pre_pdf[1].append([])
    pre_dist[0].append([])
    pre_dist[1].append([])
    for j,c1_ in enumerate(c1):
      if k <> j:
        f0pdf = w.function('bkghistpdf_{0}_{1}'.format(k,j))
        f1pdf = w.function('sighistpdf_{0}_{1}'.format(k,j))
        outputs = predict('{0}/model/{1}/{2}/{3}_{4}_{5}.pkl'.format(dir,model_g,c1_g,
        'adaptive',k,j),testdata,model_g=model_g,clf=clf)
        f0pdfdist = np.array([test.evalDist(score,f0pdf,[xs]) for xs in outputs])
        f1pdfdist = np.array([test.evalDist(score,f1pdf,[xs]) for xs in outputs])
        pre_pdf[0][k].append(f0pdfdist)
        pre_pdf[1][k].append(f1pdfdist)
      else:
        pre_pdf[0][k].append(None)
        pre_pdf[1][k].append(None)
      if true_dist == True:          
        f0 = w.pdf('f{0}'.format(k))
        f1 = w.pdf('f{0}'.format(j))
        if len(testdata.shape) > 1:
          f0dist = np.array([test.evalDist(x,f0,xs) for xs in testdata])
          f1dist = np.array([test.evalDist(x,f1,xs) for xs in testdata])
        else:
          f0dist = np.array([test.evalDist(x,f0,[xs]) for xs in testdata])
          f1dist = np.array([test.evalDist(x,f1,[xs]) for xs in testdata])
        pre_dist[0][k].append(f0dist) 
        pre_dist[1][k].append(f1dist) 
  
  # Evaluate Likelihood in different c1[0] and c1[1] values
  for i,cs in enumerate(csarray):
    for j, cs2 in enumerate(cs2array):
      c1s[:] = c1[:]
      c1s[0] = cs
      c1s[1] = cs2
      c1s[2] = 1.-cs-cs2
      decomposedRatios,trueRatios = evaluateRatio(w,testdata,
      x=x,plotting=False,roc=False,c0arr=c0,c1arr=c1s,true_dist=true_dist,
      pre_evaluation=pre_pdf,
      pre_dist=pre_dist)

      if use_log == False:
        decomposedLikelihood[i,j] = np.log(decomposedRatios).sum()
        trueLikelihood[i,j] = np.log(trueRatios).sum()
      else:
        decomposedLikelihood[i,j] = decomposedRatios.sum()
        trueLikelihood[i,j] = trueRatios.sum()

  decomposedLikelihood = decomposedLikelihood - decomposedLikelihood.min()
  X,Y = np.meshgrid(csarray, cs2array)
  decMin = np.unravel_index(decomposedLikelihood.argmin(), decomposedLikelihood.shape)
  min_value = [csarray[decMin[0]],cs2array[decMin[1]]]
  if verbose_printing == True:
      saveFig(X,[Y,decomposedLikelihood,trueLikelihood],makePlotName('comp','train',type='multilikelihood'),labels=['composed','true'],contour=True,marker=True,dir=dir,marker_value=(c1[0],c1[1]),print_pdf=True,min_value=min_value)
  if true_dist == True:
    trueLikelihood = trueLikelihood - trueLikelihood.min() 
    trueMin = np.unravel_index(trueLikelihood.argmin(), trueLikelihood.shape)
    return [[csarray[trueMin[0]],cs2array[trueMin[1]]], [csarray[decMin[0]],cs2array[decMin[1]]]]
  else:
    return [[0.,0.],[csarray[decMin[0]],cs2array[decMin[1]]]]
  

def fitCValues(test,c0,c1,dir='/afs/cern.ch/user/j/jpavezse/systematics',
            c1_g='',model_g='mlp',true_dist=False,vars_g=None,
            workspace='workspace_DecomposingTestOfMixtureModelsClassifiers.root',
            use_log=False, clf=None):
  if use_log == True:
    post = 'log'
  else:
    post = ''
  n_hist_c = 200
  keys = ['true','dec']
  c1_ = dict((key,np.zeros(n_hist_c)) for key in keys)
  c1_values = dict((key,np.zeros(n_hist_c)) for key in keys)
  c2_values = dict((key,np.zeros(n_hist_c)) for key in keys)

  fil2 = open('{0}/fitting_values_c1c2{1}.txt'.format(dir,post),'w')

  for i in range(n_hist_c):

    makeData(vars_g, c0,c1, num_train=200000,num_test=500,no_train=True,
          workspace=workspace,dir=dir,c1_g=c1_g) 

    if i == 0:
        verbose_printing = True
    else:
        verbose_printing = False
    
    ((c1_true,c2_true),(c1_dec,c2_dec)) = evalC1C2Likelihood(test,c0,c1,dir=dir,
              c1_g=c1_g,model_g=model_g, true_dist=true_dist,vars_g=vars_g,
              workspace=workspace,use_log=use_log,clf=clf, verbose_printing=verbose_printing)   

    print '2: {0} {1} {2} {3}'.format(c1_true, c1_dec, c2_true, c2_dec)

    fil2.write('{0} {1} {2} {3}\n'.format(c1_true, c1_dec, c2_true, c2_dec))

  fil2.close()

def plotCValues(test,c0,c1,dir='/afs/cern.ch/user/j/jpavezse/systematics',
            c1_g='',model_g='mlp',true_dist=False,vars_g=None,
            workspace='workspace_DecomposingTestOfMixtureModelsClassifiers.root',
            use_log=False):
  if use_log == True:
    post = 'log'
  else:
    post = ''

  n_hist_c = 200
  keys = ['true','dec']
  c1_values = dict((key,np.zeros(n_hist_c)) for key in keys)
  c2_values = dict((key,np.zeros(n_hist_c)) for key in keys)
  c1_2 = np.loadtxt('{0}/fitting_values_c1c2{1}.txt'.format(dir,post))
  c1_values['true'] = c1_2[:,0]
  c1_values['dec'] = c1_2[:,1]
  c2_values['true'] = c1_2[:,2]
  c2_values['dec'] = c1_2[:,3]
  
  saveFig([],[c1_values['true'],c1_values['dec']], 
      makePlotName('c1c2','train',type='c1_hist{0}'.format(post)),hist=True, 
      axis=['signal weight'],marker=True,marker_value=c1[0],
      labels=['true','composed'],x_range=[0.,0.2],dir=dir,
      model_g=model_g,title='Histogram for estimated values signal weight',print_pdf=True)
  saveFig([],[c2_values['true'],c2_values['dec']], 
      makePlotName('c1c2','train',type='c2_hist{0}'.format(post)),hist=True, 
      axis=['bkg. weight'],marker=True,marker_value=c1[1],
      labels=['true','composed'],x_range=[0.1,0.4],dir=dir,
      model_g=model_g,title='Histogram for estimated values bkg. weight',print_pdf=True)


if __name__ == '__main__':
  # Setting the classifier to use
  model_g = None
  classifiers = {'svc':svm.NuSVC(probability=True),'svr':svm.NuSVR(),
        'logistic': linear_model.LogisticRegression(), 
        'bdt':GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
        max_depth=5, random_state=0),
        'mlp':MLPTrainer(n_hidden=40, L2_reg=0.0001),
        'xgboost': XGBoostClassifier(num_class=2, nthread=4, silent=1,
          num_boost_round=100, eta=0.5, max_depth=4)}
  clf = None
  if (len(sys.argv) > 1):
    model_g = sys.argv[1]
    clf = classifiers.get(sys.argv[1])
  if clf == None:
    model_g = 'logistic'
    clf = classifiers['logistic']    
    print 'Not found classifier, Using logistic instead'

  # parameters of the mixture model
  c0 = np.array([.0,.3, .7])
  c1 = np.array([.1,.3, .7])
  c1_g = ''

  c0 = c0/c0.sum()
  c1[0] = sys.argv[2]
  if c1[0] < 0.01:
    c1_g = "%.3f"%c1[0]
  else:
    c1_g = "%.2f"%c1[0]
  c1[0] = (c1[0]*(c1[1]+c1[2]))/(1.-c1[0])
  c1 = c1 / c1.sum()

  verbose_printing = True
  dir = '/afs/cern.ch/user/j/jpavezse/systematics'
  workspace_file = 'workspace_DecomposingTestOfMixtureModelsClassifiers.root'
  
  # features
  vars_g = ['x0','x1','x2','x3','x4','x5','x6','x7','x8','x9']

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
  makeModelPrivateND(vars_g=vars_g,c0=c0,c1=c1,workspace=workspace_file,dir=dir,
                     model_g=model_g,verbose_printing=verbose_printing,load_cov=True)

  # make sintetic data to train the classifiers
  makeData(vars_g=vars_g,c0=c0,c1=c1,num_train=100000,num_test=50000,
    workspace=workspace_file,dir=dir, c1_g=c1_g, model_g='mlp') 
    
  # train the pairwise classifiers
  trainClassifiers(clf,3,dir=dir, model_g=model_g,
     c1_g=c1_g ,model_file='adaptive')

  # class which implement the decomposed method
  test = DecomposedTest(c0,c1,dir=dir,c1_g=c1_g,model_g=model_g,
          input_workspace=workspace_file, verbose_printing = verbose_printing,
          dataset_names=['0','1','2'], clf=clf if model_g=='mlp' else None)
  test.fit(data_file='test',importance_sampling=False, true_dist=True,vars_g=vars_g)
 
  test.computeRatios(true_dist=True,vars_g=vars_g,use_log=False) 

  # compute likelihood for c0[0] and c0[1] values
  fitCValues(test,c0,c1,dir=dir,c1_g=c1_g,model_g=model_g,true_dist=True,vars_g=vars_g,
       workspace=workspace_file,use_log=False, clf=clf if model_g=='mlp' else None)

  plotCValues(test,c0,c1,dir=dir,c1_g=c1_g,model_g=model_g,true_dist=True,vars_g=vars_g,
        workspace=workspace_file,use_log=False)


