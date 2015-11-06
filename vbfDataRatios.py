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

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from mlp import make_predictions, train_mlp

from utils import printMultiFrame, printFrame, saveFig, loadData,\
              makeROC, makeSigBkg, makePlotName, getWeights

from train_classifiers import trainClassifiers, predict
from decomposed_test import DecomposedTest

from xgboost_wrapper import XGBoostClassifier

def evalC1C2Likelihood(test,c0,c1,dir='/afs/cern.ch/user/j/jpavezse/systematics',
            workspace='workspace_DecomposingTestOfMixtureModelsClassifiers.root',
            c1_g='',model_g='mlp',use_log=False,true_dist=False,vars_g=None):

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
  #saveFig([],[testdata[:,0]], 
  #   makePlotName('data_x0','fit',type='hist'),hist=True, 
  #    axis=['x'],labels=['fit data'],dir=dir,
  #    model_g=model_g,title='Histogram for fit data',print_pdf=True)

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
        f0pdf = w.pdf('bkghistpdf_{0}_{1}'.format(k,j))
        f1pdf = w.pdf('sighistpdf_{0}_{1}'.format(k,j))
        outputs = predict('{0}/model/{1}/{2}/{3}_{4}_{5}.pkl'.format(dir,model_g,c1_g,
        'adaptive',k,j),testdata,model_g=model_g)
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

      #decomposedRatios = decomposedRatios[test.findOutliers(decomposedRatios)]
      #trueRatios = trueRatios[test.findOutliers(trueRatios)]
      #saveFig([],[decomposedRatios,trueRatios], 
      #  makePlotName('ratio','train',type='{0}_{1}_hist'.format(i,j)),hist=True, 
      #  axis=['ratio'],
      #  labels=['true','composed'],dir=dir,
      #  model_g=model_g,title='Histogram for ratios',print_pdf=True)

      if use_log == False:
        decomposedLikelihood[i,j] = np.log(decomposedRatios).sum()
        trueLikelihood[i,j] = np.log(trueRatios).sum()
      else:
        decomposedLikelihood[i,j] = decomposedRatios.sum()
        trueLikelihood[i,j] = trueRatios.sum()

  #decomposedLikelihood = decomposedLikelihood - decomposedLikelihood.min()
  #X,Y = np.meshgrid(csarray, cs2array)
  #saveFig(X,[Y,decomposedLikelihood,trueLikelihood],makePlotName('comp','train',type='multilikelihood'),labels=['composed','true'],contour=True,marker=True,dir=dir,marker_value=(c1[0],c1[1]),print_pdf=True)
  decMin = np.unravel_index(decomposedLikelihood.argmin(), decomposedLikelihood.shape)
  if true_dist == True:
    trueLikelihood = trueLikelihood - trueLikelihood.min() 
    trueMin = np.unravel_index(trueLikelihood.argmin(), trueLikelihood.shape)
    return [[csarray[trueMin[0]],cs2array[trueMin[1]]], [csarray[decMin[0]],cs2array[decMin[1]]]]
  else:
    return [[0.,0.],[csarray[decMin[0]],cs2array[decMin[1]]]]
  

def plotCValues(c0,c1,dir='/afs/cern.ch/user/j/jpavezse/systematics',
            c1_g='',model_g='mlp',true_dist=False,vars_g=None,
            workspace='workspace_DecomposingTestOfMixtureModelsClassifiers.root',
            use_log=False, n_hist=150,c_eval=0, range_min=-1.0,range_max=0.):
  if use_log == True:
    post = 'log'
  else:
    post = ''

  keys = ['true','dec']
  c1_ = dict((key,np.zeros(n_hist)) for key in keys)
  c1_values = dict((key,np.zeros(n_hist)) for key in keys)
  c2_values = dict((key,np.zeros(n_hist)) for key in keys)
  c1_1 = np.loadtxt('{0}/fitting_values_c1.txt'.format(dir))  
  c1_['true'] = c1_1[:,0]
  c1_['dec'] = c1_1[:,1]
  if true_dist == True:
    vals = [c1_['true'],c1_['dec']]
    labels = ['true','dec']
  else:
    vals = c1_['dec']
    vals1 = c1_1[:,3]
    labels = ['dec']
  #vals = vals[vals <> 0.5]
  #vals = vals[vals <> 1.4]
  #vals1 = vals1[vals1 <> 1.1]
  #vals1 = vals1[vals1 <> 1.7]
  size = min(vals.shape[0],vals1.shape[0])
  #saveFig([],[vals1], 
  #    makePlotName('g2','train',type='hist_g1g2'),hist=True, 
  #    axis=['g2'],marker=True,marker_value=c1[c_eval],
  #    labels=labels,x_range=[range_min,range_max],dir=dir,
  #    model_g=model_g,title='Histogram for fitted g2', print_pdf=True)
  saveFig([],[vals,vals1], 
      makePlotName('g1g2','train',type='hist'),hist=True,hist2D=True, 
      axis=['g1','g2'],marker=True,marker_value=c1,
      labels=labels,dir=dir,model_g=model_g,title='2D Histogram for fitted g1,g2', print_pdf=True,
      x_range=[[0.5,1.4],[1.1,1.9]])
  #saveFig([],[c1_values['true'],c1_values['dec']], 
  #    makePlotName('c1c2','train',type='c1_hist{0}'.format(post)),hist=True, 
  #    axis=['c1[0]'],marker=True,marker_value=c1[0],
  #    labels=['true','composed'],x_range=[0.,0.2],dir=dir,
  #    model_g=model_g,title='Histogram for fitted values c1[0]',print_pdf=True)
  #saveFig([],[c2_values['true'],c2_values['dec']], 
  #    makePlotName('c1c2','train',type='c2_hist{0}'.format(post)),hist=True, 
  #    axis=['c1[1]'],marker=True,marker_value=c1[1],
  #    labels=['true','composed'],x_range=[0.1,0.4],dir=dir,
  #    model_g=model_g,title='Histogram for fitted values c1[1]',print_pdf=True)


def evalDist(x,f0,val):
  iter = x.createIterator()
  v = iter.Next()
  i = 0
  while v:
    v.setVal(val[i])
    v = iter.Next()
    i = i+1
  return f0.getVal(x)


def checkCrossSection(c1,cross_section,samples,target,dir,c1_g,model_g,feature=0):
  w = ROOT.RooWorkspace('w')
  normalizer = (np.abs(np.multiply(c1,cross_section))).sum()
  normalizer = (np.multiply(c1,cross_section)).sum()
  #normalizer = cross_section.sum()

  # load S(1,1.5) data
  data_file = 'data'
  testdata = np.loadtxt('{0}/data/{1}/{2}/{3}_{4}.dat'.format(dir,'mlp',c1_g,data_file,target))
  testdata = testdata[:,feature]
  bins = 300
  low = 0.
  high = 250.
  w.factory('score[{0},{1}]'.format(low,high))
  s = w.var('score')
  target_hist = ROOT.TH1F('targethist','targethist',bins,low,high)
  for val in testdata:
    target_hist.Fill(val)
  norm = 1./target_hist.Integral()
  target_hist.Scale(norm) 

  samples_hists = []
  sum_hist = ROOT.TH1F('sampleshistsum','sampleshistsum',bins,low,high)
  for i,sample in enumerate(samples):
    samples_hist = ROOT.TH1F('sampleshist{0}'.format(i),'sampleshist',bins,low,high)
    testdata = np.loadtxt('{0}/data/{1}/{2}/{3}_{4}.dat'.format(dir,'mlp',c1_g,data_file,sample))
    testdata = testdata[:,feature]
    weight = np.abs((c1[i] * cross_section[i]))/normalizer
    weight = (c1[i] * cross_section[i])/normalizer
    for val in testdata:
      samples_hist.Fill(val)
      #samples_hist.Fill(val,weight)
    norm = 1./samples_hist.Integral()
    samples_hist.Scale(norm) 
    samples_hists.append(samples_hist)
    sum_hist.Add(samples_hist,weight)  
  
  target_datahist = ROOT.RooDataHist('{0}datahist'.format('target'),'histtarget',
        ROOT.RooArgList(s),target_hist)
  target_histpdf = ROOT.RooHistFunc('{0}histpdf'.format('target'),'histtarget',
        ROOT.RooArgSet(s), target_datahist, 0)
  #xarray = np.linspace(low, high, bins) 
  #score = ROOT.RooArgSet(s)
  #test_values = np.array([evalDist(score,target_histpdf,[xs]) for xs in xarray])
  samples_datahist = ROOT.RooDataHist('{0}datahist'.format('samples'),'histsamples',
        ROOT.RooArgList(s),sum_hist)
  samples_histpdf = ROOT.RooHistFunc('{0}histpdf'.format('samples'),'histsamples',
        ROOT.RooArgSet(s), samples_datahist, 0)
  
  printFrame(w,['score'],[target_histpdf,samples_histpdf],'check_cross_section_{0}'.format(feature),['real','weighted'],
    dir=dir, model_g=model_g,title='cross section check',x_text='x',y_text='dN')

if __name__ == '__main__':
  # Setting the classifier to use
  model_g = None
  classifiers = {'svc':svm.NuSVC(probability=True),'svr':svm.NuSVR(),
        'logistic': linear_model.LogisticRegression(), 
        'bdt':GradientBoostingClassifier(n_estimators=300, learning_rate=0.1,
        max_depth=4, random_state=0),
        'mlp':'',
        'xgboost': XGBoostClassifier(num_class=2, nthread=4, silent=0,
          num_boost_round=1000, eta=0.5, max_depth=6)}
  clf = None
  if (len(sys.argv) > 1):
    model_g = sys.argv[1]
    clf = classifiers.get(sys.argv[1])
  if clf == None:
    model_g = 'logistic'
    clf = classifiers['logistic']    
    print 'Not found classifier, Using logistic instead'

  # parameters of the mixture model
  c0 = np.array([1.,1., 1., 1.,1.])
  #c0 = np.array([1.,0.,0.,0.,0.]) 
  #c1 = np.array([-0.0625, 0.5625, 0.5625, -0.0625, 0.5625])
  #c1 = np.array([0.1,0.2,0.1,0.3,0.3])
  c1 = np.array([1.,1.5])
  #c1 = np.array([1.,0.])
  cross_section = np.array([0.1149,8.469,1.635, 27.40, 0.1882])
  c0 = np.multiply(c0,cross_section)
  #cross_section = None
  #c1 = c1/c1.sum()
  c0 = c0/c0.sum()
  print c0
  print c1
  c1_g = ''
  c1_g = 'vbf'
  print c1_g
 
  verbose_printing = True
  dir = '/afs/cern.ch/user/j/jpavezse/systematics'
  workspace_file = 'workspace_vbfDataRatios.root'
  
  data_files = ['S10','S12','S11','S13','S01']
  f1_dist = 'S1_1p5'
  #f1_dist = 'F1'
  #f0_dist = 'F0' 
  f0_dist = 'S10'
  # features
  vars_g = ["mH", "Z1_m", "Z2_m", "Mjj", "DelEta_jj", "DelPhi_jj", "jet1_eta", "jet2_eta", 
          "jet1_pt", "jet2_pt", "ZeppetaZZ", "pT_Hjj", "pT_Hjj_bin_50"]

  ROOT.gROOT.SetBatch(ROOT.kTRUE)
  ROOT.RooAbsPdf.defaultIntegratorConfig().setEpsRel(1E-15)
  ROOT.RooAbsPdf.defaultIntegratorConfig().setEpsAbs(1E-15)
  # Set this value to False if only final plots are needed
  verbose_printing = True
  random_seed = 1234
  if (len(sys.argv) > 3):
    print 'Setting seed: {0} '.format(sys.argv[3])
    random_seed = int(sys.argv[3])
    ROOT.RooRandom.randomGenerator().SetSeed(random_seed) 
  # Checking correct cross section 
  #for i in range(5):
  #  checkCrossSection(c1,cross_section,data_files,f1_dist,dir,c1_g,model_g,feature=i)
  #pdb.set_trace()

  set_size = np.zeros(len(data_files))

  scaler = None
  # train the pairwise classifiers
  #scaler = trainClassifiers(clf,c0,c1,workspace=workspace_file,dir=dir, model_g=model_g,
  #    c1_g=c1_g ,model_file='model',data_file='data',dataset_names=data_files,preprocessing=False,
  #    seed=random_seed, full_names=[f0_dist,f1_dist])
  #pdb.set_trace()

  # class which implement the decomposed method
  test = DecomposedTest(c0,c1,dir=dir,c1_g=c1_g,model_g=model_g,
          input_workspace=workspace_file, verbose_printing = verbose_printing,
          dataset_names=data_files,model_file='model',preprocessing=False,scaler=scaler,
          seed=random_seed, F1_dist=f1_dist,F0_dist=f0_dist, cross_section=cross_section)
  #test.fit(data_file='data',importance_sampling=False, true_dist=False,vars_g=vars_g)
  #test.computeRatios(true_dist=True,vars_g=vars_g,use_log=True) 
  #test.computeRatios(data_file='data',true_dist=False,vars_g=vars_g,use_log=False) 

  n_hist = 1050
  # compute likelihood for c0[0] and c0[1] values
  #test.fitCValues(c0,c1,data_file='data', true_dist=False,vars_g=vars_g,use_log=False,
  #          n_hist=n_hist, num_pseudodata=5000,weights_func=getWeights)

  plotCValues(c0,c1,dir=dir,c1_g=c1_g,model_g=model_g,true_dist=False,vars_g=vars_g,
      workspace=workspace_file,use_log=False,n_hist=n_hist,c_eval=1,range_min=1.1,
      range_max=1.9)


