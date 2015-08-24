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

from make_data import makeData, makeModelND, makeModelPrivateND,\
              makeModel
from utils import printMultiFrame, printFrame, saveFig, loadData,\
              makeROC, makeSigBkg, makePlotName

from train_classifiers import trainClassifiers, predict
from decomposed_test import DecomposedTest
''' 
 A simple example for the work on the section 
 5.4 of the paper 'Approximating generalized 
 likelihood ratio test with calibrated discriminative
 classifiers' by Kyle Cranmer
''' 

def evalC1Likelihood(test,c0,c1,dir='/afs/cern.ch/user/j/jpavezse/systematics',
            workspace='workspace_DecomposingTestOfMixtureModelsClassifiers.root',
            c1_g='',model_g='mlp',use_log=False,true_dist=False, vars_g=None):

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
  csarray = np.linspace(0.01,0.10,npoints)
  testdata = np.loadtxt('{0}/data/{1}/{2}/{3}_{4}.dat'.format(dir,'mlp',c1_g,'test','F1'))
  decomposedLikelihood = np.zeros(npoints)
  trueLikelihood = np.zeros(npoints)
  c1s = np.zeros(c1.shape[0])
  pre_pdfratios = []
  pre_ratios = []
  for k,c0_ in enumerate(c0):
    pre_pdfratios.append([])
    pre_ratios.append([])
    for j,c1_ in enumerate(c1):
      if k <> j:
        f0pdf = w.pdf('bkghistpdf_{0}_{1}'.format(k,j))
        f1pdf = w.pdf('sighistpdf_{0}_{1}'.format(k,j))
        outputs = predict('{0}/model/{1}/{2}/{3}_{4}_{5}.pkl'.format(dir,model_g,c1_g,
        'adaptive',k,j),testdata,model_g=model_g)
        pdfratios = [test.singleRatio(score,f0pdf,f1pdf,[xs]) for xs in outputs]
        pdfratios = np.array(pdfratios)
        pre_pdfratios[k].append(pdfratios)
      else:
        pre_pdfratios[k].append(None)
      if true_dist == True:          
        f0 = w.pdf('f{0}'.format(k))
        f1 = w.pdf('f{0}'.format(j))
        if len(testdata.shape) > 1:
          ratios = np.array([test.singleRatio(x,f0,f1,xs) for xs in testdata])
        else:
          ratios = np.array([test.singleRatio(x,f0,f1,[xs]) for xs in testdata])
        pre_ratios[k].append(ratios) 
  for i,cs in enumerate(csarray):
    c1s[:] = c1[:]
    c1s[0] = cs
    c1s = c1s/c1s.sum()
    decomposedRatios,trueRatios = evaluateRatio(w,testdata,x=x,
    plotting=False,roc=False,c0arr=c0,c1arr=c1s,true_dist=true_dist,pre_ratios=pre_ratios,
    pre_pdfratios=pre_pdfratios)
    if use_log == False:
      decomposedLikelihood[i] = np.log(decomposedRatios).sum()
      trueLikelihood[i] = np.log(trueRatios).sum()
    else:
      decomposedLikelihood[i] = decomposedRatios.sum()
      trueLikelihood[i] = trueRatios.sum()

  decomposedLikelihood = decomposedLikelihood - decomposedLikelihood.min()
  if true_dist == True:
    trueLikelihood = trueLikelihood - trueLikelihood.min() 
    saveFig(csarray,[decomposedLikelihood,trueLikelihood],makePlotName('comp','train',type=post+'likelihood'),labels=['decomposed','true'],axis=['c1[0]','-ln(L)'],marker=True,dir=dir,
        marker_value=c1[0],title='c1[0] Fitting',print_pdf=True)
    return (csarray[trueLikelihood.argmin()], csarray[decomposedLikelihood.argmin()])
  else:
    return (0.,csarray[decomposedLikelihood.argmin()])

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
  
  '''
  outputs = predict('{0}/model/{1}/{2}/{3}_{4}_{5}.pkl'.format(dir,model_g,c1_g,
      'adaptive',0,0),testdata,model_g=model_g)
  dist0 = np.array([test.evalDist(x,w.pdf('f0'),xs) for xs in testdata])
  dist1 = np.array([test.evalDist(x,w.pdf('f1'),xs) for xs in testdata])
  dist2 = np.array([test.evalDist(x,w.pdf('f2'),xs) for xs in testdata])
  dist0_ = np.array([test.evalDist(score,w.pdf('sighistpdf_0'),[xs]) for xs in outputs])
  dist1_ = np.array([test.evalDist(score,w.pdf('sighistpdf_1'),[xs]) for xs in outputs])
  dist2_ = np.array([test.evalDist(score,w.pdf('sighistpdf_2'),[xs]) for xs in outputs])
  '''
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

      decomposedRatios = decomposedRatios[test.findOutliers(decomposedRatios)]
      #trueRatios = trueRatios[test.findOutliers(trueRatios)]

      '''
      #Plotting ratio histograms
      #Histogram F0-f0 for composed, full and true
      all_ratios_plots = []
      all_names_plots = []
      bins = 70
      low = 0.6
      high = 1.2
      if use_log == True:
        low = -1.0
        high = 1.0
      
      ratios_names = ['composed','truth']
      ratios_vec = [decomposedRatios,trueRatios]
      minimum = min([trueRatios.min(), decomposedRatios.min()])
      maximum = max([trueRatios.max(), decomposedRatios.max()]) 
      low = minimum - ((maximum - minimum) / bins)*10
      high = maximum + ((maximum - minimum) / bins)*10
      w.factory('ratio[{0},{1}]'.format(low, high))
      ratio = w.var('ratio')
      for curr, curr_ratios in zip(ratios_names,ratios_vec):
        numtest = curr_ratios.shape[0] 
        for l,name in enumerate(['sig','bkg']):
          hist = ROOT.TH1F('{0}_{1}histfit_F0_f0'.format(curr,name),'hist',bins,low,high)
          for val in curr_ratios[l*numtest/2:(l+1)*numtest/2]:
            hist.Fill(val)
          datahist = ROOT.RooDataHist('{0}_{1}datahistfit_F0_f0'.format(curr,name),'hist',
                ROOT.RooArgList(ratio),hist)
          ratio.setBins(bins)
          histpdf = ROOT.RooHistFunc('{0}_{1}histpdffit_F0_f0'.format(curr,name),'hist',
                ROOT.RooArgSet(ratio), datahist, 0)

          histpdf.specialIntegratorConfig(ROOT.kTRUE).method1D().setLabel('RooBinIntegrator')
          getattr(w,'import')(hist)
          getattr(w,'import')(datahist) # work around for morph = w.import(morph)
          getattr(w,'import')(histpdf) # work around for morph = w.import(morph)
          #print '{0} {1} {2}'.format(curr,name,hist.Integral())
          if name == 'bkg':
            all_ratios_plots.append([w.function('{0}_sighistpdffit_F0_f0'.format(curr)),
                  w.function('{0}_bkghistpdffit_F0_f0'.format(curr))])
            all_names_plots.append(['sig_{0}'.format(curr),'bkg_{0}'.format(curr)])
          
      all_ratios_plots = [[all_ratios_plots[j][i] for j,_ in enumerate(all_ratios_plots)] 
                  for i,_ in enumerate(all_ratios_plots[0])]
      all_names_plots = [[all_names_plots[j][i] for j,_ in enumerate(all_names_plots)] 
                  for i,_ in enumerate(all_names_plots[0])]

      printMultiFrame(w,'ratio',all_ratios_plots, makePlotName('ratio_fit','comparison',type='hist'+post,dir=dir,model_g=model_g,c1_g=c1_g),all_names_plots,setLog=True,dir=dir,model_g=model_g,y_text='Count',title='Histograms for ratios',x_text='ratio value',print_pdf=True)
      trueRatios = c1s[0] * dist0 + c1s[1] * dist1 + c1s[2] * dist2 
      decomposedRatios = c1s[0] * dist0_ + c1s[1] * dist1_ + c1s[2] * dist2_ 
      '''
      '''
      saveFig([],[trueRatios], 
        makePlotName('ratio','true',type='{0}_{1}_hist'.format(i,j)),hist=True, 
        axis=['ratio'],
        labels=['true','composed'],dir=dir,
        model_g=model_g,title='Histogram for ratios',print_pdf=True)
      saveFig([],[decomposedRatios], 
        makePlotName('ratio','train',type='{0}_{1}_hist'.format(i,j)),hist=True, 
        axis=['ratio'],
        labels=['true','composed'],dir=dir,
        model_g=model_g,title='Histogram for ratios',print_pdf=True)
      '''
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
  

def fitCValues(test,c0,c1,dir='/afs/cern.ch/user/j/jpavezse/systematics',
            c1_g='',model_g='mlp',true_dist=False,vars_g=None,
            workspace='workspace_DecomposingTestOfMixtureModelsClassifiers.root',
            use_log=False):
  if use_log == True:
    post = 'log'
  else:
    post = ''
  n_hist_c = 200
  keys = ['true','dec']
  c1_ = dict((key,np.zeros(n_hist_c)) for key in keys)
  c1_values = dict((key,np.zeros(n_hist_c)) for key in keys)
  c2_values = dict((key,np.zeros(n_hist_c)) for key in keys)
  #c1_ = {key:np.zeros(n_hist_c) for key in keys}
  #c1_values = {key:np.zeros(n_hist_c) for key in keys}
  #c2_values =  {key:np.zeros(n_hist_c) for key in keys}
  #fil1 = open('{0}/fitting_values_c1.txt'.format(dir),'a')
  fil2 = open('{0}/fitting_values_c1c2{1}.txt'.format(dir,post),'a')

  for i in range(n_hist_c):
    makeData(vars_g, c0,c1, num_train=200000,num_test=500,no_train=True,
          workspace=workspace,dir=dir,c1_g=c1_g,model_g=model_g) 

    #(c1_true_1, c1_dec_1) = evalC1Likelihood(test,c0,c1,dir=dir,c1_g=c1_g,model_g=model_g,
    #          true_dist=true_dist,vars_g=vars_g,workspace=workspace)  
    ((c1_true,c2_true),(c1_dec,c2_dec)) = evalC1C2Likelihood(test,c0,c1,dir=dir,
              c1_g=c1_g,model_g=model_g, true_dist=true_dist,vars_g=vars_g,
              workspace=workspace,use_log=use_log)   
    #print '1: {0} {1}'.format(c1_true_1, c1_dec_1)
    #print '2: {0} {1} {2} {3}'.format(c1_true, c1_dec, c2_true, c2_dec)
    #fil1.write('{0} {1}\n'.format(c1_true_1, c1_dec_1))
    fil2.write('{0} {1} {2} {3}\n'.format(c1_true, c1_dec, c2_true, c2_dec))
  #fil1.close()  
  fil2.close()

def plotCValues(test,c0,c1,dir='/afs/cern.ch/user/j/jpavezse/systematics',
            c1_g='',model_g='mlp',true_dist=False,vars_g=None,
            workspace='workspace_DecomposingTestOfMixtureModelsClassifiers.root',
            use_log=False):
  if use_log == True:
    post = 'log'
  else:
    post = ''

  n_hist_c = 500
  keys = ['true','dec']
  c1_ = dict((key,np.zeros(n_hist_c)) for key in keys)
  c1_values = dict((key,np.zeros(n_hist_c)) for key in keys)
  c2_values = dict((key,np.zeros(n_hist_c)) for key in keys)
  #c1_1 = np.loadtxt('{0}/fitting_values_c1.txt'.format(dir))  
  #c1_['true'] = c1_1[:,0]
  #c1_['dec'] = c1_1[:,1]
  c1_2 = np.loadtxt('{0}/fitting_values_c1c2{1}.txt'.format(dir,post))
  c1_values['true'] = c1_2[:,0]
  c1_values['dec'] = c1_2[:,1]
  c2_values['true'] = c1_2[:,2]
  c2_values['dec'] = c1_2[:,3]
  
  #saveFig([],[c1_['true'],c1_['dec']], 
  #    makePlotName('c1','train',type='hist'),hist=True, 
  #    axis=['c1[0]'],marker=True,marker_value=c1[0],
  #    labels=['true','composed'],x_range=[0.,0.1],dir=dir,
  #    model_g=model_g)
  saveFig([],[c1_values['true'],c1_values['dec']], 
      makePlotName('c1c2','train',type='c1_hist{0}'.format(post)),hist=True, 
      axis=['c1[0]'],marker=True,marker_value=c1[0],
      labels=['true','composed'],x_range=[0.,0.2],dir=dir,
      model_g=model_g,title='Histogram for fitted values c1[0]',print_pdf=True)
  saveFig([],[c2_values['true'],c2_values['dec']], 
      makePlotName('c1c2','train',type='c2_hist{0}'.format(post)),hist=True, 
      axis=['c1[1]'],marker=True,marker_value=c1[1],
      labels=['true','composed'],x_range=[0.1,0.4],dir=dir,
      model_g=model_g,title='Histogram for fitted values c1[1]',print_pdf=True)


if __name__ == '__main__':
  # Setting the classifier to use
  model_g = None
  classifiers = {'svc':svm.NuSVC(probability=True),'svr':svm.NuSVR(),
        'logistic': linear_model.LogisticRegression(), 
        'bdt':GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
        max_depth=1, random_state=0),
        'mlp':''}
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

  c1[0] = sys.argv[2]
  if c1[0] < 0.01:
    c1_g = "%.3f"%c1[0]
  else:
    c1_g = "%.2f"%c1[0]
  c1[0] = (c1[0]*(c1[1]+c1[2]))/(1.-c1[0])
  c1 = c1 / c1.sum()
  print c0
  print c1
  print c1_g
 
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

  #makeModel(c0=c0,c1=c1,workspace=workspace_file,dir=dir,verbose_printing=
  #  verbose_printing)
  #vars_g = ['x']
  # make private mixture model
  #makeModelPrivateND(vars_g=vars_g,c0=c0,c1=c1,workspace=workspace_file,dir=dir,
  #                  model_g=model_g,verbose_printing=verbose_printing,load_cov=True)

  # make mixture model
  #makeModelND(vars_g=vars_g,c0=c0,c1=c1,workspace=workspace_file,dir=dir,
  #      verbose_printing=verbose_printing)

  # make sintetic data to train the classifiers
  #makeData(vars_g=vars_g,c0=c0,c1=c1,num_train=1000,num_test=30000,
  #  workspace=workspace_file,dir=dir, c1_g=c1_g, model_g=model_g) 
    
  # train the pairwise classifiers
  #trainClassifiers(clf,c0,c1,workspace=workspace_file,dir=dir, model_g=model_g,
  #    c1_g=c1_g ,model_file='adaptive')

  # class which implement the decomposed method
  test = DecomposedTest(c0,c1,dir=dir,c1_g=c1_g,model_g=model_g,
          input_workspace=workspace_file, verbose_printing = verbose_printing)
  #test.fit(data_file='test')
  #test.computeRatios(true_dist=True,vars_g=vars_g,use_log=True) 
  #test.computeRatios(true_dist=True,vars_g=vars_g,use_log=False) 

  # compute likelihood for c0[0] and c0[1] values
  fitCValues(test,c0,c1,dir=dir,c1_g=c1_g,model_g=model_g,true_dist=True,vars_g=vars_g,
        workspace=workspace_file,use_log=True)

  #plotCValues(test,c0,c1,dir=dir,c1_g=c1_g,model_g=model_g,true_dist=True,vars_g=vars_g,
  #      workspace=workspace_file,use_log=True)


