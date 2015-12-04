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

from make_data import makeData, makeModelND
from utils import printMultiFrame, printFrame, saveFig, loadData, printFrame, makePlotName,\
          loadData,makeSigBkg,makeROC, makeMultiROC,saveMultiFig,preProcessing, savitzky_golay
from train_classifiers import predict

from pyMorphWrapper import MorphingWrapper


class DecomposedTest:
  ''' 
    Class which implement the decomposed test on 
    the data
  ''' 
  def __init__(self,c0,c1,model_file='adaptive',
            input_workspace=None,
            output_workspace='workspace_DecomposingTestOfMixtureModelsClassifiers.root',
            dir='/afs/cern.ch/user/j/jpavezse/systematics',
            c1_g='',model_g='mlp',
            verbose_printing=False,
            dataset_names=None,
            preprocessing=False,
            scaler=None,
            seed=1234,
            F1_dist='F1',
            F0_dist='F0',
            cross_section=None,
            all_couplings=None,
            F1_couplings=None,
            basis_indexes=None):
    self.input_workspace = input_workspace
    self.workspace = output_workspace
    self.c0 = c0
    self.c1 = c1
    self.model_file = model_file
    self.dir = dir
    self.c1_g = c1_g
    self.model_g = model_g
    self.verbose_printing=verbose_printing
    self.preprocessing = preprocessing
    self.scaler = scaler
    self.seed=seed
    self.F1_dist=F1_dist
    self.F0_dist=F0_dist
    self.cross_section=cross_section
    self.dataset_names=dataset_names
    self.basis_indexes = basis_indexes if basis_indexes <> None else range(len(dataset_names))
    self.F1_couplings=F1_couplings
    self.all_couplings=all_couplings
    self.nsamples = len(dataset_names)


  def fit(self, data_file='test',importance_sampling=False, true_dist=True,vars_g=None):
    ''' 
      Create pdfs for the classifier 
      score to be used later on the ratio 
      test, input workspace only needed in case 
      there exist true pdfs for the distributions
      the models being used are ./model/{model_g}/{c1_g}/{model_file}_i_j.pkl
      and the data files are ./data/{model_g}/{c1_g}/{data_file}_i_j.dat
    '''

    bins = 80
    low = 0.
    high = 1.  
    
    if self.input_workspace <> None:
      f = ROOT.TFile('{0}/{1}'.format('/tmp/jpavezse/',self.workspace))
      w = f.Get('w')
      # TODO test this when workspace is present
      w = ROOT.RooWorkspace('w') if w == None else w
      f.Close()
    else: 
      w = ROOT.RooWorkspace('w')


    print 'Generating Score Histograms'

    w.factory('score[{0},{1}]'.format(low,high))
    s = w.var('score')
    
    if importance_sampling == True:
      if true_dist == True:
        vars = ROOT.TList()
        for var in vars_g:
          vars.Add(w.var(var))
        x = ROOT.RooArgSet(vars)
      else:
        x = None

    #This is because most of the data of the full model concentrate around 0 
    bins_full = 80
    low_full = 0.
    high_full = 1.
    w.factory('scoref[{0},{1}]'.format(low_full, high_full))
    s_full = w.var('scoref')
    histos = []
    histos_names = []
    inv_histos = []
    inv_histos_names = []
    sums_histos = []
    def saveHistos(w,outputs,s,bins,low,high,pos=None,importance_sampling=False,importance_data=None,
          importance_outputs=None):
      if pos <> None:
        k,j = pos
      else:
        k,j = ('F0','F1')
      print 'Estimating {0} {1}'.format(k,j)
      for l,name in enumerate(['sig','bkg']):
        data = ROOT.RooDataSet('{0}data_{1}_{2}'.format(name,k,j),"data",
            ROOT.RooArgSet(s))
        hist = ROOT.TH1F('{0}hist_{1}_{2}'.format(name,k,j),'hist',bins,low,high)
        values = outputs[l]
        #values = values[self.findOutliers(values)]
        for val in values:
          hist.Fill(val)
          s.setVal(val)
          data.add(ROOT.RooArgSet(s))
        norm = 1./hist.Integral()
        hist.Scale(norm) 
          
        s.setBins(bins)
        datahist = ROOT.RooDataHist('{0}datahist_{1}_{2}'.format(name,k,j),'hist',
              ROOT.RooArgList(s),hist)
        #histpdf = ROOT.RooHistPdf('{0}histpdf_{1}_{2}'.format(name,k,j),'hist',
        #      ROOT.RooArgSet(s), datahist, 1)
        histpdf = ROOT.RooHistFunc('{0}histpdf_{1}_{2}'.format(name,k,j),'hist',
              ROOT.RooArgSet(s), datahist, 1)
        #histpdf.setUnitNorm(True)
        #testvalues = np.array([self.evalDist(ROOT.RooArgSet(s), histpdf, [xs]) for xs in values])

        #histpdf.specialIntegratorConfig(ROOT.kTRUE).method1D().setLabel('RooBinIntegrator')

        #print 'INTEGRAL'
        #print histpdf.createIntegral(ROOT.RooArgSet(s)).getVal()
        #print histpdf.Integral()
      
        #histpdf.specialIntegratorConfig(ROOT.kTRUE).method1D().setLabel('RooAdaptiveGaussKronrodIntegrator1D')

        getattr(w,'import')(hist)
        getattr(w,'import')(data)
        getattr(w,'import')(datahist) # work around for morph = w.import(morph)
        getattr(w,'import')(histpdf) # work around for morph = w.import(morph)
        score_str = 'scoref' if pos == None else 'score'
        # Calculate the density of the classifier output using kernel density 
        #w.factory('KeysPdf::{0}dist_{1}_{2}({3},{0}data_{1}_{2},RooKeysPdf::NoMirror,2)'.format(name,k,j,score_str))

        # Print histograms pdfs and estimated densities
        if self.verbose_printing == True and name == 'bkg' and k <> j:
          full = 'full' if pos == None else 'dec'
          if k < j and k <> 'F0':
            histos.append([w.function('sighistpdf_{0}_{1}'.format(k,j)), w.function('bkghistpdf_{0}_{1}'.format(k,j))])
            histos_names.append(['f{0}-f{1}_f{1}(signal)'.format(k,j), 'f{0}-f{1}_f{0}(background)'.format(k,j)])
          if j < k and k <> 'F0':
            inv_histos.append([w.function('sighistpdf_{0}_{1}'.format(k,j)), w.function('bkghistpdf_{0}_{1}'.format(k,j))])
            inv_histos_names.append(['f{0}-f{1}_f{1}(signal)'.format(k,j), 'f{0}-f{1}_f{0}(background)'.format(k,j)])

    if self.scaler == None:
      self.scaler = {}

    '''
    # change this
    for k in range(self.nsamples):
      for j in range(self.nsamples):
        if k == j:
          continue
        if self.dataset_names <> None:
          name_k, name_j = (self.dataset_names[k], self.dataset_names[j])
        else:
          name_k, name_j = (k,j)
        print 'Loading {0}:{1} {2}:{3}'.format(k,name_k, j,name_j)
        traindata, targetdata = loadData(data_file,name_k,name_j,dir=self.dir,c1_g=self.c1_g,
            preprocessing=self.preprocessing,scaler=self.scaler,persist=True)
       
        numtrain = traindata.shape[0]       
        size2 = traindata.shape[1] if len(traindata.shape) > 1 else 1
        output = [predict('/tmp/jpavezse/{0}_{1}_{2}.pkl'.format(self.model_file,k,j),traindata[targetdata == 1],model_g=self.model_g),
          predict('/tmp/jpavezse/{0}_{1}_{2}.pkl'.format(self.model_file,k,j),traindata[targetdata == 0],model_g=self.model_g)]

        saveHistos(w,output,s,bins,low,high,(k,j))
        w.writeToFile('{0}/{1}'.format('/tmp/jpavezse/',self.workspace))

    if self.verbose_printing==True:
      for ind in range(1,(len(histos)/3+1)):
        print_histos = histos[(ind-1)*3:(ind-1)*3+3]
        print_histos_names = histos_names[(ind-1)*3:(ind-1)*3+3]
        printMultiFrame(w,['score']*len(print_histos),print_histos, makePlotName('dec{0}'.format(ind-1),'all',type='hist',dir=self.dir,c1_g=self.c1_g,model_g=self.model_g),print_histos_names,
          dir=self.dir,model_g=self.model_g,y_text='score(x)',print_pdf=True,title='Pairwise score distributions')
    '''
    # Full model
    traindata, targetdata = loadData(data_file,self.F0_dist,self.F1_dist,dir=self.dir,c1_g=self.c1_g,
      preprocessing=self.preprocessing, scaler=self.scaler)
    numtrain = traindata.shape[0]       
    size2 = traindata.shape[1] if len(traindata.shape) > 1 else 1
    #outputs = [predict('{0}/model/{1}/{2}/{3}_F0_F1.pkl'.format(self.dir,self.model_g,self.c1_g,self.model_file),traindata[targetdata==1],model_g=self.model_g),
    #          predict('{0}/model/{1}/{2}/{3}_F0_F1.pkl'.format(self.dir,self.model_g,self.c1_g,self.model_file),traindata[targetdata==0],model_g=self.model_g)]
    outputs = [predict('/tmp/jpavezse/{0}_F0_F1.pkl'.format(self.model_file),traindata[targetdata==1],model_g=self.model_g),
              predict('/tmp/jpavezse/{0}_F0_F1.pkl'.format(self.model_file),traindata[targetdata==0],model_g=self.model_g)]

    saveHistos(w,outputs,s_full, bins_full, low_full, high_full,importance_sampling=False)
    if self.verbose_printing == True:
      printFrame(w,['scoref'],[w.function('sighistpdf_F0_F1'),w.function('bkghistpdf_F0_F1')], makePlotName('full','all',type='hist',dir=self.dir,c1_g=self.c1_g,model_g=self.model_g),['signal','bkg'],
    dir=self.dir,model_g=self.model_g,y_text='score(x)',print_pdf=True,title='Pairwise score distributions')
   
    w.writeToFile('{0}/{1}'.format('/tmp/jpavezse/',self.workspace))
      
    w.Print()

  # To calculate the ratio between single functions
  def singleRatio(self,f0,f1):
    ratio = f1 / f0
    ratio[np.abs(ratio) == np.inf] = 0 
    ratio[np.isnan(ratio)] = 0
    return ratio

  def evalDist(self,x,f0,val):
    iter = x.createIterator()
    v = iter.Next()
    i = 0
    while v:
      v.setVal(val[i])
      v = iter.Next()
      i = i+1
    return f0.getVal(x)

  # To calculate the ratio between single functions
  def __regFunc(self,x,f0,f1,val):
    iter = x.createIterator()
    v = iter.Next()
    i = 0
    while v:
      v.setVal(val[i])
      v = iter.Next()
      i = i+1
    if (f0.getVal(x) + f1.getVal(x)) == 0.:
      return 0.
    return f1.getVal(x) / (f0.getVal(x) + f1.getVal(x))

  # Log Ratios functions
  def computeLogKi(self, f0, f1, c0, c1):
    k_ij = np.log(c1*f1) - np.log(c0*f0)
    k_ij[np.abs(k_ij) == np.inf] = 0
    return k_ij
  # ki is a vector
  def computeAi(self,k0, ki):
    ai = -k0 - np.log(1. + np.sum(np.exp(ki - k0),0))
    return ai
  def singleLogRatio(self, f0, f1):
    rat = np.log(f1) - np.log(f0)
    rat[np.abs(rat) == np.inf] = 0
    return rat

  def evaluateLogDecomposedRatio(self, w, evalData, x=None, plotting=True,roc=False, gridsize=None,
            c0arr=None,c1arr=None,true_dist=False,pre_evaluation=None,pre_dist=None,debug=False):
    score = ROOT.RooArgSet(w.var('score'))
    npoints = evalData.shape[0]
    pdfratios = np.zeros(npoints)
    ratios = np.zeros(npoints)
    c0arr = self.c0 if c0arr == None else c0arr
    c1arr = self.c1 if c1arr == None else c1arr
    
    # Log formula array initialization
    ksTrained = np.zeros((c0arr.shape[0]-1, c1arr.shape[0],npoints))
    ks = np.zeros((c0arr.shape[0]-1, c1arr.shape[0],npoints))
    k0Trained = np.zeros(npoints)
    k0 = np.zeros(npoints)
    ai = np.zeros((c0arr.shape[0]-1,npoints))
    aiTrained = np.zeros((c0arr.shape[0]-1,npoints))

    true_score = []
    train_score = []
    all_targets = []

    # I can do this with list comprehension
    for k, c in enumerate(c0arr):
      if c == 0:
        continue
      for j, c_ in enumerate(c1arr):
        if true_dist == True:
          f0 = w.pdf('f{0}'.format(k))
          f1 = w.pdf('f{0}'.format(j))
          if pre_dist == None:
            if len(evalData.shape) > 1:
              f0dist = np.array([self.evalDist(x,f0,xs) for xs in evalData])
              f1dist = np.array([self.evalDist(x,f1,xs) for xs in evalData])
            else:
              f0dist = np.array([self.evalDist(x,f0,[xs]) for xs in evalData])
              f1dist = np.array([self.evalDist(x,f1,[xs]) for xs in evalData])
          else:
            f0dist = pre_dist[0][k][j]
            f1dist = pre_dist[1][k][j]
          
          ks[k-1][j] = self.computeLogKi(f0dist,f1dist,c,c_)

        f0pdf = w.function('bkghistpdf_{0}_{1}'.format(k,j))
        f1pdf = w.function('sighistpdf_{0}_{1}'.format(k,j))
        if k <> j:
          if pre_evaluation == None:
            outputs = predict('{0}/model/{1}/{2}/{3}_{4}_{5}.pkl'.format(self.dir,self.model_g,self.c1_g,self.model_file,k,j),evalData,model_g=self.model_g)
            f0pdfdist = np.array([self.evalDist(score,f0pdf,[xs]) for xs in outputs])
            f1pdfdist = np.array([self.evalDist(score,f1pdf,[xs]) for xs in outputs])
          else:
            f0pdfdist = pre_evaluation[0][k][j]
            f1pdfdist = pre_evaluation[1][k][j]
        else:
          # Must check this
          f0pdfdist = np.ones(npoints)
          f1pdfdist = np.ones(npoints)

        ksTrained[k-1][j] = self.computeLogKi(f0pdfdist,f1pdfdist,c,c_)
        # ROC curves for pair-wise ratios
        if (roc == True or plotting==True) and j < k:
          if roc == True:
            testdata, testtarget = loadData('test',k,j,dir=self.dir,c1_g=self.c1_g) 
          else:
            testdata = evalData
          size2 = testdata.shape[1] if len(testdata.shape) > 1 else 1
          outputs = predict('{0}/model/{1}/{2}/{3}_{4}_{5}.pkl'.format(self.dir,self.model_g,
                    self.c1_g,self.model_file,k,j),
                    testdata.reshape(testdata.shape[0],size2),model_g=self.model_g)
          f0pdfdist = np.array([self.evalDist(score,f0pdf,[xs]) for xs in outputs])
          f1pdfdist = np.array([self.evalDist(score,f1pdf,[xs]) for xs in outputs])
          clfRatios = self.singleRatio(f0pdfdist,f1pdfdist)
          train_score.append(clfRatios)
          if roc == True:
            all_targets.append(testtarget)
          #individual ROC
          #makeROC(clfRatios, testtarget,makePlotName('dec','train',k,j,type='roc',dir=self.dir,
          #model_g=self.model_g,c1_g=self.c1_g),dir=self.dir,model_g=self.model_g)
          if true_dist == True:
            if len(evalData.shape) > 1:
              f0dist = np.array([self.evalDist(x,f0,xs) for xs in testdata])
              f1dist = np.array([self.evalDist(x,f1,xs) for xs in testdata])
            else:
              f0dist = np.array([self.evalDist(x,f0,[xs]) for xs in testdata])
              f1dist = np.array([self.evalDist(x,f1,[xs]) for xs in testdata])

            trRatios = self.singleRatio(f0dist,f1dist)

            true_score.append(trRatios)
          

        if plotting == True and k <> j:
          saveFig(evalDist, [ks[k-1][j],ksTrained[k-1][j]], makePlotName('decomposed','trained',k,j,type='ratio_log',dir=self.dir,model_g=self.model_g,c1_g=self.c1_g),labels=['trained','truth'],model_g=self.model_g,dir=self.dir)

      #check this
      if true_dist == True:
        kSorted = np.sort(ks[k-1],0)
        ai[k-1] = self.computeAi(kSorted[0],kSorted[1:])
      kSortedTrained = np.sort(ksTrained[k-1],0)
      aiTrained[k-1] = self.computeAi(kSortedTrained[0],kSortedTrained[1:])
    if true_dist == True:
      aSorted = np.sort(ai,0) 
      ratios = aSorted[0] + np.log(1. + np.sum(np.exp(aSorted[1:] - aSorted[0]),0))
    aSortedTrained = np.sort(aiTrained,0)
    pdfratios = aSortedTrained[0] + np.log(1. + np.sum(np.exp(aSortedTrained[1:] - aSortedTrained[0]),0))
    if roc == True:
      if true_dist == True:
        makeMultiROC(train_score, all_targets,makePlotName('all','comparison',type='roc',
          dir=self.dir,model_g=self.model_g,c1_g=self.c1_g),dir=self.dir,model_g=self.model_g,
          true_score = true_score,print_pdf=True,title='ROC for pairwise trained classifier',pos=[(0,1),(0,2),(1,2)])
      else:
        makeMultiROC(train_score, all_targets,makePlotName('all','comparison',type='roc',
          dir=self.dir,model_g=self.model_g,c1_g=self.c1_g),dir=self.dir,model_g=self.model_g,
          print_pdf=True,title='ROC for pairwise trained classifier',pos=[(0,1),(0,2),(1,2)])
    if plotting == True:
      saveMultiFig(evalData,[x for x in zip(train_score,true_score)],
      makePlotName('all_dec','train',type='ratio'),labels=[['f0-f1(trained)','f0-f1(truth)'],['f0-f2(trained)','f0-f2(truth)'],['f1-f2(trained)','f1-f2(truth)']],title='Pairwise Ratios',print_pdf=True,dir=self.dir)
    if debug == True:
      #printRatios = pdfratios[pdfratios < 0.]
      pdb.set_trace()
      saveFig([],[pdfratios], makePlotName('full_ratio', 'hist', type='hist'), hist=True,
      axis=['x'],labels=['full ratio'], dir=self.dir, model_g=self.model_g, 
      title='Full Ratio',print_pdf=True)
      pdb.set_trace()
    return pdfratios,ratios 


  def evaluateDecomposedRatio(self,w,evalData,x=None,plotting=True, roc=False,gridsize=None,c0arr=None, c1arr=None,true_dist=False,pre_evaluation=None,pre_dist=None,data_type='test',debug=False,cross_section=None,indexes=None):
    # pair-wise ratios
    # and decomposition computation
    #f = ROOT.TFile('{0}/{1}'.format(self.dir,self.workspace))
    #w = f.Get('w')
    #f.Close()

    if indexes == None:
      indexes = self.basis_indexes

    score = ROOT.RooArgSet(w.var('score'))
    npoints = evalData.shape[0]
    fullRatios = np.zeros(npoints)
    fullRatiosReal = np.zeros(npoints)
    c0arr = self.c0 if c0arr == None else c0arr
    c1arr = self.c1 if c1arr == None else c1arr

    true_score = []
    train_score = []
    all_targets = []
    all_positions = []
    all_ratios = []
    for k,c in enumerate(c0arr):
      innerRatios = np.zeros(npoints)
      innerTrueRatios = np.zeros(npoints)
      if c == 0:
        continue
      for j,c_ in enumerate(c1arr):
        index_k, index_j = (indexes[k],indexes[j])
        f0pdf = w.function('bkghistpdf_{0}_{1}'.format(index_k,index_j))
        f1pdf = w.function('sighistpdf_{0}_{1}'.format(index_k,index_j))
        if k<>j:
          if pre_evaluation == None:
            traindata = evalData
            if self.preprocessing == True:
              traindata = preProcessing(evalData,self.dataset_names[min(index_k,index_j)],
              self.dataset_names[max(index_k,index_j)],self.scaler) 
            #outputs = predict('{0}/model/{1}/{2}/{3}_{4}_{5}.pkl'.format(self.dir,self.model_g,self.c1_g,self.model_file,k,j),traindata,model_g=self.model_g)
            outputs = predict('/tmp/jpavezse/{0}_{1}_{2}.pkl'.format(self.model_file,index_k,
            index_j),traindata,model_g=self.model_g)
            f0pdfdist = np.array([self.evalDist(score,f0pdf,[xs]) for xs in outputs])
            f1pdfdist = np.array([self.evalDist(score,f1pdf,[xs]) for xs in outputs])
          else:
            f0pdfdist = pre_evaluation[0][index_k][index_j]
            f1pdfdist = pre_evaluation[1][index_k][index_j]
          pdfratios = self.singleRatio(f0pdfdist,f1pdfdist)
        else:
          pdfratios = np.ones(npoints) 
        all_ratios.append(pdfratios)
        innerRatios += (c_/c) * pdfratios
        if true_dist == True:
          if pre_dist == None:
            f0 = w.pdf('f{0}'.format(index_k))
            f1 = w.pdf('f{0}'.format(index_j))
            if len(evalData.shape) > 1:
              f0dist = np.array([self.evalDist(x,f0,xs) for xs in evalData])
              f1dist = np.array([self.evalDist(x,f1,xs) for xs in evalData])
            else:
              f0dist = np.array([self.evalDist(x,f0,[xs]) for xs in evalData])
              f1dist = np.array([self.evalDist(x,f1,[xs]) for xs in evalData])
          else:
            f0dist = pre_dist[0][index_k][index_j]
            f1dist = pre_dist[1][index_k][index_j]
          ratios = self.singleRatio(f0dist, f1dist)
          innerTrueRatios += (c_/c) * ratios
        # ROC curves for pair-wise ratios
        if (roc == True or plotting==True) and k < j:
          all_positions.append((k,j))
          if roc == True:
            if self.dataset_names <> None:
              name_k, name_j = (self.dataset_names[index_k], self.dataset_names[index_j])
            else:
              name_k, name_j = (index_k,index_j)
            testdata, testtarget = loadData(data_type,name_k,name_j,dir=self.dir,c1_g=self.c1_g,
                  preprocessing=self.preprocessing, scaler=self.scaler) 
          else:
            testdata = evalData
          size2 = testdata.shape[1] if len(testdata.shape) > 1 else 1
          #outputs = predict('{0}/model/{1}/{2}/{3}_{4}_{5}.pkl'.format(self.dir,self.model_g,
          #          self.c1_g,self.model_file,k,j),
          #          testdata.reshape(testdata.shape[0],size2),model_g=self.model_g)
          outputs = predict('/tmp/jpavezse/{0}_{1}_{2}.pkl'.format(self.model_file,index_k,
                    index_j),testdata.reshape(testdata.shape[0],size2),model_g=self.model_g)
          f0pdfdist = np.array([self.evalDist(score,f0pdf,[xs]) for xs in outputs])
          f1pdfdist = np.array([self.evalDist(score,f1pdf,[xs]) for xs in outputs])
          clfRatios = self.singleRatio(f0pdfdist,f1pdfdist)
          train_score.append(clfRatios)
          if roc == True:
            all_targets.append(testtarget)
          #individual ROC
          #makeROC(clfRatios, testtarget,makePlotName('dec','train',k,j,type='roc',dir=self.dir,
          #model_g=self.model_g,c1_g=self.c1_g),dir=self.dir,model_g=self.model_g)
          if true_dist == True:
            if len(evalData.shape) > 1:
              f0dist = np.array([self.evalDist(x,f0,xs) for xs in testdata])
              f1dist = np.array([self.evalDist(x,f1,xs) for xs in testdata])
            else:
              f0dist = np.array([self.evalDist(x,f0,[xs]) for xs in testdata])
              f1dist = np.array([self.evalDist(x,f1,[xs]) for xs in testdata])

            trRatios = self.singleRatio(f0dist,f1dist)

            true_score.append(trRatios)
 
          #  makeROC(trRatios, testtarget, makePlotName('dec','truth',k,j,type='roc',
          #  dir=self.dir,model_g=self.model_g,c1_g=self.c1_g),dir=self.dir,model_g=self.model_g)
          
          '''
          if plotting == True:
            # Scatter plot to compare regression function and classifier score
            if len(testdata.shape) > 1:
              reg = np.array([self.__regFunc(x,f0,f1,xs) for xs in testdata])
            else:
              reg = np.array([self.__regFunc(x,f0,f1,[xs]) for xs in testdata])
            #reg = reg/np.max(reg)
            saveFig(outputs,[reg], makePlotName('dec','train',k,j,type='scat',dir=self.dir,
            model_g=self.model_g,c1_g=self.c1_g),scatter=True, axis=['score','regression'],
            dir=self.dir,model_g=self.model_g)
          '''

      innerRatios = 1./innerRatios
      innerRatios[np.abs(innerRatios) == np.inf] = 0.
      fullRatios += innerRatios
      if true_dist == True:
        innerTrueRatios = 1./innerTrueRatios
        innerTrueRatios[np.abs(innerTrueRatios) == np.inf] = 0.
        fullRatiosReal += innerTrueRatios
    if roc == True:
      for ind in range(1,(len(train_score)/3+1)):
        print_scores = train_score[(ind-1)*3:(ind-1)*3+3]
        print_targets = all_targets[(ind-1)*3:(ind-1)*3+3]
        print_positions = all_positions[(ind-1)*3:(ind-1)*3+3]
        if true_dist == True:
          makeMultiROC(print_scores, print_targets,makePlotName('all{0}'.format(ind-1),'comparison',type='roc',
          dir=self.dir,model_g=self.model_g,c1_g=self.c1_g),dir=self.dir,model_g=self.model_g,
          true_score = true_score,print_pdf=True,title='ROC for pairwise trained classifier',pos=print_positions)
        else:
          makeMultiROC(print_scores, print_targets,makePlotName('all{0}'.format(ind-1),'comparison',type='roc',
          dir=self.dir,model_g=self.model_g,c1_g=self.c1_g),dir=self.dir,model_g=self.model_g,
          print_pdf=True,title='ROC for pairwise trained classifier',pos=print_positions)

    if plotting == True:
      saveMultiFig(evalData,[x for x in zip(train_score,true_score)],
      makePlotName('all_dec','train',type='ratio'),labels=[['f0-f1(trained)','f0-f1(truth)'],['f0-f2(trained)','f0-f2(truth)'],['f1-f2(trained)','f1-f2(truth)']],title='Pairwise Ratios',print_pdf=True,dir=self.dir)
    if debug == True:
      indices = np.where(fullRatios < 0.)
      for l,print_ratio in enumerate(all_ratios):
        saveFig([],[print_ratio[indices]], makePlotName('neg_ratio{0}'.format(l), 'hist', type='hist'), hist=True,
        axis=['x'],labels=['full ratio'], dir=self.dir, model_g=self.model_g, 
        title='Full Ratio',print_pdf=True)

      saveFig([],[fullRatios[indices]], makePlotName('full_ratio', 'hist', type='hist'), hist=True,
      axis=['x'],labels=['full ratio'], dir=self.dir, model_g=self.model_g, 
      title='Full Ratio',print_pdf=True)
      pdb.set_trace()
    return fullRatios,fullRatiosReal

  def findOutliers(self, x):
    q5, q95 = np.percentile(x, [5,95])  
    iqr = 2.0*(q95 - q5)
    outliers = (x <= q95 + iqr) & (x >= q5 - iqr)
    return outliers

  def computeRatios(self,true_dist=False, vars_g=None,
      data_file='test',use_log=False):
    '''
      Use the computed score densities to compute 
      the decompose ratio test.
      set true_dist to True if workspace have the true distributions to 
      make plots, in that case vars_g also must be provided
      Final result is histogram for ratios and signal - bkf rejection curves
    '''

    f = ROOT.TFile('{0}/{1}'.format(self.dir,self.workspace))
    w = f.Get('w')
    f.Close()
    
    #TODO: This are Harcoded for now
    c1 = self.c1
    c0 = self.c0
    c1 = np.multiply(c1, self.cross_section)
    c1 = c1/c1.sum()
    c0 = c0/c0.sum()

    print 'Calculating ratios'

    npoints = 50

    if true_dist == True:
      vars = ROOT.TList()
      for var in vars_g:
        vars.Add(w.var(var))
      x = ROOT.RooArgSet(vars)

    if use_log == True:
      evaluateRatio = self.evaluateLogDecomposedRatio
      post = 'log'
    else:
      evaluateRatio = self.evaluateDecomposedRatio
      post = ''

    score = ROOT.RooArgSet(w.var('score'))
    scoref = ROOT.RooArgSet(w.var('scoref'))

    if use_log == True:
      getRatio = self.singleLogRatio
    else:
      getRatio = self.singleRatio
   
    if self.preprocessing == True:
      if self.scaler == None:
        self.scaler = {}
        for k in range(self.nsamples):
         for j in range(self.nsamples):
           if k < j:
            self.scaler[(k,j)] = joblib.load('{0}/model/{1}/{2}/{3}_{4}_{5}.dat'.format(self.dir,'mlp',self.c1_g,'scaler',self.dataset_names[k],self.dataset_names[j]))
            

    # NN trained on complete model
    F0pdf = w.function('bkghistpdf_F0_F1')
    F1pdf = w.function('sighistpdf_F0_F1')

    # TODO Here assuming that signal is first dataset  
    testdata, testtarget = loadData(data_file,self.F0_dist,self.F1_dist,dir=self.dir,c1_g=self.c1_g,preprocessing=False) 
    if len(vars_g) == 1:
      xarray = np.linspace(0,5,npoints)
      fullRatios,_ = evaluateRatio(w,xarray,x=x,plotting=True,roc=False,true_dist=True)

      F1dist = np.array([self.evalDist(x,w.pdf('F1'),[xs]) for xs in xarray])
      F0dist = np.array([self.evalDist(x,w.pdf('F0'),[xs]) for xs in xarray])
      y2 = getRatio(F1dist, F0dist)

      # NN trained on complete model
      outputs = predict('{0}/model/{1}/{2}/adaptive_F0_F1.pkl'.format(self.dir,self.model_g,self.c1_g),xarray.reshape(xarray.shape[0],1),model_g=self.model_g)
      F1fulldist = np.array([self.evalDist(scoref,F1pdf,[xs]) for xs in outputs])
      F0fulldist = np.array([self.evalDist(scoref,F0pdf,[xs]) for xs in outputs])

      pdfratios = getRatio(F1fulldist, F0fulldist)

      saveFig(xarray, [fullRatios, y2, pdfratios], makePlotName('all','train',type='ratio'+post),title='Likelihood Ratios',labels=['Composed trained', 'Truth', 'Full Trained'],print_pdf=True,dir=self.dir)
      
    if true_dist == True:
      decomposedRatio,_ = evaluateRatio(w,testdata,x=x,plotting=False,roc=self.verbose_printing,true_dist=True)
    else:
      decomposedRatio,_ = evaluateRatio(w,testdata,c0arr=c0,c1arr=c1,plotting=True,
      roc=True,data_type=data_file)
    if len(testdata.shape) > 1:
      #outputs = predict('{0}/model/{1}/{2}/{3}_F0_F1.pkl'.format(self.dir,self.model_g,self.c1_g,self.model_file),testdata,model_g=self.model_g)
      outputs = predict('/tmp/jpavezse/{0}_F0_F1.pkl'.format(self.model_file),testdata,model_g=self.model_g)

    else:
      outputs = predict('{0}/model/{1}/{2}/{3}_F0_F1.pkl'.format(self.dir,self.model_g,self.c1_g,self.model_file),testdata.reshape(testdata.shape[0],1),model_g=self.model_g)

    F1fulldist = np.array([self.evalDist(scoref,F1pdf,[xs]) for xs in outputs])
    F0fulldist = np.array([self.evalDist(scoref,F0pdf,[xs]) for xs in outputs])

    completeRatio = getRatio(F1fulldist,F0fulldist)
    if true_dist == True:
      if len(testdata.shape) > 1:
        F1dist = np.array([self.evalDist(x,w.pdf('F1'),xs) for xs in testdata])
        F0dist = np.array([self.evalDist(x,w.pdf('F0'),xs) for xs in testdata])
      else:
        F1dist = np.array([self.evalDist(x,w.pdf('F1'),[xs]) for xs in testdata])
        F0dist = np.array([self.evalDist(x,w.pdf('F0'),[xs]) for xs in testdata])

      realRatio = getRatio(F1dist,F0dist)

    decomposed_target = testtarget
    complete_target = testtarget
    real_target = testtarget
    #Histogram F0-f0 for composed, full and true

    # Removing outliers
    numtest = decomposedRatio.shape[0] 
    #decomposedRatio[decomposedRatio < 0.] = completeRatio[decomposedRatio < 0.]

    decomposed_outliers = np.zeros(numtest,dtype=bool)
    complete_outliers = np.zeros(numtest,dtype=bool)
    decomposed_outliers = self.findOutliers(decomposedRatio)
    complete_outliers = self.findOutliers(completeRatio)
    decomposed_target = testtarget[decomposed_outliers] 
    complete_target = testtarget[complete_outliers] 
    decomposedRatio = decomposedRatio[decomposed_outliers]
    completeRatio = completeRatio[complete_outliers]
    if true_dist == True:
      real_outliers = np.zeros(numtest,dtype=bool)
      real_outliers = self.findOutliers(realRatio)
      real_target = testtarget[real_outliers] 
      realRatio = realRatio[real_outliers]

    all_ratios_plots = []
    all_names_plots = []
    bins = 70
    low = 0.6
    high = 1.2
    if use_log == True:
      low = -1.0
      high = 1.0
    low = []
    high = []
    low = []
    high = []
    ratios_vars = []
    for l,name in enumerate(['sig','bkg']):
      if true_dist == True:
        ratios_names = ['truth','full','composed']
        ratios_vec = [realRatio, completeRatio, decomposedRatio]
        minimum = min([realRatio[real_target == 1-l].min(), 
              completeRatio[complete_target == 1-l].min(), 
              decomposedRatio[decomposed_target == 1-l].min()])
        maximum = max([realRatio[real_target == 1-l].max(), 
              completeRatio[complete_target == 1-l].max(), 
              decomposedRatio[decomposed_target == 1-l].max()])

      else:
        ratios_names = ['full','composed']
        ratios_vec = [completeRatio, decomposedRatio]
        target_vec = [complete_target, decomposed_target] 
        minimum = min([completeRatio[complete_target == 1-l].min(), 
              decomposedRatio[decomposed_target == 1-l].min()])
        maximum = max([completeRatio[complete_target == 1-l].max(), 
              decomposedRatio[decomposed_target == 1-l].max()])

      low.append(minimum - ((maximum - minimum) / bins)*10)
      high.append(maximum + ((maximum - minimum) / bins)*10)
      w.factory('ratio{0}[{1},{2}]'.format(name, low[l], high[l]))
      ratios_vars.append(w.var('ratio{0}'.format(name)))
    for curr, curr_ratios, curr_targets in zip(ratios_names,ratios_vec,target_vec):
      numtest = curr_ratios.shape[0] 
      for l,name in enumerate(['sig','bkg']):
        hist = ROOT.TH1F('{0}_{1}hist_F0_f0'.format(curr,name),'hist',bins,low[l],high[l])
        for val in curr_ratios[curr_targets == 1-l]:
          hist.Fill(val)
        datahist = ROOT.RooDataHist('{0}_{1}datahist_F0_f0'.format(curr,name),'hist',
              ROOT.RooArgList(ratios_vars[l]),hist)
        ratios_vars[l].setBins(bins)
        histpdf = ROOT.RooHistFunc('{0}_{1}histpdf_F0_f0'.format(curr,name),'hist',
              ROOT.RooArgSet(ratios_vars[l]), datahist, 0)

        histpdf.specialIntegratorConfig(ROOT.kTRUE).method1D().setLabel('RooBinIntegrator')
        getattr(w,'import')(hist)
        getattr(w,'import')(datahist) # work around for morph = w.import(morph)
        getattr(w,'import')(histpdf) # work around for morph = w.import(morph)
        #print '{0} {1} {2}'.format(curr,name,hist.Integral())
        if name == 'bkg':
          all_ratios_plots.append([w.function('{0}_sighistpdf_F0_f0'.format(curr)),
                w.function('{0}_bkghistpdf_F0_f0'.format(curr))])
          all_names_plots.append(['sig_{0}'.format(curr),'bkg_{0}'.format(curr)])
        
    all_ratios_plots = [[all_ratios_plots[j][i] for j,_ in enumerate(all_ratios_plots)] 
                for i,_ in enumerate(all_ratios_plots[0])]
    all_names_plots = [[all_names_plots[j][i] for j,_ in enumerate(all_names_plots)] 
                for i,_ in enumerate(all_names_plots[0])]

    printMultiFrame(w,['ratiosig','ratiobkg'],all_ratios_plots, makePlotName('ratio','comparison',type='hist'+post,dir=self.dir,model_g=self.model_g,c1_g=self.c1_g),all_names_plots,setLog=True,dir=self.dir,model_g=self.model_g,y_text='Count',title='Histograms for ratios',x_text='ratio value',print_pdf=True)

    # scatter plot true ratio - composed - full ratio

    if self.verbose_printing == True and true_dist == True:
      saveFig(completeRatio,[realRatio], makePlotName('full','train',type='scat'+post,dir=self.dir,model_g=self.model_g,c1_g=self.c1_g),scatter=True,axis=['full trained ratio','true ratio'],dir=self.dir,model_g=self.model_g)
      saveFig(decomposedRatio,[realRatio], makePlotName('comp','train',type='scat'+post,dir=self.dir, model_g=self.model_g, c1_g=self.c1_g),scatter=True, axis=['composed trained ratio','true ratio'],dir=self.dir, model_g=self.model_g)
    # signal - bkg rejection plots
    if use_log == True:
      decomposedRatio = np.exp(decomposedRatio)
      completeRatio = np.exp(completeRatio)
      if true_dist == True:
        realRatio = np.exp(realRatio)
    if true_dist == True:
      ratios_list = [decomposedRatio/decomposedRatio.max(), 
                    completeRatio/completeRatio.max(),
                    realRatio/realRatio.max()]
      targets_list = [decomposed_target, complete_target, real_target]
      legends_list = ['composed', 'full', 'true']
    else:

      #decomposedRatio = decomposedRatio + np.abs(decomposedRatio.min())
      indices = (decomposedRatio > 0.)
      decomposedRatio = decomposedRatio[indices] 
      decomposed_target = decomposed_target[indices]
      indices = (completeRatio > 0.)
      #decomposedRatio = decomposedRatio[indices] 
      #decomposed_target = decomposed_target[indices]
      completeRatio = completeRatio[indices]
      complete_target = complete_target[indices]
      #decomposedRatio[decomposedRatio < 0.] = completeRatio[decomposedRatio < 0.] 
      #decomposedRatio = 1./decomposedRatio

      completeRatio = np.log(completeRatio)
      decomposedRatio = np.log(decomposedRatio)
      decomposedRatio = decomposedRatio + np.abs(decomposedRatio.min())
      completeRatio = completeRatio + np.abs(completeRatio.min())
      ratios_list = [decomposedRatio/decomposedRatio.max(), 
                    completeRatio/completeRatio.max()]
      targets_list = [decomposed_target, complete_target]
      legends_list = ['composed','full']
    makeSigBkg(ratios_list,targets_list,makePlotName('comp','all',type='sigbkg'+post,dir=self.dir,model_g=self.model_g,c1_g=self.c1_g),dir=self.dir,model_g=self.model_g,print_pdf=True,legends=legends_list,title='Signal-Background rejection curves')

    # Scatter plot to compare regression function and classifier score
    if self.verbose_printing == True and true_dist == True:
      testdata, testtarget = loadData('test',self.F0_dist,self.F1_dist,dir=self.dir,c1_g=self.c1_g) 
      if len(testdata.shape) > 1:
        reg = np.array([self.__regFunc(x,w.pdf('F0'),w.pdf('F1'),xs) for xs in testdata])
      else:
        reg = np.array([self.__regFunc(x,w.pdf('F0'),w.pdf('F1'),[xs]) for xs in testdata])
      if len(testdata.shape) > 1:
        outputs = predict('{0}/model/{1}/{2}/adaptive_F0_F1.pkl'.format(self.dir,self.model_g,self.c1_g),testdata.reshape(testdata.shape[0],testdata.shape[1]),model_g=self.model_g)
      else:
        outputs = predict('{0}/model/{1}/{2}/adaptive_F0_F1.pkl'.format(self.dir,self.model_g,self.c1_g),testdata.reshape(testdata.shape[0],1),model_g=self.model_g)
      #saveFig(outputs,[reg], makePlotName('full','train',type='scat',dir=self.dir,model_g=self.model_g,c1_g=self.c1_g),scatter=True,axis=['score','regression'],dir=self.dir,model_g=self.model_g)

  # Code for calibration of full Ratios
  def calibrateFullRatios(self, w, ratios,c0,c1,debug=False,samples_data=None,hist_data=60000,
        index=None):
    bins = 80
    low = -0.5
    high = 1.5
    print 'Calibrating Full Ratios'
    samples = self.dataset_names
    rng = np.random.RandomState(self.seed)
    s = w.var('score')
    #print 'Fitting samples'
    calibrated_F0 = np.zeros(ratios.shape[0])
    calibrated_F1 = np.zeros(ratios.shape[0])
    for i,sample in enumerate(samples):
      histpdf = w.function('histpdf_ratio_{0}_{1}'.format(i,index))
      if samples_data == None:
        testdata = np.loadtxt('{0}/data/{1}/{2}/{3}_{4}.dat'.format(self.dir,'mlp',self.c1_g,'data',sample))
      else:
        testdata = samples_data[i]
      if histpdf == None:
        indices = rng.choice(testdata.shape[0],hist_data)
        testdata = testdata[indices]
        composed_ratios,_ = self.evaluateDecomposedRatio(w, testdata,x=None,plotting=False,roc=False,c0arr=c0,c1arr=c1,true_dist=False,pre_dist=None,pre_evaluation=None,debug=False)
        composed_ratios = composed_ratios[self.findOutliers(composed_ratios)]
        minimum= composed_ratios.min()
        maximum = composed_ratios.max()
        low = minimum - ((maximum - minimum) / bins)*10
        high = maximum + ((maximum - minimum) / bins)*10

        w.factory('ratio_{0}_{1}[{2},{3}]'.format(i,index,low,high))
        ratio = w.var('ratio_{0}_{1}'.format(i,index))
        hist = ROOT.TH1F('hist_ratio_{0}'.format(i),'hist',bins,low,high)
        for val in composed_ratios:
          hist.Fill(val)
        norm = 1./hist.Integral()
        hist.Scale(norm) 
        #ratio.setBins(bins)
        datahist = ROOT.RooDataHist('data_ratio_{0}'.format(i),'data',
          ROOT.RooArgList(ratio),hist)
        histpdf = ROOT.RooHistFunc('histpdf_ratio_{0}_{1}'.format(i,index),'hist',
                ROOT.RooArgSet(ratio), datahist, 1)
        getattr(w,'import')(histpdf) # work around for morph = w.import(morph)

        if debug == True:
          printFrame(w,['ratio_{0}_{1}'.format(i,index)],[histpdf],makePlotName(
            'sample{0}'.format(i),'score',type='hist'), ['score'],dir=self.dir,model_g=self.model_g,print_pdf=True,title='Ratio')
      else:
        ratio = w.var('ratio_{0}_{1}'.format(i,index))
      calibrated_F0 += c0[i]*np.array([self.evalDist(
      ROOT.RooArgSet(ratio), histpdf, [xs]) for xs in ratios])
      calibrated_F1 += c1[i]*np.array([self.evalDist(
      ROOT.RooArgSet(ratio), histpdf, [xs]) for xs in ratios])
    calibrated_ratio = self.singleRatio(calibrated_F0,calibrated_F1)        
    return calibrated_ratio
 
  def pre2DMorphedBasis(self,c_min,c_max,npoints):
    csarray = np.linspace(c_min[0],c_max[0],npoints)
    csarray2 = np.linspace(c_min[1], c_max[1], npoints)
    n_eff_ratio = np.zeros((npoints,npoints))
    cross_section = None
    morph = MorphingWrapper()
    # TODO: Most of the values harcoded here
    #bkg_morph = MorphingWrapper()
    # TODO: Most of the values harcoded here
    morph.setSampleData(nsamples=15,ncouplings=3,types=['S','S','S'],morphed=self.F1_couplings,samples=self.all_couplings)
    target = self.F1_couplings[:]
    for i,cs in enumerate(csarray):
      for j,cs2 in enumerate(csarray2):
        target[1] = cs
        target[2] = cs2
        morph.resetTarget(target)
        indexes = np.array(morph.dynamicMorphing(self.F1_couplings))
        couplings = np.array(morph.getWeights())
        cross_section = np.array(morph.getCrossSections())
        sorted_indexes = np.argsort(indexes)
        indexes,couplings,cross_section = (indexes[sorted_indexes],couplings[sorted_indexes],
                          cross_section[sorted_indexes])
        #bkg_morph.resetBasis([self.all_couplings[i] for i in indexes]) 
        all_indexes = np.vstack([all_indexes,indexes]) if i <> 0 or j <> 0 else indexes 
        all_couplings = np.vstack([all_couplings,couplings])  if i <> 0 or j <> 0 else couplings
        all_cross_sections = np.vstack([all_cross_sections, cross_section]) if i <> 0 or j <> 0 else cross_section
        c1s = np.multiply(couplings,cross_section)
          #c1s = np.abs(c1s)
        n_eff = c1s.sum()
        n_tot = np.abs(c1s).sum()
        n_eff_ratio[i] = n_eff/n_tot
        print '{0} {1}'.format(i,j)
        print target
        print indexes
        print 'n_eff: {0}, n_tot: {1}, n_eff/n_tot: {2}'.format(n_eff, n_tot, n_eff/n_tot)
    pdb.set_trace()
    np.savetxt('indexes_{0:.2f}_{1:.2f}_{2:.2f}_{3:.2f}_{4}.dat'.format(c_min[0],c_min[1],c_max[0],c_max[1],npoints),all_indexes) 
    np.savetxt('couplings_{0:.2f}_{1:.2f}_{2:.2f}_{3:.2f}_{4}.dat'.format(c_min[0],c_min[1],c_max[0],c_max[1],npoints),all_couplings) 
    np.savetxt('crosssection_{0:.2f}_{1:.2f}_{2:.2f}_{3:.2f}_{4}.dat'.format(c_min[0],c_min[1],c_max[0],c_max[1],npoints),all_cross_sections) 
   

  def preMorphedBasis(self,c_eval=0,c_min=0.01,c_max=0.2, npoints=50):
    csarray = np.linspace(c_min,c_max,npoints)
    n_eff_ratio = np.zeros(csarray.shape[0])
    n_zeros = np.zeros(csarray.shape[0])
    cross_section = None
    morph = MorphingWrapper()
    # TODO: Most of the values harcoded here
    #bkg_morph = MorphingWrapper()
    # TODO: Most of the values harcoded here
    morph.setSampleData(nsamples=15,ncouplings=3,types=['S','S','S'],morphed=self.F1_couplings,samples=self.all_couplings)
    target = self.F1_couplings[:]
    for i,cs in enumerate(csarray):
      target[c_eval] = cs
      morph.resetTarget(target)
      indexes = np.array(morph.dynamicMorphing(self.F1_couplings))
      couplings = np.array(morph.getWeights())
      cross_section = np.array(morph.getCrossSections())
      sorted_indexes = np.argsort(indexes)
      indexes,couplings,cross_section = (indexes[sorted_indexes],couplings[sorted_indexes],
                        cross_section[sorted_indexes])
      #bkg_morph.resetBasis([self.all_couplings[i] for i in indexes]) 
      all_indexes = np.vstack([all_indexes,indexes]) if i <> 0 else indexes 
      all_couplings = np.vstack([all_couplings,couplings])  if i <> 0 else couplings
      all_cross_sections = np.vstack([all_cross_sections, cross_section]) if i <> 0 else cross_section
      c1s = np.multiply(couplings,cross_section)
        #c1s = np.abs(c1s)
      n_eff = c1s.sum()
      n_tot = np.abs(c1s).sum()
      n_eff_ratio[i] = n_eff/n_tot
      print target
      print 'n_eff: {0}, n_tot: {1}, n_eff/n_tot: {2}'.format(n_eff, n_tot, n_eff/n_tot)
    np.savetxt('indexes_{0:.2f}_{1:.2f}_{2}.dat'.format(c_min,c_max,npoints),all_indexes) 
    np.savetxt('couplings_{0:.2f}_{1:.2f}_{2}.dat'.format(c_min,c_max,npoints),all_couplings) 
    np.savetxt('crosssection_{0:.2f}_{1:.2f}_{2}.dat'.format(c_min,c_max,npoints),all_cross_sections) 
   

  def evalMorphC1C2Likelihood(self,w,testdata,c0,c1,c_eval=0,c_min=0.01,c_max=0.2,use_log=False,true_dist=False, vars_g=None, npoints=50,samples_ids=None,weights_func=None):

    if true_dist == True:
      vars = ROOT.TList()
      for var in vars_g:
        vars.Add(w.var(var))
      x = ROOT.RooArgSet(vars)
    else:
      x = None

    score = ROOT.RooArgSet(w.var('score'))
    if use_log == True:
      evaluateRatio = self.evaluateLogDecomposedRatio
      post = 'log'
    else:
      evaluateRatio = self.evaluateDecomposedRatio
      post = ''

    #if not os.path.isfile('indexes_{0:.2f}_{1:.2f}_{2:.2f}_{3:.2f}_{4}.dat'.format(c_min[0],c_min[1],c_max[0],c_max[1],npoints)): 
    if True:
      self.pre2DMorphedBasis(c_min=c_min,c_max=c_max,npoints=npoints)   

    csarray = np.linspace(c_min[0],c_max[0],npoints)
    csarray2 = np.linspace(c_min[1], c_max[1], npoints)
    decomposedLikelihood = np.zeros((npoints,npoints))
    trueLikelihood = np.zeros((npoints,npoints))

    all_indexes = np.loadtxt('indexes_{0:.2f}_{1:.2f}_{2:.2f}_{3:.2f}_{4}.dat'.format(c_min[0],c_min[1],c_max[0],c_max[1],npoints)) 
    all_indexes = np.array([[int(x) for x in rows] for rows in all_indexes])
    all_couplings = np.loadtxt('couplings_{0:.2f}_{1:.2f}_{2:.2f}_{3:.2f}_{4}.dat'.format(c_min[0],c_min[1],c_max[0],c_max[1],npoints)) 
    all_cross_sections = np.loadtxt('crosssection_{0:.2f}_{1:.2f}_{2:.2f}_{3:.2f}_{4}.dat'.format(c_min[0],c_min[1],c_max[0],c_max[1],npoints))
   
    #Print loading dynamic basis, ensure that they exist before running
    pre_pdf = [[range(self.nsamples) for _ in range(self.nsamples)],[range(self.nsamples) for _ in range(self.nsamples)]]
    pre_dist = [[range(self.nsamples) for _ in range(self.nsamples)],[range(self.nsamples) for _ in range(self.nsamples)]]
    # Only precompute distributions that will be used
    unique_indexes = set()
    for indexes in all_indexes:
      unique_indexes |= set(indexes) 
    # change this enumerates
    unique_indexes = list(unique_indexes)
    for k in range(len(unique_indexes)):
      for j in range(len(unique_indexes)):
        index_k,index_j = (unique_indexes[k],unique_indexes[j])
        #print 'Pre computing {0} {1}'.format(index_k,index_j)
        if k <> j:
          f0pdf = w.function('bkghistpdf_{0}_{1}'.format(index_k,index_j))
          f1pdf = w.function('sighistpdf_{0}_{1}'.format(index_k,index_j))
          data = testdata
          if self.preprocessing == True:
            data = preProcessing(testdata,self.dataset_names[min(k,j)],
            self.dataset_names[max(k,j)],self.scaler) 
          #outputs = predict('{0}/model/{1}/{2}/{3}_{4}_{5}.pkl'.format(self.dir,self.model_g,
          outputs = predict('/tmp/jpavezse/{0}_{1}_{2}.pkl'.format(self.model_file,index_k,index_j),data,model_g=self.model_g)
          f0pdfdist = np.array([self.evalDist(score,f0pdf,[xs]) for xs in outputs])
          f1pdfdist = np.array([self.evalDist(score,f1pdf,[xs]) for xs in outputs])
          pre_pdf[0][index_k][index_j] = f0pdfdist
          pre_pdf[1][index_k][index_j] = f1pdfdist
        else:
          pre_pdf[0][index_k][index_j] = None
          pre_pdf[1][index_k][index_j] = None
        if true_dist == True:
          f0 = w.pdf('f{0}'.format(index_k))
          f1 = w.pdf('f{0}'.format(index_j))
          if len(testdata.shape) > 1:
            f0dist = np.array([self.evalDist(x,f0,xs) for xs in testdata])
            f1dist = np.array([self.evalDist(x,f1,xs) for xs in testdata])
          else:
            f0dist = np.array([self.evalDist(x,f0,[xs]) for xs in testdata])
            f1dist = np.array([self.evalDist(x,f1,[xs]) for xs in testdata])
          pre_dist[0][index_k][index_j] = f0dist
          pre_dist[1][index_k][index_j] = f1dist

    indices = np.ones(testdata.shape[0], dtype=bool)
    ratiosList = []
    samples = []
    # This is needed for calibration of full ratios
    #for i,sample in enumerate(self.dataset_names):
    #  samples.append(np.loadtxt('{0}/data/{1}/{2}/{3}_{4}.dat'.format(self.dir,'mlp',self.c1_g,'data',sample)))
    n_eff_ratio = np.zeros((csarray.shape[0], csarray2.shape[0]))
    n_zeros = np.zeros((npoints,npoints))
    target = self.F1_couplings[:]
    for i,cs in enumerate(csarray):
      ratiosList.append([])
      for j, cs2 in enumerate(csarray2):
        target[1] = cs
        target[2] = cs2
        print '{0} {1}'.format(i,j)
        print target
        print all_indexes[i*npoints + j]
        c1s = all_couplings[i*npoints + j]
        cross_section = all_cross_sections[i*npoints + j] 
        c1s = np.multiply(c1s,cross_section)
        c0_arr = c0
        n_eff = c1s.sum()
        n_tot = np.abs(c1s).sum()
        n_eff_ratio[i,j] = n_eff/n_tot 
        #print '{0} {1}'.format(i,j)
        #print 'n_eff: {0}, n_tot: {1}, n_eff/n_tot: {2}'.format(n_eff, n_tot, n_eff/n_tot)
        c1s = c1s/c1s.sum()
        #print c1s
        decomposedRatios,trueRatios = evaluateRatio(w,testdata,x=x,
        plotting=False,roc=False,c0arr=c0_arr,c1arr=c1s,true_dist=true_dist,
        pre_dist=pre_dist,pre_evaluation=pre_pdf,cross_section=cross_section,
        indexes=all_indexes[i*npoints + j])
        decomposedRatios = 1./decomposedRatios
        print '{0} {1}'.format(i,j)
        print decomposedRatios[decomposedRatios < 0.].shape
        n_zeros[i,j] = decomposedRatios[decomposedRatios < 0.].shape[0]
        ratiosList[i].append(decomposedRatios)

        #print('{0} {1} '.format(i,j)),
        #print decomposedRatios[decomposedRatios < 0.].shape 
        #print c1s
        #indices = np.logical_and(indices, decomposedRatios > 0.)
        decomposedRatios = ratiosList[i][j]
        if use_log == False:
          if samples_ids <> None:
            ratios = decomposedRatios
            ids = samples_ids
            decomposedLikelihood[i,j] = (np.dot(np.log(ratios),
                np.array([c1[x] for x in ids]))).sum()
          else:
            #decomposedRatios[decomposedRatios < 0.] = 0.9
            decomposedRatios[decomposedRatios < 0.] = 1.0
            #decomposedRatios = decomposedRatios[self.findOutliers(decomposedRatios)]
            if n_eff_ratio[i,j] <= 0.5:
              #TODO: Harcoded number
              decomposedLikelihood[i,j] = 20000
            else:
              decomposedLikelihood[i,j] = -np.log(decomposedRatios).sum()
            #print decomposedLikelihood[i,j]
            #print '{0} {1} {2}'.format(i,j,decomposedLikelihood[i,j])
          trueLikelihood[i,j] = -np.log(trueRatios).sum()
        else:
          decomposedLikelihood[i,j] = decomposedRatios.sum()
          trueLikelihood[i,j] = trueRatios.sum()
      #print '\n {0}'.format(i)
    pdb.set_trace()
    decomposedLikelihood = decomposedLikelihood - decomposedLikelihood.min()
    decMin = np.unravel_index(decomposedLikelihood.argmin(), decomposedLikelihood.shape)
    # pixel plots
    #saveFig(csarray,[csarray2,decomposedLikelihood],makePlotName('comp','train',type='likelihood_g1g2'),labels=['composed'],pixel=True,marker=True,dir=self.dir,model_g=self.model_g,marker_value=(c1[0],c1[1]),print_pdf=True,contour=True,title='Likelihood fit for g1,g2')

    #decMin = [np.sum(decomposedLikelihood,1).argmin(),np.sum(decomposedLikelihood,0).argmin()] 
    X,Y = np.meshgrid(csarray, csarray2)

    saveFig(X,[Y,decomposedLikelihood],makePlotName('comp','train',type='multilikelihood'),labels=['composed'],contour=True,marker=True,dir=self.dir,model_g=self.model_g,marker_value=(self.F1_couplings[0],self.F1_couplings[1]),print_pdf=True,min_value=(csarray[decMin[0]],csarray2[decMin[1]]))
    #print decMin
    print [csarray[decMin[0]],csarray2[decMin[1]]]
    pdb.set_trace()
    if true_dist == True:
      trueLikelihood = trueLikelihood - trueLikelihood.min()
      trueMin = np.unravel_index(trueLikelihood.argmin(), trueLikelihood.shape)
      saveFig(csarray,[decomposedLikelihood,trueLikelihood],makePlotName('comp','train',type=post+'likelihood_{0}'.format(n_sample)),labels=['decomposed','true'],axis=['c1[0]','-ln(L)'],marker=True,dir=self.dir,marker_value=c1[0],title='c1[0] Fitting',print_pdf=True)
      return [[csarray[trueMin[0]],csarray2[trueMin[1]]],
          [csarray2[decMin[0],csarray2[decMin[1]]]]]
    else:
      return [[0.,0.],[csarray[decMin[0]],csarray2[decMin[1]]]]



  def evalMorphC1Likelihood(self,w,testdata,c0,c1,c_eval=0,c_min=0.01,c_max=0.2,use_log=False,true_dist=False, vars_g=None, npoints=50,samples_ids=None,weights_func=None,coef_index=0):

    if true_dist == True:
      vars = ROOT.TList()
      for var in vars_g:
        vars.Add(w.var(var))
      x = ROOT.RooArgSet(vars)
    else:
      x = None

    score = ROOT.RooArgSet(w.var('score'))
    if use_log == True:
      evaluateRatio = self.evaluateLogDecomposedRatio
      post = 'log'
    else:
      evaluateRatio = self.evaluateDecomposedRatio
      post = ''

    csarray = np.linspace(c_min,c_max,npoints)
    decomposedLikelihood = np.zeros(npoints)
    trueLikelihood = np.zeros(npoints)
    c1s = np.zeros(c0.shape[0])
    pre_pdf = []
    pre_dist = []
    pre_pdf.extend([[],[]])
    pre_dist.extend([[],[]])

    if not os.path.isfile('indexes_{0:.2f}_{1:.2f}_{2}.dat'.format(c_min,c_max,npoints)): 
      self.preMorphedBasis(c_eval=c_eval,c_min=c_min,c_max=c_max,npoints=npoints)   

    all_indexes = np.loadtxt('indexes_{0:.2f}_{1:.2f}_{2}.dat'.format(c_min,c_max,npoints)) 
    all_indexes = np.array([[int(x) for x in rows] for rows in all_indexes])
    all_couplings = np.loadtxt('couplings_{0:.2f}_{1:.2f}_{2}.dat'.format(c_min,c_max,npoints)) 
    all_cross_sections = np.loadtxt('crosssection_{0:.2f}_{1:.2f}_{2}.dat'.format(c_min,c_max,npoints)) 
    print all_indexes

    #Print loading dynamic basis, ensure that they exist before running
    pre_pdf = [[range(self.nsamples) for _ in range(self.nsamples)],[range(self.nsamples) for _ in range(self.nsamples)]]
    pre_dist = [[range(self.nsamples) for _ in range(self.nsamples)],[range(self.nsamples) for _ in range(self.nsamples)]]
    # Only precompute distributions that will be used
    unique_indexes = set()
    for indexes in all_indexes:
      unique_indexes |= set(indexes) 
    # change this enumerates
    unique_indexes = list(unique_indexes)
    for k in range(len(unique_indexes)):
      for j in range(len(unique_indexes)):
        index_k,index_j = (unique_indexes[k],unique_indexes[j])
        print 'Pre computing {0} {1}'.format(index_k,index_j)
        if k <> j:
          f0pdf = w.function('bkghistpdf_{0}_{1}'.format(index_k,index_j))
          f1pdf = w.function('sighistpdf_{0}_{1}'.format(index_k,index_j))
          data = testdata
          if self.preprocessing == True:
            data = preProcessing(testdata,self.dataset_names[min(k,j)],
            self.dataset_names[max(k,j)],self.scaler) 
          #outputs = predict('{0}/model/{1}/{2}/{3}_{4}_{5}.pkl'.format(self.dir,self.model_g,
          outputs = predict('/tmp/jpavezse/{0}_{1}_{2}.pkl'.format(self.model_file,index_k,index_j),data,model_g=self.model_g)
          f0pdfdist = np.array([self.evalDist(score,f0pdf,[xs]) for xs in outputs])
          f1pdfdist = np.array([self.evalDist(score,f1pdf,[xs]) for xs in outputs])
          pre_pdf[0][index_k][index_j] = f0pdfdist
          pre_pdf[1][index_k][index_j] = f1pdfdist
        else:
          pre_pdf[0][index_k][index_j] = None
          pre_pdf[1][index_k][index_j] = None
        if true_dist == True:
          f0 = w.pdf('f{0}'.format(index_k))
          f1 = w.pdf('f{0}'.format(index_j))
          if len(testdata.shape) > 1:
            f0dist = np.array([self.evalDist(x,f0,xs) for xs in testdata])
            f1dist = np.array([self.evalDist(x,f1,xs) for xs in testdata])
          else:
            f0dist = np.array([self.evalDist(x,f0,[xs]) for xs in testdata])
            f1dist = np.array([self.evalDist(x,f1,[xs]) for xs in testdata])
          pre_dist[0][index_k][index_j] = f0dist
          pre_dist[1][index_k][index_j] = f1dist
    indices = np.ones(testdata.shape[0], dtype=bool)
    ratiosList = []
    samples = []
    # This is needed for calibration of full ratios
    #for i,sample in enumerate(self.dataset_names):
    #  samples.append(np.loadtxt('{0}/data/{1}/{2}/{3}_{4}.dat'.format(self.dir,'mlp',self.c1_g,'data',sample)))

    #cross_section = self.cross_section / np.sum(self.cross_section)
    n_eff_ratio = np.zeros(csarray.shape[0])
    n_zeros = np.zeros(csarray.shape[0])
    cross_section = None
    for i,cs in enumerate(csarray):
      c1s = all_couplings[i]
      cross_section = all_cross_sections[i] 
      print c1s
      c1s = np.multiply(c1s,cross_section)
      c1s = c1s/c1s.sum()
      c0_arr = c0
      #c0_arr = np.multiply(c0,cross_section)
      #c0_arr = c0_arr/c0_arr.sum()
      #pdb.set_trace()
      decomposedRatios,trueRatios = evaluateRatio(w,testdata,x=x,
      plotting=False,roc=False,c0arr=c0_arr,c1arr=c1s,true_dist=true_dist,pre_dist=pre_dist,
      pre_evaluation=pre_pdf,cross_section=cross_section,indexes=all_indexes[i])
      decomposedRatios = 1./decomposedRatios
      print decomposedRatios[decomposedRatios < 0.].shape 
      #calibratedRatios = self.calibrateFullRatios(w, decomposedRatios,
      #    c0,c1s,debug=debug,samples_data=samples,index=i) 
      #saveFig(decomposedRatios2, [calibratedRatios], makePlotName('calibrated_{0}'.format(i),'ratio',type='scat',
      #dir=self.dir, model_g=self.model_g, c1_g=self.c1_g),scatter=True, axis=['composed ratio', 
      #'composed calibrated'], dir=self.dir, model_g=self.model_g)
      ratiosList.append(decomposedRatios)
      #indices = np.logical_and(indices, decomposedRatios > 0.)
      if use_log == False:
        if samples_ids <> None:
          ratios = decomposedRatios
          ids = samples_ids
          decomposedLikelihood[i] = (np.dot(np.log(ratios),
              np.array([c1[x] for x in ids]))).sum()
        else:
          decomposedRatios[decomposedRatios < 0.] = 1.0
          decomposedLikelihood[i] = -np.log(decomposedRatios).sum()
          print decomposedLikelihood[i]
          
        trueLikelihood[i] = -np.log(trueRatios).sum()
      else:
        decomposedLikelihood[i] = decomposedRatios.sum()
        trueLikelihood[i] = trueRatios.sum()
    decomposedLikelihood = decomposedLikelihood - decomposedLikelihood.min()
    # print n_eff/n_zero relation
    #saveFig(csarray,[n_eff_ratio, n_zeros/n_zeros.max()],makePlotName('eff_ratio','zeros',type=post+'plot_g2'),labels=['n_eff/n_tot','zeros/{0}'.format(n_zeros.max())],axis=['g2','values'],marker=True,dir=self.dir,marker_value=c1[0],title='#zeros and n_eff/n_tot given g2',print_pdf=True,model_g=self.model_g)
    #saveFig(n_eff_ratio, [n_zeros/n_zeros.max()], makePlotName('eff_ratio','zeros',type='scat',
    #dir=self.dir, model_g=self.model_g, c1_g=self.c1_g),scatter=True, axis=['n_eff/n_tot', 
    #'#zeros/{0}'.format(n_zeros.max())], dir=self.dir, model_g=self.model_g,title='# zeros given n_eff/n_tot ratio')

    if true_dist == True:
      trueLikelihood = trueLikelihood - trueLikelihood.min()
      saveFig(csarray,[decomposedLikelihood,trueLikelihood],makePlotName('comp','train',type=post+'likelihood_{0}'.format(n_sample)),labels=['decomposed','true'],axis=['c1[0]','-ln(L)'],marker=True,dir=self.dir,marker_value=c1[0],title='c1[0] Fitting',print_pdf=True)
      return (csarray[trueLikelihood.argmin()], csarray[decomposedLikelihood.argmin()])
    else:
      saveFig(csarray[1:],[decomposedLikelihood[1:]],makePlotName('comp','train',type='likelihood_g1'),labels=['decomposed'],axis=['Khzz','-ln(L)'],marker=True,dir=self.dir,marker_value=self.F1_couplings[c_eval],title='Khzz Fitting',print_pdf=True,model_g=self.model_g)
      pdb.set_trace()
      return (0.,csarray[decomposedLikelihood.argmin()])


  def evalC1Likelihood(self,w,testdata,c0,c1,c_eval=0,c_min=0.01,c_max=0.2,use_log=False,true_dist=False, vars_g=None, npoints=50,samples_ids=None,weights_func=None,coef_index=0):

    if true_dist == True:
      vars = ROOT.TList()
      for var in vars_g:
        vars.Add(w.var(var))
      x = ROOT.RooArgSet(vars)
    else:
      x = None

    score = ROOT.RooArgSet(w.var('score'))
    if use_log == True:
      evaluateRatio = self.evaluateLogDecomposedRatio
      post = 'log'
    else:
      evaluateRatio = self.evaluateDecomposedRatio
      post = ''

    csarray = np.linspace(c_min,c_max,npoints)
    decomposedLikelihood = np.zeros(npoints)
    trueLikelihood = np.zeros(npoints)
    c1s = np.zeros(c0.shape[0])
    pre_pdf = []
    pre_dist = []
    pre_pdf.extend([[],[]])
    pre_dist.extend([[],[]])
    # change this enumerates
    for k in enumerate(self.nsamples):
      pre_pdf[0].append([])
      pre_pdf[1].append([])
      pre_dist[0].append([])
      pre_dist[1].append([])
      for j in enumerate(self.nsamples):
        index_k,index_j = (self.basis_indexes[k],self.basis_indexes[j])
        if k <> j:
          f0pdf = w.function('bkghistpdf_{0}_{1}'.format(index_k,index_j))
          f1pdf = w.function('sighistpdf_{0}_{1}'.format(index_k,index_j))
          data = testdata
          if self.preprocessing == True:
            data = preProcessing(testdata,self.dataset_names[min(index_k,index_j)],
            self.dataset_names[max(index_k,index_j)],self.scaler) 
          outputs = predict('{0}/model/{1}/{2}/{3}_{4}_{5}.pkl'.format(self.dir,self.model_g,
          self.c1_g,self.model_file,index_k,index_j),data,model_g=self.model_g)
          f0pdfdist = np.array([self.evalDist(score,f0pdf,[xs]) for xs in outputs])
          f1pdfdist = np.array([self.evalDist(score,f1pdf,[xs]) for xs in outputs])
          pre_pdf[0][k].append(f0pdfdist)
          pre_pdf[1][k].append(f1pdfdist)
        else:
          pre_pdf[0][k].append(None)
          pre_pdf[1][k].append(None)
        if true_dist == True:
          f0 = w.pdf('f{0}'.format(index_k))
          f1 = w.pdf('f{0}'.format(index_j))
          if len(testdata.shape) > 1:
            f0dist = np.array([self.evalDist(x,f0,xs) for xs in testdata])
            f1dist = np.array([self.evalDist(x,f1,xs) for xs in testdata])
          else:
            f0dist = np.array([self.evalDist(x,f0,[xs]) for xs in testdata])
            f1dist = np.array([self.evalDist(x,f1,[xs]) for xs in testdata])
          pre_dist[0][k].append(f0dist)
          pre_dist[1][k].append(f1dist)
    indices = np.ones(testdata.shape[0], dtype=bool)
    ratiosList = []
    samples = []
    # This is needed for calibration of full ratios
    #for i,sample in enumerate(self.dataset_names):
    #  samples.append(np.loadtxt('{0}/data/{1}/{2}/{3}_{4}.dat'.format(self.dir,'mlp',self.c1_g,'data',sample)))

    #cross_section = self.cross_section / np.sum(self.cross_section)
    n_eff_ratio = np.zeros(csarray.shape[0])
    n_zeros = np.zeros(csarray.shape[0])
    cross_section = None
    for i,cs in enumerate(csarray):
      if weights_func <> None: 
        c1s = weights_func(cs,c1[1]) if coef_index == 0 else weights_func(c1[0],cs)
        print '{0} {1}'.format(cs, c1[1]) if coef_index == 0 else '{0} {1}'.format(c1[0],cs)
        print c1s
      else:
        c1s[:] = c1[:]
        c1s[c_eval] = cs
      if self.cross_section <> None:
        c1s = np.multiply(c1s,self.cross_section)
        #c1s = np.abs(c1s)
      n_eff = c1s.sum()
      n_tot = np.abs(c1s).sum()
      print 'n_eff: {0}, n_tot: {1}, n_eff/n_tot: {2}'.format(n_eff, n_tot, n_eff/n_tot)
      c1s = c1s/c1s.sum()
      decomposedRatios,trueRatios = evaluateRatio(w,testdata,x=x,
      plotting=False,roc=False,c0arr=c0,c1arr=c1s,true_dist=true_dist,pre_dist=pre_dist,
      pre_evaluation=pre_pdf,cross_section=cross_section)
      decomposedRatios = 1./decomposedRatios
      n_eff_ratio[i] = n_eff/n_tot
      n_zeros[i] = decomposedRatios[decomposedRatios < 0.].shape[0]
      print decomposedRatios[decomposedRatios < 0.].shape 
      #calibratedRatios = self.calibrateFullRatios(w, decomposedRatios,
      #    c0,c1s,debug=debug,samples_data=samples,index=i) 
      #saveFig(decomposedRatios2, [calibratedRatios], makePlotName('calibrated_{0}'.format(i),'ratio',type='scat',
      #dir=self.dir, model_g=self.model_g, c1_g=self.c1_g),scatter=True, axis=['composed ratio', 
      #'composed calibrated'], dir=self.dir, model_g=self.model_g)
      ratiosList.append(decomposedRatios)
      #indices = np.logical_and(indices, decomposedRatios > 0.)
    for i,cs in enumerate(csarray):
      decomposedRatios = ratiosList[i]
      if use_log == False:
        if samples_ids <> None:
          ratios = decomposedRatios
          ids = samples_ids
          decomposedLikelihood[i] = (np.dot(np.log(ratios),
              np.array([c1[x] for x in ids]))).sum()
        else:
          decomposedRatios[decomposedRatios < 0.] = 1.0
          decomposedLikelihood[i] = -np.log(decomposedRatios).sum()
          print decomposedLikelihood[i]
          
        trueLikelihood[i] = -np.log(trueRatios).sum()
      else:
        decomposedLikelihood[i] = decomposedRatios.sum()
        trueLikelihood[i] = trueRatios.sum()
    decomposedLikelihood = decomposedLikelihood - decomposedLikelihood.min()
    # print n_eff/n_zero relation
    #saveFig(csarray,[n_eff_ratio, n_zeros/n_zeros.max()],makePlotName('eff_ratio','zeros',type=post+'plot_g2'),labels=['n_eff/n_tot','zeros/{0}'.format(n_zeros.max())],axis=['g2','values'],marker=True,dir=self.dir,marker_value=c1[0],title='#zeros and n_eff/n_tot given g2',print_pdf=True,model_g=self.model_g)
    #saveFig(n_eff_ratio, [n_zeros/n_zeros.max()], makePlotName('eff_ratio','zeros',type='scat',
    #dir=self.dir, model_g=self.model_g, c1_g=self.c1_g),scatter=True, axis=['n_eff/n_tot', 
    #'#zeros/{0}'.format(n_zeros.max())], dir=self.dir, model_g=self.model_g,title='# zeros given n_eff/n_tot ratio')

    if true_dist == True:
      trueLikelihood = trueLikelihood - trueLikelihood.min()
      saveFig(csarray,[decomposedLikelihood,trueLikelihood],makePlotName('comp','train',type=post+'likelihood_{0}'.format(n_sample)),labels=['decomposed','true'],axis=['c1[0]','-ln(L)'],marker=True,dir=self.dir,marker_value=c1[0],title='c1[0] Fitting',print_pdf=True)
      return (csarray[trueLikelihood.argmin()], csarray[decomposedLikelihood.argmin()])
    else:
      saveFig(csarray,[decomposedLikelihood],makePlotName('comp','train',type='likelihood_g2'),labels=['decomposed'],axis=['g2','-ln(L)'],marker=True,dir=self.dir,marker_value=c1[c_eval],title='g2 Fitting',print_pdf=True,model_g=self.model_g)
      pdb.set_trace()
      return (0.,csarray[decomposedLikelihood.argmin()])

  def computeAUC(self,outputs, target):
    thresholds = np.linspace(0,1.0,150) 
    outputs = np.log(outputs)
    outputs = outputs + np.abs(outputs.min())
    outputs = outputs/outputs.max()
    fnr = np.array([float(np.sum((outputs > tr) * (target == 0)))/float(np.sum(target == 0)) for tr in thresholds])
    tpr = np.array([float(np.sum((outputs < tr) * (target == 1)))/float(np.sum(target == 1)) for tr in thresholds])
    return auc(tpr,fnr)

  def evalC1C2AUC(self,w,testdata,c0,c1,c_eval=0,c_min=0.01,c_max=0.2,use_log=False,true_dist=False, vars_g=None, npoints=50,samples_ids=None,weights_func=None):

    if true_dist == True:
      vars = ROOT.TList()
      for var in vars_g:
        vars.Add(w.var(var))
      x = ROOT.RooArgSet(vars)
    else:
      x = None
    c0 = np.array([1.,0.,0.,0.,0.]) 

    score = ROOT.RooArgSet(w.var('score'))
    if use_log == True:
      evaluateRatio = self.evaluateLogDecomposedRatio
      post = 'log'
    else:
      evaluateRatio = self.evaluateDecomposedRatio
      post = ''
    testdata, testtarget = loadData('data',self.F0_dist,self.F1_dist,dir=self.dir,c1_g=self.c1_g,preprocessing=False) 

    csarray = np.linspace(c_min[0],c_max[0],npoints)
    csarray2 = np.linspace(c_min[1], c_max[1], npoints)
    decomposedAUC = np.zeros((npoints,npoints))
    decomposedS2 = np.zeros((npoints,npoints))
    c1s = np.zeros(c0.shape[0])
    pre_pdf = []
    pre_dist = []
    pre_pdf.extend([[],[]])
    pre_dist.extend([[],[]])
    # change this enumerates
    for k in range(nsamples):
      pre_pdf[0].append([])
      pre_pdf[1].append([])
      pre_dist[0].append([])
      pre_dist[1].append([])
      for j in range(nsamples):
        index_k,index_j = (self.basis_indexes[k],self.basis_indexes[j])
        if k <> j:
          f0pdf = w.function('bkghistpdf_{0}_{1}'.format(index_k,index_j))
          f1pdf = w.function('sighistpdf_{0}_{1}'.format(index_k,index_j))
          data = testdata
          if self.preprocessing == True:
            data = preProcessing(testdata,self.dataset_names[min(index_k,index_j)],
            self.dataset_names[max(index_k,index_j)],self.scaler) 
          outputs = predict('{0}/model/{1}/{2}/{3}_{4}_{5}.pkl'.format(self.dir,self.model_g,
          self.c1_g,self.model_file,index_k,index_j),data,model_g=self.model_g)
          #Evaluate histogram values with outputs directly
          #outputs = np.linspace(0.,1.,150) 
          f0pdfdist = np.array([self.evalDist(score,f0pdf,[xs]) for xs in outputs])
          f1pdfdist = np.array([self.evalDist(score,f1pdf,[xs]) for xs in outputs])
          pre_pdf[0][k].append(f0pdfdist)
          pre_pdf[1][k].append(f1pdfdist)
        else:
          pre_pdf[0][k].append(None)
          pre_pdf[1][k].append(None)
        if true_dist == True:
          f0 = w.pdf('f{0}'.format(index_k))
          f1 = w.pdf('f{0}'.format(index_j))
          if len(testdata.shape) > 1:
            f0dist = np.array([self.evalDist(x,f0,xs) for xs in testdata])
            f1dist = np.array([self.evalDist(x,f1,xs) for xs in testdata])
          else:
            f0dist = np.array([self.evalDist(x,f0,[xs]) for xs in testdata])
            f1dist = np.array([self.evalDist(x,f1,[xs]) for xs in testdata])
          pre_dist[0][k].append(f0dist)
          pre_dist[1][k].append(f1dist)
    indices = np.ones(testdata.shape[0], dtype=bool)
    ratiosList = []
    samples = []
    # This is needed for calibration of full ratios
    #for i,sample in enumerate(self.dataset_names):
    #  samples.append(np.loadtxt('{0}/data/{1}/{2}/{3}_{4}.dat'.format(self.dir,'mlp',self.c1_g,'data',sample)))

    n_eff_ratio = np.zeros((csarray.shape[0], csarray2.shape[0]))
    for i,cs in enumerate(csarray):
      ratiosList.append([])
      for j, cs2 in enumerate(csarray2):
        if weights_func <> None: 
          c1s = weights_func(cs,cs2)
          print '{0} {1}'.format(cs,cs2)
          #print c1s
        else:
          c1s[:] = c1[:]
          c1s[c_eval] = cs
        if self.cross_section <> None:
          c1s = np.multiply(c1s,self.cross_section)
        n_eff = c1s.sum()
        n_tot = np.abs(c1s).sum()
        n_eff_ratio[i,j] = n_eff/n_tot 

        print 'n_eff: {0}, n_tot: {1}, n_eff/n_tot: {2}'.format(n_eff, n_tot, n_eff/n_tot)
        c1s = c1s/c1s.sum()
        #print c1s
        decomposedRatios,trueRatios = evaluateRatio(w,testdata,x=x,
        plotting=False,roc=False,c0arr=c0,c1arr=c1s,true_dist=true_dist,pre_dist=pre_dist,
        pre_evaluation=pre_pdf)
        #decomposedRatios = 1./decomposedRatios

        targetdata = testtarget[decomposedRatios > 0.]
        decomposedRatios = decomposedRatios[decomposedRatios > 0.]
        outliers = self.findOutliers(decomposedRatios)
        decomposedRatios = decomposedRatios[outliers]
        targetdata = targetdata[outliers]

        bkgRatios = decomposedRatios[targetdata == 0]
        sigRatios = decomposedRatios[targetdata == 1]
      
        ratios_vec = [sigRatios, bkgRatios]
        minimum = min([sigRatios.min(),bkgRatios.min()])
        maximum = max([sigRatios.max(),bkgRatios.max()])

        bins = 150
        bkgValues = np.zeros(bins)
        sigValues = np.zeros(bins)
        histValues = [sigValues, bkgValues]
        ratios_vars = []
        low = minimum - ((maximum - minimum) / bins)*10
        high = maximum + ((maximum - minimum) / bins)*10
        names = ['sig','bkg']
        hists = []
        for l,(name,ratio) in enumerate(zip(names,ratios_vec)):
          w.factory('ratio{0}_{1}_{2}[{3},{4}]'.format(name,i,j, low, high))
          ratios_vars.append(w.var('ratio{0}_{1}_{2}'.format(name,i,j)))
          hist = ROOT.TH1F('{0}hist_F0_f0'.format(name),'hist',bins,low,high)
          for val in ratio:
            hist.Fill(val)
          norm = 1./hist.Integral()
          hist.Scale(norm) 
          hists.append(hist)
          datahist = ROOT.RooDataHist('{0}_datahist_F0_f0'.format(name),'hist',
                ROOT.RooArgList(ratios_vars[l]),hist)
          ratios_vars[l].setBins(bins)
          histpdf = ROOT.RooHistFunc('{0}_histpdf_F0_f0'.format(name),'hist',
                ROOT.RooArgSet(ratios_vars[l]), datahist, 0)

          histpdf.specialIntegratorConfig(ROOT.kTRUE).method1D().setLabel('RooBinIntegrator')
          getattr(w,'import')(hist)
          getattr(w,'import')(datahist) # work around for morph = w.import(morph)
          getattr(w,'import')(histpdf) # work around for morph = w.import(morph)
          #print '{0} {1} {2}'.format(curr,name,hist.Integral())
   
          #histpdf.specialIntegratorConfig(ROOT.kTRUE).method1D().setLabel('RooBinIntegrator')
          for actual_bin in range(bins):
            histValues[l][actual_bin] = hist.GetBinContent(actual_bin + 1)  
        printFrame(w,['ratiosig_{0}_{1}','ratiobkg_{0}_{1}'],[w.function('sig_histpdf_F0_f0'),
            w.function('bkg_histpdf_F0_f0')],makePlotName('ratio','temp',type='hist',dir=self.dir,model_g=self.model_g,c1_g=self.c1_g),['sig','bkg'],dir=self.dir,model_g=self.model_g,y_text='Count',title='Histograms for ratios',x_text='ratio value',print_pdf=True)
        S2 = self.singleRatio((histValues[0] + histValues[1]),((histValues[0] - histValues[1])**2))
        decomposedS2[i,j] = np.sum(S2)/2.
        ratiosList[i].append(decomposedRatios)
        print('{0} {1} '.format(i,j)),
        print decomposedS2[i,j]
    #decomposedLikelihood = decomposedLikelihood - decomposedLikelihood.min()
    #X,Y = np.meshgrid(csarray, csarray2)
    saveFig(csarray,[csarray2,n_eff_ratio],makePlotName('comp','train',type='n_eff'),labels=['composed'],pixel=True,marker=True,dir=self.dir,model_g=self.model_g,marker_value=(c1[0],c1[1]),print_pdf=True,contour=True,title='n_eff/n_tot values for g1,g2')
    #saveFig([],[vals[:size],vals1[:size]], 
    #    makePlotName('g1g2','train',type='hist'),hist=True,hist2D=True, 
    #    axis=['g1','g2'],marker=True,marker_value=c1,
    #    labels=labels,dir=dir,model_g=model_g,title='2D Histogram for fitted g1,g2', print_pdf=True,
    #    x_range=[[0.5,1.4],[1.1,1.9]])

  def evalC1C2Likelihood(self,w,testdata,c0,c1,c_eval=0,c_min=0.01,c_max=0.2,use_log=False,true_dist=False, vars_g=None, npoints=50,samples_ids=None,weights_func=None):

    if true_dist == True:
      vars = ROOT.TList()
      for var in vars_g:
        vars.Add(w.var(var))
      x = ROOT.RooArgSet(vars)
    else:
      x = None

    score = ROOT.RooArgSet(w.var('score'))
    if use_log == True:
      evaluateRatio = self.evaluateLogDecomposedRatio
      post = 'log'
    else:
      evaluateRatio = self.evaluateDecomposedRatio
      post = ''

    csarray = np.linspace(c_min[0],c_max[0],npoints)
    csarray2 = np.linspace(c_min[1], c_max[1], npoints)
    decomposedLikelihood = np.zeros((npoints,npoints))
    trueLikelihood = np.zeros((npoints,npoints))
    c1s = np.zeros(c0.shape[0])
    pre_pdf = []
    pre_dist = []
    pre_pdf.extend([[],[]])
    pre_dist.extend([[],[]])
    # change this enumerates
    for k,c0_ in enumerate(c0):
      pre_pdf[0].append([])
      pre_pdf[1].append([])
      pre_dist[0].append([])
      pre_dist[1].append([])
      for j,c1_ in enumerate(c0):
        index_k,index_j = (self.basis_indexes[k],self.basis_indexes[j])
        if k <> j:
          f0pdf = w.function('bkghistpdf_{0}_{1}'.format(index_k,index_j))
          f1pdf = w.function('sighistpdf_{0}_{1}'.format(index_k,index_j))
          data = testdata
          if self.preprocessing == True:
            data = preProcessing(testdata,self.dataset_names[min(index_k,index_j)],
            self.dataset_names[max(index_k,index_j)],self.scaler) 
          outputs = predict('{0}/model/{1}/{2}/{3}_{4}_{5}.pkl'.format(self.dir,self.model_g,
          self.c1_g,self.model_file,index_k,index_j),data,model_g=self.model_g)
          f0pdfdist = np.array([self.evalDist(score,f0pdf,[xs]) for xs in outputs])
          f1pdfdist = np.array([self.evalDist(score,f1pdf,[xs]) for xs in outputs])
          pre_pdf[0][k].append(f0pdfdist)
          pre_pdf[1][k].append(f1pdfdist)
        else:
          pre_pdf[0][k].append(None)
          pre_pdf[1][k].append(None)
        if true_dist == True:
          f0 = w.pdf('f{0}'.format(k))
          f1 = w.pdf('f{0}'.format(j))
          if len(testdata.shape) > 1:
            f0dist = np.array([self.evalDist(x,f0,xs) for xs in testdata])
            f1dist = np.array([self.evalDist(x,f1,xs) for xs in testdata])
          else:
            f0dist = np.array([self.evalDist(x,f0,[xs]) for xs in testdata])
            f1dist = np.array([self.evalDist(x,f1,[xs]) for xs in testdata])
          pre_dist[0][k].append(f0dist)
          pre_dist[1][k].append(f1dist)
    indices = np.ones(testdata.shape[0], dtype=bool)
    ratiosList = []
    samples = []
    # This is needed for calibration of full ratios
    #for i,sample in enumerate(self.dataset_names):
    #  samples.append(np.loadtxt('{0}/data/{1}/{2}/{3}_{4}.dat'.format(self.dir,'mlp',self.c1_g,'data',sample)))
    n_eff_ratio = np.zeros((csarray.shape[0], csarray2.shape[0]))
    for i,cs in enumerate(csarray):
      ratiosList.append([])
      for j, cs2 in enumerate(csarray2):
        if weights_func <> None: 
          c1s = weights_func(cs,cs2)
          #print '{0} {1}'.format(cs,cs2)
          #print c1s
        else:
          c1s[:] = c1[:]
          c1s[c_eval] = cs
        if self.cross_section <> None:
          c1s = np.multiply(c1s,self.cross_section)
        n_eff = c1s.sum()
        n_tot = np.abs(c1s).sum()
        n_eff_ratio[i,j] = n_eff/n_tot 
        #print '{0} {1}'.format(i,j)
        #print 'n_eff: {0}, n_tot: {1}, n_eff/n_tot: {2}'.format(n_eff, n_tot, n_eff/n_tot)
        c1s = c1s/c1s.sum()
        #print c1s
        decomposedRatios,trueRatios = evaluateRatio(w,testdata,x=x,
        plotting=False,roc=False,c0arr=c0,c1arr=c1s,true_dist=true_dist,pre_dist=pre_dist,
        pre_evaluation=pre_pdf)
        decomposedRatios = 1./decomposedRatios
        #calibratedRatios = self.calibrateFullRatios(w, decomposedRatios,
        #    c0,c1s,debug=debug,samples_data=samples,index=i) 
        #saveFig(decomposedRatios2, [calibratedRatios], makePlotName('calibrated_{0}'.format(i),'ratio',type='scat',
        #dir=self.dir, model_g=self.model_g, c1_g=self.c1_g),scatter=True, axis=['composed ratio', 
        #'composed calibrated'], dir=self.dir, model_g=self.model_g)
        ratiosList[i].append(decomposedRatios)
        #print('{0} {1} '.format(i,j)),
        #print decomposedRatios[decomposedRatios < 0.].shape 
        #print c1s
        #indices = np.logical_and(indices, decomposedRatios > 0.)
    for i,cs in enumerate(csarray):
      for j, cs2 in enumerate(csarray2):
        decomposedRatios = ratiosList[i][j]
        if use_log == False:
          if samples_ids <> None:
            ratios = decomposedRatios
            ids = samples_ids
            decomposedLikelihood[i,j] = (np.dot(np.log(ratios),
                np.array([c1[x] for x in ids]))).sum()
          else:
            #decomposedRatios[decomposedRatios < 0.] = 0.9
            decomposedRatios[decomposedRatios < 0.] = 1.0
            #decomposedRatios = decomposedRatios[self.findOutliers(decomposedRatios)]
            if n_eff_ratio[i,j] <= 0.5:
              #TODO: Harcoded number
              decomposedLikelihood[i,j] = 20000
            else:
              decomposedLikelihood[i,j] = -np.log(decomposedRatios).sum()
            #print decomposedLikelihood[i,j]
            #print '{0} {1} {2}'.format(i,j,decomposedLikelihood[i,j])
          trueLikelihood[i,j] = -np.log(trueRatios).sum()
        else:
          decomposedLikelihood[i,j] = decomposedRatios.sum()
          trueLikelihood[i,j] = trueRatios.sum()
      #print '\n {0}'.format(i)
    decomposedLikelihood = decomposedLikelihood - decomposedLikelihood.min()
    decMin = np.unravel_index(decomposedLikelihood.argmin(), decomposedLikelihood.shape)
    # pixel plots
    #saveFig(csarray,[csarray2,decomposedLikelihood],makePlotName('comp','train',type='likelihood_g1g2'),labels=['composed'],pixel=True,marker=True,dir=self.dir,model_g=self.model_g,marker_value=(c1[0],c1[1]),print_pdf=True,contour=True,title='Likelihood fit for g1,g2')

    #decMin = [np.sum(decomposedLikelihood,1).argmin(),np.sum(decomposedLikelihood,0).argmin()] 
    X,Y = np.meshgrid(csarray, csarray2)

    saveFig(X,[Y,decomposedLikelihood],makePlotName('comp','train',type='multilikelihood'),labels=['composed'],contour=True,marker=True,dir=self.dir,model_g=self.model_g,marker_value=(c1[0],c1[1]),print_pdf=True,min_value=(csarray[decMin[0]],csarray2[decMin[1]]))
    #print decMin
    print [csarray[decMin[0]],csarray2[decMin[1]]]
    pdb.set_trace()
    if true_dist == True:
      trueLikelihood = trueLikelihood - trueLikelihood.min()
      trueMin = np.unravel_index(trueLikelihood.argmin(), trueLikelihood.shape)
      saveFig(csarray,[decomposedLikelihood,trueLikelihood],makePlotName('comp','train',type=post+'likelihood_{0}'.format(n_sample)),labels=['decomposed','true'],axis=['c1[0]','-ln(L)'],marker=True,dir=self.dir,marker_value=c1[0],title='c1[0] Fitting',print_pdf=True)
      return [[csarray[trueMin[0]],csarray2[trueMin[1]]],
          [csarray2[decMin[0],csarray2[decMin[1]]]]]
    else:
      return [[0.,0.],[csarray[decMin[0]],csarray2[decMin[1]]]]

  def fitCValues(self,c0,c1,data_file = 'test',true_dist=False,vars_g=None,use_log=False,n_hist=150,num_pseudodata=1000,weights_func=None):
    if use_log == True:
      post = 'log'
    else:
      post = ''
    npoints = 15
    c_eval = 1
    c_min = [0.6,0.1]
    c_max = [1.5,0.9]
    #c_min = 0.6
    #c_max = 1.5

    f = ROOT.TFile('{0}/{1}'.format('/tmp/jpavezse/',self.workspace))
    w = f.Get('w')
    f.Close()
    assert w 

    #self.evalC1C2AUC(w,None, c0,c1,c_eval=c_eval,c_min=c_min,
    #   c_max=c_max,true_dist=true_dist,vars_g=vars_g,weights_func=weights_func,
    #                npoints=npoints,use_log=use_log)  
    #return

    print '{0} {1}'.format(c_min,c_max)
    rng = np.random.RandomState(self.seed)
    # Needed in case of working of NN with scaled features
    if self.preprocessing == True:
      if self.scaler == None:
        self.scaler = {}
        for k in range(self.nsamples):
         for j in enumerate(self.nsamples):
           if k < j:
            self.scaler[(k,j)] = joblib.load('{0}/model/{1}/{2}/{3}_{4}_{5}.dat'.format(self.dir,'mlp',self.c1_g,'scaler',self.dataset_names[k],self.dataset_names[j]))
  
    testdata = np.loadtxt('{0}/data/{1}/{2}/{3}_{4}.dat'.format(self.dir,'mlp',self.c1_g,data_file,self.F1_dist))
    #self.preMorphedBasis(c_eval=c_eval,c_min=c_min,c_max=c_max,npoints=npoints)   

    fil1 = open('{0}/fitting_values_c1.txt'.format(self.dir),'a')
    for i in range(n_hist):
      indexes = rng.choice(testdata.shape[0], num_pseudodata) 
      dataset = testdata[indexes]
      #(c1_true_1, c1_dec_1) = self.evalMorphC1Likelihood(w,dataset, c0,c1,c_eval=c_eval,c_min=c_min,
      #c_max=c_max,true_dist=true_dist,vars_g=vars_g,weights_func=weights_func,
      #              npoints=npoints,use_log=use_log,coef_index=c_eval)  
      ((c1_true,c2_true),(c1_dec,c2_dec)) = self.evalMorphC1C2Likelihood(w,dataset, c0,c1,c_eval=c_eval,c_min=c_min,
      c_max=c_max,true_dist=true_dist,vars_g=vars_g,weights_func=weights_func,
                    npoints=npoints,use_log=use_log)  
      print '2: {0} {1} {2} {3}'.format(c1_true, c1_dec, c2_true, c2_dec)
      fil1.write('{0} {1} {2} {3}\n'.format(c1_true, c1_dec, c2_true, c2_dec))
      #print '1: {0} {1}'.format(c1_true_1, c1_dec_1)
      #fil1.write('{0} {1}\n'.format(c1_true_1, c1_dec_1))
    fil1.close()  
