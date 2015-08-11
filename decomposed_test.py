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
          loadData,makeSigBkg,makeROC, makeMultiROC,saveMultiFig
from train_classifiers import predict


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
            verbose_printing=False):
    self.input_workspace = input_workspace
    self.workspace = output_workspace
    self.c0 = c0
    self.c1 = c1
    self.model_file = model_file
    self.dir = dir
    self.c1_g = c1_g
    self.model_g = model_g
    self.verbose_printing=verbose_printing
      

  def fit(self, data_file='test'):

    ''' 
      Create pdfs for the classifier 
      score to be used later on the ratio 
      test, input workspace only needed in case 
      there exist true pdfs for the distributions
      the models being used are ./model/{model_g}/{c1_g}/{model_file}_i_j.pkl
      and the data files are ./data/{model_g}/{c1_g}/{data_file}_i_j.dat
    '''

    bins = 40
    low = 0.
    high = 1.  
    
    if self.input_workspace <> None:
      f = ROOT.TFile('{0}/{1}'.format(self.dir,self.input_workspace))
      w = f.Get('w')
      f.Close()
    else: 
      w = ROOT.RooWorkspace('w')
      
    print 'Generating Score Histograms'

    w.factory('score[{0},{1}]'.format(low,high))
    s = w.var('score')
    
    #This is because most of the data of the full model concentrate around 0 
    bins_full = 40
    low_full = 0.
    high_full = 1.
    w.factory('scoref[{0},{1}]'.format(low_full, high_full))
    s_full = w.var('scoref')
    histos = []
    histos_names = []

    def saveHistos(w,ouputs,s,bins,low,high,pos=None):
      numtrain = outputs.shape[0]
      if pos <> None:
        k,j = pos
      else:
        k,j = ('F0','F1')
      for l,name in enumerate(['sig','bkg']):
        data = ROOT.RooDataSet('{0}data_{1}_{2}'.format(name,k,j),"data",
            ROOT.RooArgSet(s))
        hist = ROOT.TH1F('{0}hist_{1}_{2}'.format(name,k,j),'hist',bins,low,high)
        for val in outputs[l*numtrain/2:(l+1)*numtrain/2]:
          hist.Fill(val)
          s.setVal(val)
          data.add(ROOT.RooArgSet(s))
        
        datahist = ROOT.RooDataHist('{0}datahist_{1}_{2}'.format(name,k,j),'hist',
              ROOT.RooArgList(s),hist)
        s.setBins(bins)
        histpdf = ROOT.RooHistPdf('{0}histpdf_{1}_{2}'.format(name,k,j),'hist',
              ROOT.RooArgSet(s), datahist, 1)
      
        histpdf.specialIntegratorConfig(ROOT.kTRUE).method1D().setLabel('RooBinIntegrator')

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
          # print individual histograms
          #printFrame(w,[score_str],[w.pdf('sighistpdf_{0}_{1}'.format(k,j)), w.pdf('bkghistpdf_{0}_{1}'.format(k,j))], makePlotName(full,'train',k,j,type='hist',dir=self.dir,model_g=self.model_g,c1_g=self.c1_g),['signal','bkg'],dir=self.dir, model_g=self.model_g)
          if k < j and k <> 'F0':
            histos.append([w.pdf('sighistpdf_{0}_{1}'.format(k,j)), w.pdf('bkghistpdf_{0}_{1}'.format(k,j))])
            histos_names.append(['f{0}-f{1}_f{0}(signal)'.format(k,j), 'f{0}-f{1}_f{1}(background)'.format(k,j)])

    for k,c in enumerate(self.c0):
      for j,c_ in enumerate(self.c1):
        if k == j: 
          continue
        traindata, targetdata = loadData(data_file,k,j,dir=self.dir,c1_g=self.c1_g)
        numtrain = traindata.shape[0]       

        # Should I be using test data here?
        size2 = traindata.shape[1] if len(traindata.shape) > 1 else 1
        outputs = predict('{0}/model/{1}/{2}/{3}_{4}_{5}.pkl'.format(self.dir,self.model_g,self.c1_g,self.model_file,k,j),traindata.reshape(traindata.shape[0],size2),model_g=self.model_g)
        saveHistos(w,outputs,s,bins,low,high,(k,j))

    if self.verbose_printing==True:
      printMultiFrame(w,'score',histos, makePlotName('dec0','all',type='hist',dir=self.dir,c1_g=self.c1_g,model_g=self.model_g),histos_names,
        dir=self.dir,model_g=self.model_g,y_text='score(x)',print_pdf=True,title='Pairwise score distributions')

    # Full model
    traindata, targetdata = loadData(data_file,'F0','F1',dir=self.dir,c1_g=self.c1_g)
    numtrain = traindata.shape[0]       
    size2 = traindata.shape[1] if len(traindata.shape) > 1 else 1
    outputs = predict('{0}/model/{1}/{2}/{3}_F0_F1.pkl'.format(self.dir,self.model_g,self.c1_g,self.model_file),traindata.reshape(traindata.shape[0],size2),model_g=self.model_g)
    saveHistos(w,outputs,s_full, bins_full, low_full, high_full)
       
    w.Print()

    w.writeToFile('{0}/{1}'.format(self.dir,self.workspace))
      
  # To calculate the ratio between single functions
  def __singleRatio(self,x,f0,f1,val):
    iter = x.createIterator()
    v = iter.Next()
    i = 0
    while v:
      v.setVal(val[i])
      v = iter.Next()
      i = i+1
    if f0.getVal(x) == 0.:
      return 0.
    return f1.getVal(x) / f0.getVal(x)
    #return f0.getVal(ROOT.RooArgSet(x))

  def __evalDist(self,x,f0,val):
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

  def evaluateDecomposedRatio(self,w,evalData,x=None,plotting=True, roc=False,gridsize=None,c0arr=None, c1arr=None, true_dist=False):
    # pair-wise ratios
    # and decomposition computation
    #f = ROOT.TFile('{0}/{1}'.format(self.dir,self.workspace))
    #w = f.Get('w')
    #f.Close()

    score = ROOT.RooArgSet(w.var('score'))
    npoints = evalData.shape[0]
    fullRatios = np.zeros(npoints)
    fullRatiosReal = np.zeros(npoints)
    c0arr = self.c0 if c0arr == None else c0arr
    c1arr = self.c1 if c1arr == None else c1arr

    true_score = []
    train_score = []
    all_targets = []
    for k,c in enumerate(c0arr):
      innerRatios = np.zeros(npoints)
      innerTrueRatios = np.zeros(npoints)
      if c == 0:
        continue
      for j,c_ in enumerate(c1arr):
        f0pdf = w.pdf('bkghistpdf_{0}_{1}'.format(k,j))
        f1pdf = w.pdf('sighistpdf_{0}_{1}'.format(k,j))
        if k<>j:
          outputs = predict('{0}/model/{1}/{2}/{3}_{4}_{5}.pkl'.format(self.dir,self.model_g,self.c1_g,self.model_file,k,j),evalData,model_g=self.model_g)
          pdfratios = [self.__singleRatio(score,f0pdf,f1pdf,[xs]) for xs in outputs]
          pdfratios = np.array(pdfratios)
        else:
          pdfratios = np.ones(npoints)
        innerRatios += (c_/c) * pdfratios

        if true_dist == True:
          f0 = w.pdf('f{0}'.format(k))
          f1 = w.pdf('f{0}'.format(j))
          if len(evalData.shape) > 1:
            ratios = np.array([self.__singleRatio(x,f0,f1,xs) for xs in evalData])
          else:
            ratios = np.array([self.__singleRatio(x,f0,f1,[xs]) for xs in evalData])
          innerTrueRatios += (c_/c) * ratios

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
          clfRatios = np.array([self.__singleRatio(score,f0pdf,f1pdf,[xs]) for xs in outputs])
          train_score.append(clfRatios)
          if roc == True:
            all_targets.append(testtarget)
          #individual ROC
          #makeROC(clfRatios, testtarget,makePlotName('dec','train',k,j,type='roc',dir=self.dir,
          #model_g=self.model_g,c1_g=self.c1_g),dir=self.dir,model_g=self.model_g)
          if true_dist == True:
            if len(testdata.shape) > 1:
              trRatios = np.array([self.__singleRatio(x,f0,f1,xs) for xs in testdata])
            else:
              trRatios = np.array([self.__singleRatio(x,f0,f1,[xs]) for xs in testdata])

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
      innerRatios[innerRatios == np.inf] = 0.
      fullRatios += innerRatios
      if true_dist == True:
        innerTrueRatios = 1./innerTrueRatios
        innerTrueRatios[innerTrueRatios == np.inf] = 0.
        fullRatiosReal += innerTrueRatios
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
      saveMultiFig(evalData,[x for x in zip(train_score,true_score)], makePlotName('all_dec','train',type='ratio'),labels=[['f0-f1(trained)','f0-f1(truth)'],['f0-f2(trained)','f0-f2(truth)'],['f1-f2(trained)','f1-f2(truth)']],title='Pairwise Ratios',print_pdf=True)


    return fullRatios,fullRatiosReal

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
    
    print 'Calculating ratios'

    npoints = 50

    if true_dist == True:
      vars = ROOT.TList()
      for var in vars_g:
        vars.Add(w.var(var))
      x = ROOT.RooArgSet(vars)

    if use_log == True:
      evaluateRatio = evaluateLogDecomposedRatio
      post = 'log'
    else:
      evaluateRatio = self.evaluateDecomposedRatio
      post = ''

    score = ROOT.RooArgSet(w.var('score'))
    scoref = ROOT.RooArgSet(w.var('scoref'))

    if use_log == True:
      getRatio = singleLogRatio
    else:
      getRatio = self.__singleRatio
   
    # NN trained on complete model
    F0pdf = w.pdf('bkghistpdf_F0_F1')
    F1pdf = w.pdf('sighistpdf_F0_F1')

    testdata, testtarget = loadData(data_file,'F0',0,dir=self.dir,c1_g=self.c1_g) 
    if len(vars_g) == 1:
      xarray = np.linspace(0,5,npoints)
      fullRatios,_ = self.evaluateDecomposedRatio(w,xarray,x=x,plotting=True,roc=False,true_dist=True)

      y2 = [getRatio(x,w.pdf('F1'),w.pdf('F0'),[xs]) for xs in xarray]

      # NN trained on complete model
      outputs = predict('{0}/model/{1}/{2}/adaptive_F0_F1.pkl'.format(self.dir,self.model_g,self.c1_g),xarray.reshape(xarray.shape[0],1),model_g=self.model_g)
     
      pdfratios = [getRatio(scoref,F1pdf,F0pdf,[xs]) for xs in outputs]
      pdfratios = np.array(pdfratios)
      saveFig(xarray, [fullRatios, y2, pdfratios], makePlotName('all','train',type='ratio'+post),title='Likelihood Ratios',labels=['Composed trained', 'Truth', 'Full Trained'],print_pdf=True)
      
    if true_dist == True:
      decomposedRatio,_ = self.evaluateDecomposedRatio(w,testdata,x=x,plotting=False,roc=self.verbose_printing,true_dist=True)
    else:
      decomposedRatio,_ = self.evaluateDecomposedRatio(w,testdata,plotting=False,roc=self.verbose_printing)

    if len(testdata.shape) > 1:
      outputs = predict('{0}/model/{1}/{2}/{3}_F0_F1.pkl'.format(self.dir,self.model_g,self.c1_g,self.model_file),testdata.reshape(testdata.shape[0],testdata.shape[1]),model_g=self.model_g)
    else:
      outputs = predict('{0}/model/{1}/{2}/{3}_F0_F1.pkl'.format(self.dir,self.model_g,self.c1_g,self.model_file),testdata.reshape(testdata.shape[0],1),model_g=self.model_g)


    completeRatio = np.array([getRatio(scoref,F1pdf,F0pdf,[xs]) for xs in outputs])
    if true_dist == True:
      if len(testdata.shape) > 1:
        realRatio = np.array([getRatio(x,w.pdf('F1'),w.pdf('F0'),xs) for xs in testdata])
      else:
        realRatio = np.array([getRatio(x,w.pdf('F1'),w.pdf('F0'),[xs]) for xs in testdata])
    
    #Histogram F0-f0 for composed, full and true
    all_ratios_plots = []
    all_names_plots = []
    bins = 70
    low = 0.6
    high = 1.2
    if use_log == True:
      low = -1.0
      high = 1.0
    
    if true_dist == True:
      ratios_names = ['composed','full','truth']
      ratios_vec = [realRatio, completeRatio, decomposedRatio]
      minimum = min([realRatio.min(), completeRatio.min(), decomposedRatio.min()])
      maximum = max([realRatio.max(), completeRatio.max(), decomposedRatio.max()]) 
    else:
      ratios_names = ['composed','full']
      ratios_vec = [completeRatio, decomposedRatio]
      minimum = min([completeRatio.min(), decomposedRatio.min()])
      maximum = max([completeRatio.max(), decomposedRatio.max()]) 

    low = minimum - ((maximum - minimum) / bins)*10
    high = maximum + ((maximum - minimum) / bins)*10
    w.factory('ratio[{0},{1}]'.format(low, high))
    ratio = w.var('ratio')
    for curr, curr_ratios in zip(ratios_names,ratios_vec):
      numtest = curr_ratios.shape[0] 
      for l,name in enumerate(['sig','bkg']):
        hist = ROOT.TH1F('{0}_{1}hist_F0_f0'.format(curr,name),'hist',bins,low,high)
        for val in curr_ratios[l*numtest/2:(l+1)*numtest/2]:
          hist.Fill(val)
        datahist = ROOT.RooDataHist('{0}_{1}datahist_F0_f0'.format(curr,name),'hist',
              ROOT.RooArgList(ratio),hist)
        ratio.setBins(bins)
        histpdf = ROOT.RooHistFunc('{0}_{1}histpdf_F0_f0'.format(curr,name),'hist',
              ROOT.RooArgSet(ratio), datahist, 0)

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

    printMultiFrame(w,'ratio',all_ratios_plots, makePlotName('ratio','comparison',type='hist'+post,dir=self.dir,model_g=self.model_g,c1_g=self.c1_g),all_names_plots,setLog=True,dir=self.dir,model_g=self.model_g,y_text='Count',title='Histograms for ratios',x_text='ratio value',print_pdf=True)

    # scatter plot true ratio - composed - full ratio
    '''
    if self.verbose_printing == True and true_dist == True:
      saveFig(completeRatio,[realRatio], makePlotName('full','train',type='scat'+post,dir=self.dir,model_g=self.model_g,c1_g=self.c1_g),scatter=True,axis=['full trained ratio','true ratio'],dir=self.dir,model_g=self.model_g)
      saveFig(decomposedRatio,[realRatio], makePlotName('comp','train',type='scat'+post,dir=self.dir, model_g=self.model_g, c1_g=self.c1_g),scatter=True, axis=['composed trained ratio','true ratio'],dir=self.dir, model_g=self.model_g)
    '''
    # signal - bkg rejection plots
    ratios_list = [decomposedRatio/decomposedRatio.max(), 
                    completeRatio/completeRatio.max(),
                    realRatio/realRatio.max()]
    makeSigBkg(ratios_list,testtarget,makePlotName('comp','all',type='sigbkg'+post,dir=self.dir,model_g=self.model_g,c1_g=self.c1_g),dir=self.dir,model_g=self.model_g,print_pdf=True,legends=['composed','full','truth'],title='Signal-Background rejection curves')

    # Scatter plot to compare regression function and classifier score
    if self.verbose_printing == True and true_dist == True:
      testdata, testtarget = loadData('test','F0','F1',dir=self.dir,c1_g=self.c1_g) 
      if len(testdata.shape) > 1:
        reg = np.array([self.__regFunc(x,w.pdf('F0'),w.pdf('F1'),xs) for xs in testdata])
      else:
        reg = np.array([self.__regFunc(x,w.pdf('F0'),w.pdf('F1'),[xs]) for xs in testdata])
      if len(testdata.shape) > 1:
        outputs = predict('{0}/model/{1}/{2}/adaptive_F0_F1.pkl'.format(self.dir,self.model_g,self.c1_g),testdata.reshape(testdata.shape[0],testdata.shape[1]),model_g=self.model_g)
      else:
        outputs = predict('{0}/model/{1}/{2}/adaptive_F0_F1.pkl'.format(self.dir,self.model_g,self.c1_g),testdata.reshape(testdata.shape[0],1),model_g=self.model_g)

      #saveFig(outputs,[reg], makePlotName('full','train',type='scat',dir=self.dir,model_g=self.model_g,c1_g=self.c1_g),scatter=True,axis=['score','regression'],dir=self.dir,model_g=self.model_g)

