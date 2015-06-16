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
import pylab as plt

from mlp import make_predictions, train_mlp

''' 
 A simple example for the work on the section 
 5.4 of the paper 'Approximating generalized 
 likelihood ratio test with calibrated discriminative
 classifiers' by Kyle Cranmer
''' 

# Constants for each different model
c0 = np.array([.0,.3, .7])
c1 = np.array([.1,.3, .7])
#c1 = c1 / c1.sum()

#c1 = [.1,.5, .4]
verbose_printing = True
model_g = None
dir = '/afs/cern.ch/user/j/jpavezse/systematics'

def printFrame(w,obs,pdf,name,legends):
  '''
    This just print a bunch of pdfs 
    in a Canvas
  ''' 

  # Hope I don't need more colors ...
  colors = [ROOT.kBlack,ROOT.kRed,ROOT.kGreen,ROOT.kBlue,
    ROOT.kYellow]

  x = w.var(obs)
  funcs = []
  line_colors = []
  for i,p in enumerate(pdf):
    funcs.append(p)
    line_colors.append(ROOT.RooFit.LineColor(colors[i]))
  
  c1 = ROOT.TCanvas('c1')
  frame = x.frame()
  for i,f in enumerate(funcs):
      if isinstance(f,str):
        funcs[0].plotOn(frame, ROOT.RooFit.Components(f),ROOT.RooFit.Name(legends[i]), line_colors[i])
      else:
        f.plotOn(frame,ROOT.RooFit.Name(legends[i]),line_colors[i])
  leg = ROOT.TLegend(0.65, 0.73, 0.86, 0.87)
  #leg.SetFillColor(ROOT.kWhite)
  #leg.SetLineColor(ROOT.kWhite)
  for i,l in enumerate(legends):
    if i == 0:
      leg.AddEntry(frame.findObject(legends[i]), l, 'l')
    else:
      leg.AddEntry(frame.findObject(legends[i]), l, 'l')
  
  frame.Draw()
  leg.Draw()
  c1.SaveAs('{0}/plots/{1}/{2}.png'.format(dir,model_g,name))

def makeData(num_train=500,num_test=100):
  '''
  RooFit statistical model for the data
  
  '''  
  # Statistical model
  w = ROOT.RooWorkspace('w')
  #w.factory("EXPR::f1('cos(x)**2 + .01',x)")
  w.factory("EXPR::f2('exp(x*-1)',x[0,5])")
  w.factory("EXPR::f1('0.3 + exp(-(x-5)**2/5.)',x)")
  w.factory("EXPR::f0('exp(-(x-2.5)**2/1.)',x)")
  #w.factory("EXPR::f2('exp(-(x-2)**2/2)',x)")
  w.factory("SUM::F0(c00[{0}]*f0,c01[{1}]*f1,f2)".format(c0[0],c0[1]))
  w.factory("SUM::F1(c10[{0}]*f0,c11[{1}]*f1,f2)".format(c1[0],c1[1]))
  
  # Check Model
  w.Print()
  w.writeToFile('{0}/workspace_DecomposingTestOfMixtureModelsClassifiers.root'.format(dir))
  if verbose_printing == True:
    printFrame(w,'x',[w.pdf('f0'),w.pdf('f1'),w.pdf('f2')],'decomposed_model',['f0','f1','f2']) 
    printFrame(w,'x',[w.pdf('F0'),w.pdf('F1')],'full_model',['F0','F1'])
    printFrame(w,'x',[w.pdf('F1'),'f0'],'full_signal', ['F1','f0'])
  # Start generating data
  ''' 
    Each function will be discriminated pair-wise
    so n*n datasets are needed (maybe they can be reused?)
  ''' 
   
  def makeDataset(x,bkgpdf,sigpdf,num):
    traindata = np.zeros(num*2)
    targetdata = np.zeros(num*2)
    bkgdata = bkgpdf.generate(ROOT.RooArgSet(x),num)
    sigdata = sigpdf.generate(ROOT.RooArgSet(x),num)
    
    traindata[:num] = [sigdata.get(i).getRealValue('x') 
        for i in range(num)]
    targetdata[:num].fill(1)

    traindata[num:] = [bkgdata.get(i).getRealValue('x')
        for i in range(num)]
    targetdata[num:].fill(0)
    
    return traindata, targetdata  
 

  x = w.var('x')
  for k,c in enumerate(c0):
    for j,c_ in enumerate(c1):  
      traindata, targetdata = makeDataset(x,w.pdf('f{0}'.format(k)),w.pdf('f{0}'.format(j))
      ,num_train)
      #plt.hist(traindata[:10000],bins=100,color='red')
      #plt.hist(traindata[10000:],bins=100,color='blue')
      #plt.title('traindata_{1}_{2}'.format(model_g, k, j))
      #plt.show()
      np.savetxt('{0}/data/{1}/traindata_{2}_{3}.dat'.format(dir,model_g,k,j),
              np.column_stack((traindata,targetdata)),fmt='%f')

      testdata, testtarget = makeDataset(x,w.pdf('f{0}'.format(k)),w.pdf('f{0}'.format(j))
      ,num_test)

      np.savetxt('{0}/data/{1}/testdata_{2}_{3}.dat'.format(dir,model_g,k,j),
              np.column_stack((testdata,testtarget)),fmt='%f')


  traindata, targetdata = makeDataset(x,w.pdf('F0'),w.pdf('F1'),num_train)

  np.savetxt('{0}/data/{1}/traindata_F0_F1.dat'.format(dir,model_g),
          np.column_stack((traindata,targetdata)),fmt='%f')

  testdata, testtarget = makeDataset(x,w.pdf('F0'.format(k)),w.pdf('F1'.format(j))
  ,num_test)

  np.savetxt('{0}/data/{1}/testdata_F0_F1.dat'.format(dir,model_g),
          np.column_stack((testdata,testtarget)),fmt='%f')

  testdata, testtarget = makeDataset(x,w.pdf('F0'),w.pdf('f0'),num_test)


  np.savetxt('{0}/data/{1}/testdata_F0_f0.dat'.format(dir,model_g),
          np.column_stack((testdata,testtarget)),fmt='%f')


def loadData(filename):
  traintarget = np.loadtxt(filename)
  traindata = traintarget[:,0]
  targetdata = traintarget[:,1]
  return (traindata, targetdata)

def logit(p):
  return np.log(p) - np.log(1.-p)

def predict(filename, traindata):
  if model_g == 'mlp':
    result = logit(make_predictions(dataset=traindata, model_file=filename)[:,1])
    #return make_predictions(dataset=traindata, model_file=filename)[:,1]
    return result
  else:
    clf = joblib.load(filename)
    if clf.__class__.__name__ == 'NuSVR':
      output = clf.predict(traindata)
      return np.clip(output,0.,1.)
    else:
      return clf.predict_proba(traindata)[:,1]

def makeROC(outputs, target, label):
  '''
    make plots for ROC curve of classifier and 
    test data.
  '''
  fpr, tpr, _  = roc_curve(target.ravel(),outputs.ravel())
  roc_auc = auc(fpr, tpr)
  fig = plt.figure()
  plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
  plt.plot([0, 1], [0, 1], 'k--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('{0}'.format(label))
  plt.legend(loc="lower right")
  np.savetxt('{0}/plots/{1}/{2}.txt'.format(dir,model_g,label),np.column_stack((fpr,tpr)))
  plt.savefig('{0}/plots/{1}/{2}.png'.format(dir,model_g,label))
  plt.close(fig)
  plt.clf()


def makeSigBkg(outputs, target, label):
  '''
    make plots for ROC curve of classifier and 
    test data.
  '''
  fpr, tpr, _  = roc_curve(target.ravel(),outputs.ravel())
  tnr = 1. - fpr
  roc_auc = auc(tpr,tnr)
  fig = plt.figure()
  plt.plot(tpr, tnr, label='Signal Eff Bkg Rej (area = %0.2f)' % roc_auc)
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('Signal Efficiency')
  plt.ylabel('Background Rejection')
  plt.title('{0}'.format(label))
  plt.legend(loc="lower right")
  np.savetxt('{0}/plots/{1}/{2}.txt'.format(dir,model_g,label),np.column_stack((fpr,tpr)))
  plt.savefig('{0}/plots/{1}/{2}.png'.format(dir,model_g,label))
  plt.close(fig)
  plt.clf()

def makePlotName(full, truth, f0 = None, f1 = None, type=None):
  if full == 'decomposed':
    return '{0}_{1}_f{2}_f{3}_{4}_{5}'.format(full, truth, f0, f1, model_g,type)
  else:
    return '{0}_{1}_{2}_{3}'.format(full, truth, model_g,type)

def trainClassifier(clf):
  '''
    Train classifiers pair-wise on 
    datasets
  '''

  for k,c in enumerate(c0):
    for j,c_ in enumerate(c1):

      print " Training Classifier on f{0}/f{1}".format(k,j)
      #clf = svm.NuSVC(probability=True) #Why use a SVR??
      if model_g == 'mlp':
        train_mlp(dataset='{0}/data/{1}/traindata_{2}_{3}.dat'.format(dir,model_g,k,j),
          save_file='{0}/model/{1}/adaptive_{2}_{3}.pkl'.format(dir,model_g,k,j))
      else:
        traindata,targetdata = loadData('{0}/data/{1}/traindata_{2}_{3}.dat'.format(dir,model_g,k,j)) 
        clf.fit(traindata.reshape(traindata.shape[0],1)
            ,targetdata)
        joblib.dump(clf, '{0}/model/{1}/adaptive_{2}_{3}.pkl'.format(dir,model_g,k,j))


      #makeROC(outputs, testtarget, makePlotName('decomposed','trained',k,j,'roc'))
  
  print " Training Classifier on F0/F1"
  if model_g == 'mlp':
    train_mlp(dataset='{0}/data/{1}/traindata_F0_F1.dat'.format(dir,model_g), 
        save_file='{0}/model/{1}/adaptive_F0_F1.pkl'.format(dir,model_g))
  else:
    traindata,targetdata = loadData('{0}/data/{1}/traindata_F0_F1.dat'.format(dir,model_g)) 
    #clf = svm.NuSVC(probability=True) #Why use a SVR??
    clf.fit(traindata.reshape(traindata.shape[0],1)
        ,targetdata)
    joblib.dump(clf, '{0}/model/{1}/adaptive_F0_F1.pkl'.format(dir,model_g))

  #testdata, testtarget = loadData('data/{0}/testdata_F0_F1.dat'.format(model_g)) 
  #outputs = predict(clf,testdata.reshape(testdata.shape[0],1))
  #makeROC(outputs, testtarget, makePlotName('full','trained',type='roc'))


def classifierPdf():
  ''' 
    Create pdfs for the classifier 
    score to be used later on the ratio 
    test
  '''

  bins = 70
  low = -7.
  high = 7.  

  f = ROOT.TFile('{0}/workspace_DecomposingTestOfMixtureModelsClassifiers.root'.format(dir))
  w = f.Get('w')
  f.Close()

  w.factory('score[{0},{1}]'.format(low,high))
  s = w.var('score')
  
  #This is because most of the data of the full model concentrate around 0 
  bins_full = 70
  low_full = -2.
  high_full = 2.
  w.factory('scoref[{0},{1}]'.format(low_full, high_full))
  s_full = w.var('scoref')

  def saveHistos(w,ouputs,s,bins,low,high,pos=None):
    numtrain = outputs.shape[0]
    if pos <> None:
      k,j = pos
    else:
      k,j = ('F0','F1')
    for l,name in enumerate(['sig','bkg']):
      data = ROOT.RooDataSet('{0}data_{1}_{2}'.format(name,k,j),"data",
          ROOT.RooArgSet(s))
      #low = outputs.min()
      #high = outputs.max() 
      hist = ROOT.TH1F('{0}hist_{1}_{2}'.format(name,k,j),'hist',bins,low,high)
      for val in outputs[l*numtrain/2:(l+1)*numtrain/2]:
        hist.Fill(val)
        s.setVal(val)
        data.add(ROOT.RooArgSet(s))
      
      datahist = ROOT.RooDataHist('{0}datahist_{1}_{2}'.format(name,k,j),'hist',
            ROOT.RooArgList(s),hist)
      s.setBins(bins)
      histpdf = ROOT.RooHistPdf('{0}histpdf_{1}_{2}'.format(name,k,j),'hist',
            ROOT.RooArgSet(s), datahist, 0)
    
      histpdf.specialIntegratorConfig(ROOT.kTRUE).method1D().setLabel('RooBinIntegrator')

      getattr(w,'import')(hist)
      getattr(w,'import')(data)
      getattr(w,'import')(datahist) # work around for morph = w.import(morph)
      getattr(w,'import')(histpdf) # work around for morph = w.import(morph)

      score_str = 'scoref' if pos == None else 'score'

      # Calculate the density of the classifier output using kernel density 
      # estimation technique
      w.factory('KeysPdf::{0}dist_{1}_{2}({3},{0}data_{1}_{2})'.format(name,k,j,score_str))

      # Print histograms pdfs and estimated densities
      if verbose_printing == True and name == 'bkg' and k <> j:
        full = 'full' if pos == None else 'decomposed'
        # print histograms
        #printFrame(w,score_str,[w.pdf('sighistpdf_{0}_{1}'.format(k,j)), w.pdf('bkghistpdf_{0}_{1}'.format(k,j))], makePlotName(full,'trained',k,j,type='hist'),['signal','bkg'])
        # print histogram and density estimation together
        printFrame(w,score_str,[w.pdf('sighistpdf_{0}_{1}'.format(k,j)), w.pdf('bkghistpdf_{0}_{1}'.format(k,j)),w.pdf('sigdist_{0}_{1}'.format(k,j)),w.pdf('bkgdist_{0}_{1}'.format(k,j))], makePlotName(full,'trained',k,j,type='hist'),['signal_hist','bkg_hist','signal_est','bkg_est'])
        # print density estimation
        #printFrame(w,'score',[w.pdf('sigdist_{0}_{1}'.format(k,j)), w.pdf('bkgdist_{0}_{1}'.format(k,j))], makePlotName(full,'trained',k,j,type='density'),['signal','bkg'])

  for k,c in enumerate(c0):
    for j,c_ in enumerate(c1):
      traindata, targetdata = loadData('{0}/data/{1}/traindata_{2}_{3}.dat'.format(dir,model_g,k,j))
      numtrain = traindata.shape[0]       

      # Should I be using test data here?
      outputs = predict('{0}/model/{1}/adaptive_{2}_{3}.pkl'.format(dir,model_g,k,j),traindata.reshape(traindata.shape[0],1))
      #outputs = clf.predict_proba(traindata.reshape(traindata.shape[0],1)) 
      saveHistos(w,outputs,s,bins,low,high,(k,j))

  traindata, targetdata = loadData('{0}/data/{1}/traindata_F0_F1.dat'.format(dir,model_g))
  numtrain = traindata.shape[0]       

  # Should I be using test data here?
  outputs = predict('{0}/model/{1}/adaptive_F0_F1.pkl'.format(dir,model_g),traindata.reshape(traindata.shape[0],1))
  #outputs = clf.predict_proba(traindata.reshape(traindata.shape[0],1)) 
  saveHistos(w,outputs,s_full, bins_full, low_full, high_full)
     
  w.Print()

  w.writeToFile('{0}/workspace_DecomposingTestOfMixtureModelsClassifiers.root'.format(dir))


# this is not very efficient
def scikitlearnFunc(filename,x=0.):
  '''
    Needed for the scikit-learn wrapper
  '''
  traindata = np.array(([x])) if model_g == 'mlp' else np.array((x))
  outputs = predict(filename,traindata)[0]

  #if outputs[0] > 1:
  #  return 1.
  return outputs

class ScikitLearnCallback:
  def __init__(self,file):
    filename = file

  def __call__(self,x = 0.):
    train = np.array((x))
    outputs = predict(filename,train)[0]
    
    return outputs


def saveFig(x,y,file,labels=None,scatter=False,axis=None):
  fig,ax = plt.subplots()
  if scatter == True:
    if len(y) == 1: 
      ax.scatter(x,y[0],s=2)
      ax.set_xlabel(axis[0])
      ax.set_ylabel(axis[1])
    else:
      sc1 = ax.scatter(x,y[0],color='black')
      sc2 = ax.scatter(x,y[1],color='red')
      ax.legend((sc1,sc2),(labels[0],labels[1]))
      ax.set_xlabel('x')
      ax.set_ylabel('regression(score)')
  else:
    if len(y) == 1:
      ax.plot(x,y[0],'b')
    else:
      #Just supporting two plots for now
      ax.plot(x,y[0],'b-',label=labels[0]) 
      ax.plot(x,y[1],'r-',label=labels[1])
      ax.legend()
    ax.set_ylabel('LR')
    ax.set_xlabel('x')
  ax.set_title(file)
  if (len(y) > 1):
    # This breaks the naming convention for plots, I will solve
    # it later
    for i,l in enumerate(labels):
      np.savetxt('{0}/plots/{1}/{2}_{3}.txt'.format(dir,model_g,file,l),y[i])
  else:
    np.savetxt('{0}/plots/{1}/{2}.txt'.format(dir,model_g,file),y[0])
  fig.savefig('{0}/plots/{1}/{2}.png'.format(dir,model_g,file))
  plt.close(fig)
  plt.clf()

def fitAdaptive(use_log = False):
  '''
    Use the computed score densities to compute 
    the decompose ratio test
  '''
  ROOT.gSystem.Load('{0}/parametrized-learning/SciKitLearnWrapper/libSciKitLearnWrapper'.format(dir))
  ROOT.gROOT.ProcessLine('.L CompositeFunctionPdf.cxx+')


  f = ROOT.TFile('{0}/workspace_DecomposingTestOfMixtureModelsClassifiers.root'.format(dir))
  w = f.Get('w')
  f.Close()

  #x = w.var('x[-5,5]')
  x = ROOT.RooRealVar('x','x',0.2,0.,5.)
  getattr(w,'import')(ROOT.RooArgSet(x),ROOT.RooFit.RecycleConflictNodes()) 

  def constructDensity(w,pos = None):
    if pos <> None:
      k,j = pos
    else:
      k,j = ('F0','F1')
    #test = scikitlearnFunc('model/{0}/adaptive_{1}_{2}.pkl'.format(model_g,k,j),2.0)
    nn = ROOT.SciKitLearnWrapper('nn_{0}_{1}'.format(k,j),'nn_{0}_{1}'.format(k,j),x)
    nn.RegisterCallBack(lambda x: scikitlearnFunc('{0}/model/{1}/adaptive_{2}_{3}.pkl'.format(dir,model_g
    ,k,j),x))

    #printFrame(w,'x',[nn],makePlotName('decomposed','trained',k,j,'score'),['score'])

    # I should find the way to use this method
    #callbck = ScikitLearnCallback('model/{0}/adaptive_{1}_{2}.pkl'.format(model_g,k,j))
    #nn.RegisterCallBack(lambda x: callbck(x))

    getattr(w,'import')(ROOT.RooArgSet(nn),ROOT.RooFit.RecycleConflictNodes()) 

    # Inserting the nn output into the pdf graph
    for l,name in enumerate(['sig','bkg']):
      #w.factory('CompositeFunctionPdf::{0}template_{1}_{2}({0}histpdf_{1}_{2})'.
      #    format(name,k,j))
      #w.factory('CompositeFunctionPdf::{0}template_{1}_{2}({0}dist_{1}_{2})'.
      #    format(name,k,j))
      w.factory('EDIT::{0}moddist_{1}_{2}({0}histpdf_{1}_{2},score=nn_{1}_{2})'
              .format(name,k,j))

    if verbose_printing == False and k <> j:
      full = 'full' if pos == None else 'decomposed'
      printFrame(w,'x',[w.pdf('sigmoddist_{0}_{1}'.format(k,j)),
                w.pdf('bkgmoddist_{0}_{1}'.format(k,j))],makePlotName(full,'trained',k,j,'dist'),['signal','bkg'])

  # Avoiding the composition since make MLP prediction very slow, 
  # still is usefull to print density distributions

  #for k,c in enumerate(c0):
  #  for j,c_ in enumerate(c1):

      #constructDensity(w,(k,j))
      #w.Print()
      # Save graphs
      #sigpdf.graphVizTree('sigpdfgraph.dot')
      #bkgpdf.graphVizTree('bkgpdfgraph.dot')
      
  #constructDensity(w)

  # To calculate the ratio between single functions
  def singleRatio(x,f0,f1,val):
    x.setVal(val)
    if f0.getVal(ROOT.RooArgSet(x)) < 10E-10:
      return 0.
    return f1.getVal(ROOT.RooArgSet(x)) / f0.getVal(ROOT.RooArgSet(x))
    #return f0.getVal(ROOT.RooArgSet(x))


  # To calculate the ratio between single functions
  def regFunc(x,f0,f1,val):
    x.setVal(val)
    if (f0.getVal(ROOT.RooArgSet(x)) + f1.getVal(ROOT.RooArgSet(x))) < 10E-10:
      return 0.
    return f1.getVal(ROOT.RooArgSet(x)) / (f0.getVal(ROOT.RooArgSet(x)) + f1.getVal(ROOT.RooArgSet(x)))

  # Functions for Log Ratios
  def singleLogRatio(x, f0, f1, val):
    x.setVal(val)
    rat = np.log(f1.getVal(ROOT.RooArgSet(x))) - np.log(f0.getVal(ROOT.RooArgSet(x)))
    return rat
  def computeLogKi(x, f0, f1, c0, c1, val):
    x.setVal(val)
    k_ij = np.log(c1*f1.getVal(ROOT.RooArgSet(x))) - np.log(c0*f0.getVal(ROOT.RooArgSet(x)))
    return k_ij
  # ki is a vector
  def computeAi(k0, ki):
    ai = -k0 - np.log(1. + np.sum(np.exp(ki - k0),0))
    return ai

  # pair-wise ratios
  # and decomposition computation
  npoints = 100
  x = w.var('x')

  #Log-Ratio computation
  def evaluateLogDecomposedRatio(w, x, xarray, plotting = True, roc = False):
    score = w.var('score')
    npoints = xarray.shape[0]
    fullRatios = np.zeros(npoints)
    ksTrained = np.zeros((c0.shape[0]-1, c1.shape[0],npoints))
    ks = np.zeros((c0.shape[0]-1, c1.shape[0],npoints))
    k0Trained = np.zeros(npoints)
    k0 = np.zeros(npoints)
    ai = np.zeros((c0.shape[0]-1,npoints))
    aiTrained = np.zeros((c0.shape[0]-1,npoints))
    # I can do this with list comprehension
    for k, c in enumerate(c0):
      if c == 0:
        continue
      for j, c_ in enumerate(c1):
        f0pdf = w.pdf('bkghistpdf_{0}_{1}'.format(k,j))
        f1pdf = w.pdf('sighistpdf_{0}_{1}'.format(k,j))
        f0 = w.pdf('f{0}'.format(k))
        f1 = w.pdf('f{0}'.format(j))
        outputs = predict('model/{0}/adaptive_{1}_{2}.pkl'.format(model_g,k,j),
                  xarray.reshape(xarray.shape[0],1))
        ks[k-1][j] = np.array([computeLogKi(x,f0,f1,c,c_,xs) for xs in xarray])
        if k == j:
          ksTrained[k-1][j] = ks[k-1][j]
        else:
          ksTrained[k-1][j] = np.array([computeLogKi(score,f0pdf,f1pdf,c,c_,xs) for xs in outputs])
        if plotting == True and k <> j:
          saveFig(xarray, [ks[k-1][j],ksTrained[k-1][j]], makePlotName('decomposed','trained',k,j,type='ratio_log'),
            ['trained','truth'])
        if roc == True and k <> j:
          testdata, testtarget = loadData('data/{0}/testdata_{1}_{2}.dat'.format(model_g,k,j)) 
          outputs = predict('model/{0}/adaptive_{1}_{2}.pkl'.format(model_g,k,j),
                    testdata.reshape(testdata.shape[0],1))
          clfRatios = [singleRatio(score,f0pdf,f1pdf,xs) for xs in outputs]
          trRatios = [singleRatio(x,f0,f1,xs) for xs in testdata]
          makeROC(np.array(trRatios), testtarget, makePlotName('decomposed','truth',k,j,type='roc'))
          makeROC(np.array(clfRatios), testtarget,makePlotName('decomposed','trained',k,j,type='roc'))
          # Scatter plot to compare regression function and classifier score
          reg = np.array([regFunc(x,f0,f1,xs) for xs in testdata])
          #reg = reg/np.max(reg)
          #pdb.set_trace()
          saveFig(outputs,[reg], makePlotName('decomposed','trained',k,j,type='scatter'),scatter=True, axis=['score','regression'])
          saveFig(testdata, [reg, outputs],  makePlotName('decomposed','trained',k,j,type='multi_scatter'),scatter=True,labels=['regression', 'score'])
      #check this
      kSortedTrained = np.sort(ksTrained[k-1],0)
      kSorted = np.sort(ks[k-1],0)
      ai[k-1] = computeAi(kSorted[0],kSorted[1:])
      aiTrained[k-1] = computeAi(kSortedTrained[0],kSortedTrained[1:])
    aSorted = np.sort(ai,0) 
    aSortedTrained = np.sort(aiTrained,0)
    ratios = aSorted[0] + np.log(1. + np.sum(np.exp(aSorted[1:] - aSorted[0]),0))
    pdfratios = aSortedTrained[0] + np.log(1. + np.sum(np.exp(aSortedTrained[1:] - aSortedTrained[0]),0))
    return pdfratios      

  #Ratio computation
  def evaluateDecomposedRatio(w,x,xarray,plotting=True, roc=False):
    score = w.var('score')
    npoints = xarray.shape[0]
    fullRatios = np.zeros(npoints)
    for k,c in enumerate(c0):
      innerRatios = np.zeros(npoints)
      if c == 0:
        continue
      for j,c_ in enumerate(c1):
        #testdata, testtarget = loadData('data/{0}/testdata_{1}_{2}.dat'.format(model_g, k, j)) 
        #xarray = np.sort(testdata)
        #xarray = np.sort(testdata)

        f0pdf = w.pdf('bkghistpdf_{0}_{1}'.format(k,j))
        f1pdf = w.pdf('sighistpdf_{0}_{1}'.format(k,j))
        f0 = w.pdf('f{0}'.format(k))
        f1 = w.pdf('f{0}'.format(j))
        outputs = predict('{0}/model/{1}/adaptive_{2}_{3}.pkl'.format(dir,model_g,k,j),
                  xarray.reshape(xarray.shape[0],1))
        pdfratios = [singleRatio(score,f0pdf,f1pdf,xs) for xs in outputs]
        # the cases in which both distributions are the same can be problematic
        # one will expect that the classifier gives same prob to both signal and bkg
        # but it can behave in weird ways, I will just avoid this for now 
        pdfratios = np.array(pdfratios) if k <> j else np.ones(npoints)
        innerRatios += (c_/c) * pdfratios
        ratios = [singleRatio(x,f0,f1,xs) for xs in xarray]
        #if k == 1 and  j == 2:
          #pdb.set_trace()
        if plotting == True and k <> j:
          saveFig(xarray, [pdfratios,ratios], makePlotName('decomposed','trained',k,j,type='ratio'),
            ['trained','truth'])
        if roc == True and k <> j:
          testdata, testtarget = loadData('{0}/data/{1}/testdata_{2}_{3}.dat'.format(dir,model_g,k,j)) 
          outputs = predict('{0}/model/{1}/adaptive_{2}_{3}.pkl'.format(dir,model_g,k,j),
                    testdata.reshape(testdata.shape[0],1))
          clfRatios = [singleRatio(score,f0pdf,f1pdf,xs) for xs in outputs]
          trRatios = [singleRatio(x,f0,f1,xs) for xs in testdata]
          makeROC(np.array(trRatios), testtarget, makePlotName('decomposed','truth',k,j,type='roc'))
          makeROC(np.array(clfRatios), testtarget,makePlotName('decomposed','trained',k,j,type='roc'))
          # Scatter plot to compare regression function and classifier score
          reg = np.array([regFunc(x,f0,f1,xs) for xs in testdata])
          #reg = reg/np.max(reg)
          #pdb.set_trace()
          saveFig(outputs,[reg], makePlotName('decomposed','trained',k,j,type='scatter'),scatter=True, axis=['score','regression'])
          saveFig(testdata, [reg, outputs],  makePlotName('decomposed','trained',k,j,type='multi_scatter'),scatter=True,labels=['regression', 'score'])


        #saveFig(xarray, ratios, makePlotName('decomposed','truth',k,j,type='ratio'))
      fullRatios += 1./innerRatios
    return fullRatios

  if use_log == True:
    evaluateRatio = evaluateLogDecomposedRatio
  else:
    evaluateRatio = evaluateDecomposedRatio

  score = w.var('score')
  scoref = w.var('scoref')
  xarray = np.linspace(0,5,npoints)
 
  fullRatios = evaluateRatio(w,x,xarray)

  saveFig(xarray, [fullRatios],  makePlotName('composite','trained',type='ratio')) 

  if use_log == True:
    getRatio = singleLogRatio
  else:
    getRatio = singleRatio
    
  y2 = [getRatio(x,w.pdf('F1'),w.pdf('F0'),xs) for xs in xarray]

  saveFig(xarray, [y2], makePlotName('full','truth',type='ratio'))
  saveFig(xarray, [np.array(y2) - fullRatios], makePlotName('composite','trained',type='diff'))

  # NN trained on complete model
  F0pdf = w.pdf('bkghistpdf_F0_F1')
  F1pdf = w.pdf('sighistpdf_F0_F1')
  outputs = predict('{0}/model/{1}/adaptive_F0_F1.pkl'.format(dir,model_g),xarray.reshape(xarray.shape[0],1))
 
  pdfratios = [getRatio(scoref,F1pdf,F0pdf,xs) for xs in outputs]
  pdfratios = np.array(pdfratios)
  saveFig(xarray, [pdfratios], makePlotName('full','trained',type='ratio'))
  saveFig(xarray, [np.array(y2) - pdfratios],makePlotName('full','trained',type='diff'))

  # ROC for ratios
  # load test data
  # check if ratios fulfill the requeriments of type
  testdata, testtarget = loadData('{0}/data/{1}/testdata_F0_f0.dat'.format(dir,model_g)) 
  decomposedRatio = evaluateRatio(w,x,testdata,plotting=False,roc=True)
  outputs = predict('{0}/model/{1}/adaptive_F0_F1.pkl'.format(dir,model_g),testdata.reshape(testdata.shape[0],1))
  completeRatio = [getRatio(scoref,F1pdf,F0pdf,xs) for xs in outputs]
  realRatio = [getRatio(x,w.pdf('F1'),w.pdf('F0'),xs) for xs in testdata]

  #Histogram F0-f0 for decomposed
  bins = 70
  low = -1.
  high = 3.
  w.factory('ratio[{0},{1}]'.format(low, high))
  ratio = w.var('ratio')
  numtest = decomposedRatio.shape[0] 
  for l,name in enumerate(['sig','bkg']):
    hist = ROOT.TH1F('{0}hist_F0_f0'.format(name),'hist',bins,low,high)
    for val in decomposedRatio[l*numtest/2:(l+1)*numtest/2]:
      hist.Fill(val)
    datahist = ROOT.RooDataHist('{0}datahist_F0_f0'.format(name),'hist',
          ROOT.RooArgList(ratio),hist)
    ratio.setBins(bins)
    histpdf = ROOT.RooHistPdf('{0}histpdf_F0_f0'.format(name),'hist',
          ROOT.RooArgSet(ratio), datahist, 0)

    histpdf.specialIntegratorConfig(ROOT.kTRUE).method1D().setLabel('RooBinIntegrator')
    getattr(w,'import')(hist)
    getattr(w,'import')(datahist) # work around for morph = w.import(morph)
    getattr(w,'import')(histpdf) # work around for morph = w.import(morph)

    if verbose_printing == True and name == 'bkg':
      printFrame(w,'ratio',[w.pdf('sighistpdf_F0_f0'), w.pdf('bkghistpdf_F0_f0')], makePlotName('composite','trained',type='hist'),['signal','bkg'])


  saveFig(completeRatio,[realRatio], makePlotName('full','trained',type='scatter'),scatter=True,axis=['full trained ratio','true ratio'])
  saveFig(decomposedRatio,[realRatio], makePlotName('composite','trained',type='scatter'),scatter=True, axis=['composed trained ratio','true ratio'])

  makeSigBkg(1.-np.array(realRatio), testtarget,makePlotName('full','truth',type='sigbkg'))
  makeSigBkg(1.-np.array(decomposedRatio), testtarget,makePlotName('composite','trained',type='sigbkg'))
  makeSigBkg(1.-np.array(completeRatio), testtarget,makePlotName('full','trained',type='sigbkg'))


  testdata, testtarget = loadData('{0}/data/{1}/testdata_F0_F1.dat'.format(dir,model_g)) 
  # Scatter plot to compare regression function and classifier score
  reg = np.array([regFunc(x,w.pdf('F0'),w.pdf('F1'),xs) for xs in testdata])
  #reg = reg/np.max(reg)
  outputs = predict('{0}/model/{1}/adaptive_F0_F1.pkl'.format(dir,model_g),testdata.reshape(testdata.shape[0],1))
  #pdb.set_trace()
  saveFig(outputs,[reg], makePlotName('full','trained',type='scatter'),scatter=True,axis=['score','regression'])
  saveFig(testdata, [reg, outputs],  makePlotName('full','trained',type='multi_scatter'),scatter=True,labels=['regression', 'score'])


  #w.Print()

if __name__ == '__main__':
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
  
  use_log = False
  if (len(sys.argv) > 2):
    use_log = True

  c1[0] = sys.argv[2]
  c1 = c1 / c1.sum()
  print c0
  print c1
  
  # Set this value to False if only final plots are needed
  verbose_printing = True

  makeData(num_train=100000,num_test=50000) 
  trainClassifier(clf)
  classifierPdf()
  fitAdaptive()

