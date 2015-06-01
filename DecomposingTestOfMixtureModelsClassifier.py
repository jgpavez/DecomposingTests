#!/usr/bin/env python
__author__ = "Pavez J. <juan.pavezs@alumnos.usm.cl>"


import ROOT
import numpy as np
from sklearn import svm, linear_model
from sklearn.externals import joblib
from sklearn.metrics import roc_curve, auc

import sys

import os.path
import pdb

import matplotlib.pyplot as plt
import pylab as plt


''' 
 A simple example for the work on the section 
 5.4 of the paper 'Approximating generalized 
 likelihood ratio test with calibrated discriminative
 classifiers' by Kyle Cranmer
''' 

# Constants for each different model
c0 = [.0,.3, .7]
c1 = [.1,.5, .4]
verbose_printing = True
model_g = None

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
  c1.SaveAs('plots/{0}/{1}.png'.format(model_g,name))

def makeData(num_train=500,num_test=100):
  '''
  RooFit statistical model for the data
  
  '''  
  # Statistical model
  w = ROOT.RooWorkspace('w')
  w.factory("EXPR::f0('exp(x*-1)',x[0,5])")
  w.factory("EXPR::f1('cos(x)**2 + .01',x)")
  w.factory("EXPR::f2('exp(-(x-2)**2/2)',x)")
  w.factory("SUM::F0(c00[{0}]*f0,c01[{1}]*f1,f2)".format(c0[0],c0[1]))
  w.factory("SUM::F1(c10[{0}]*f0,c11[{1}]*f1,f2)".format(c1[0],c1[1]))
  
  # Check Model
  w.Print()
  w.writeToFile('workspace_DecomposingTestOfMixtureModelsClassifiers.root')
  #printFrame(w,'x',['f0','f1','f2']) 
  #printFrame(w,'x',['F0','F1'])

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

      np.savetxt('data/{0}/traindata_{1}_{2}.dat'.format(model_g,k,j),
              np.column_stack((traindata,targetdata)),fmt='%f')

      testdata, testtarget = makeDataset(x,w.pdf('f{0}'.format(k)),w.pdf('f{0}'.format(j))
      ,num_test)

      np.savetxt('data/{0}/testdata_{1}_{2}.dat'.format(model_g,k,j),
              np.column_stack((testdata,testtarget)),fmt='%f')


  traindata, targetdata = makeDataset(x,w.pdf('F0'),w.pdf('F1'),num_train)

  np.savetxt('data/{0}/traindata_F0_F1.dat'.format(model_g),
          np.column_stack((traindata,targetdata)),fmt='%f')

  testdata, testtarget = makeDataset(x,w.pdf('F0'.format(k)),w.pdf('F1'.format(j))
  ,num_test)

  np.savetxt('data/{0}/testdata_F0_F1.dat'.format(model_g),
          np.column_stack((testdata,testtarget)),fmt='%f')


def loadData(filename):
  traintarget = np.loadtxt(filename)
  traindata = traintarget[:,0]
  targetdata = traintarget[:,1]
  return (traindata, targetdata)

def predict(clf, traindata):
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
  np.savetxt('plots/{0}/{1}.txt'.format(model_g,label),np.column_stack((fpr,tpr)))
  plt.savefig('plots/{0}/{1}.png'.format(model_g,label))
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
      traindata,targetdata = loadData('data/{0}/traindata_{1}_{2}.dat'.format(model_g,k,j)) 

      print " Training Classifier on f{0}/f{1}".format(k,j)
      #clf = svm.NuSVC(probability=True) #Why use a SVR??
      clf.fit(traindata.reshape(traindata.shape[0],1)
          ,targetdata)
      joblib.dump(clf, 'model/{0}/adaptive_{1}_{2}.pkl'.format(model_g,k,j))


      #makeROC(outputs, testtarget, makePlotName('decomposed','trained',k,j,'roc'))
  
  traindata,targetdata = loadData('data/{0}/traindata_F0_F1.dat'.format(model_g)) 
  print " Training Classifier on F0/F1"
  #clf = svm.NuSVC(probability=True) #Why use a SVR??
  clf.fit(traindata.reshape(traindata.shape[0],1)
      ,targetdata)
  joblib.dump(clf, 'model/{0}/adaptive_F0_F1.pkl'.format(model_g))

  #testdata, testtarget = loadData('data/{0}/testdata_F0_F1.dat'.format(model_g)) 
  #outputs = predict(clf,testdata.reshape(testdata.shape[0],1))
  #makeROC(outputs, testtarget, makePlotName('full','trained',type='roc'))


def classifierPdf():
  ''' 
    Create pdfs for the classifier 
    score to be used later on the ratio 
    test
  '''

  bins = 150
  low = 0.
  high = 1.  

  f = ROOT.TFile('workspace_DecomposingTestOfMixtureModelsClassifiers.root')
  w = f.Get('w')
  f.Close()

  w.factory('score[{0},{1}]'.format(low,high))
  s = w.var('score')

  def saveHistos(w,ouputs,pos=None):
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
    
      #histpdf.specialIntegratorConfig(ROOT.kTRUE).method1D().setLabel('RooBinIntegrator')

      getattr(w,'import')(hist)
      getattr(w,'import')(data)
      getattr(w,'import')(datahist) # work around for morph = w.import(morph)
      getattr(w,'import')(histpdf) # work around for morph = w.import(morph)


      # Calculate the density of the classifier output using kernel density 
      # estimation technique
      #w.factory('KeysPdf::{0}dist_{1}_{2}(score,{0}data_{1}_{2})'.format(name,k,j))

      # Print histograms pdfs and estimated densities
      if verbose_printing == True and name == 'bkg':
        full = 'full' if pos == None else 'decomposed'
        printFrame(w,'score',[w.pdf('sighistpdf_{0}_{1}'.format(k,j)), w.pdf('bkghistpdf_{0}_{1}'.format(k,j))], makePlotName(full,'trained',k,j,type='hist'),['signal','bkg'])
        #printFrame(w,'score',['sigdist_{0}_{1}'.format(k,j),'bkgdist_{0}_{1}'.format(k,j)], makePlotName(full,'trained',k,j,'kernel'))

  for k,c in enumerate(c0):
    for j,c_ in enumerate(c1):
      traindata, targetdata = loadData('data/{0}/traindata_{1}_{2}.dat'.format(model_g,k,j))
      numtrain = traindata.shape[0]       

      clf = joblib.load('model/{0}/adaptive_{1}_{2}.pkl'.format(model_g,k,j))
      
      # Should I be using test data here?
      outputs = predict(clf,traindata.reshape(traindata.shape[0],1))
      #outputs = clf.predict_proba(traindata.reshape(traindata.shape[0],1)) 
      saveHistos(w,outputs,(k,j))

  traindata, targetdata = loadData('data/{0}/traindata_F0_F1.dat'.format(model_g))
  numtrain = traindata.shape[0]       

  clf = joblib.load('model/{0}/adaptive_F0_F1.pkl'.format(model_g))

  # Should I be using test data here?
  outputs = predict(clf,traindata.reshape(traindata.shape[0],1))
  #outputs = clf.predict_proba(traindata.reshape(traindata.shape[0],1)) 
  saveHistos(w,outputs)
     
  w.Print()

  w.writeToFile('workspace_DecomposingTestOfMixtureModelsClassifiers.root')


# this is not very efficient
def scikitlearnFunc(filename,x=0.):
  '''
    Needed for the scikit-learn wrapper
  '''
  clf = joblib.load(filename)
  traindata = np.array((x))
  outputs = predict(clf,traindata)[0]

  #if outputs[0] > 1:
  #  return 1.
  return outputs

class ScikitLearnCallback:
  def __init__(self,file):
    clf_ = joblib.load(file)

  def __call__(self,x = 0.):
    train = np.array((x))
    outputs = predict(clf_,train)[0]
    
    return outputs


def saveFig(x,y,file,labels=None,scatter=False):
  fig,ax = plt.subplots()
  if scatter == True:
    ax.scatter(x,y[0])
    ax.set_xlabel('score')
    ax.set_ylabel('regression')
  else:
    if len(y) == 1:
      ax.plot(x,y[0],'b')
    else:
      #Just supporting two plots for now
      ax.plot(x,y[0],'b-',label=labels[0]) 
      ax.plot(x,y[1],'r-',label=labels[1])
    ax.set_ylabel('LR')
    ax.set_xlabel('x')
  ax.set_title(file)
  if (len(y) > 1):
    ax.legend()
  if (len(y) > 1):
    # This breaks the naming convention for plots, I will solve
    # it later
    np.savetxt('plots/{0}/{1}_{2}.txt'.format(model_g,file,labels[0]),y[0])
    np.savetxt('plots/{0}/{1}_{2}.txt'.format(model_g,file,labels[1]),y[1])
  else:
    np.savetxt('plots/{0}/{1}.txt'.format(model_g,file),y[0])
  fig.savefig('plots/{0}/{1}.png'.format(model_g,file))
  plt.close(fig)
  plt.clf()

def fitAdaptive():
  '''
    Use the computed score densities to compute 
    the decompose ratio test
  '''
  ROOT.gSystem.Load('parametrized-learning/SciKitLearnWrapper/libSciKitLearnWrapper')
  ROOT.gROOT.ProcessLine('.L CompositeFunctionPdf.cxx+')


  f = ROOT.TFile('workspace_DecomposingTestOfMixtureModelsClassifiers.root')
  w = f.Get('w')
  f.Close()

  #x = w.var('x[-5,5]')
  x = ROOT.RooRealVar('x','x',0.2,0,5)
  getattr(w,'import')(ROOT.RooArgSet(x),ROOT.RooFit.RecycleConflictNodes()) 

  def constructDensity(w,pos = None):
    if pos <> None:
      k,j = pos
    else:
      k,j = ('F0','F1')
    test = scikitlearnFunc('model/{0}/adaptive_{1}_{2}.pkl'.format(model_g,k,j),2.0)
    nn = ROOT.SciKitLearnWrapper('nn_{0}_{1}'.format(k,j),'nn_{0}_{1}'.format(k,j),x)
    nn.RegisterCallBack(lambda x: scikitlearnFunc('model/{0}/adaptive_{1}_{2}.pkl'.format(model_g,k,j),x))
    getattr(w,'import')(ROOT.RooArgSet(nn),ROOT.RooFit.RecycleConflictNodes()) 
    

    # I should find the way to use this method
    #callbck = ScikitLearnCallback('model/{0}/adaptive_f{0}_f{1}.pkl'.format(model_g,k,j))
    #nn.RegisterCallBack(callbck)

    # Inserting the nn output into the pdf graph
    for l,name in enumerate(['sig','bkg']):
      w.factory('CompositeFunctionPdf::{0}template_{1}_{2}({0}histpdf_{1}_{2})'.
          format(name,k,j))
      w.factory('EDIT::{0}moddist_{1}_{2}({0}template_{1}_{2},score=nn_{1}_{2})'
              .format(name,k,j))
     

    if verbose_printing == True:
      full = 'full' if pos == None else 'decomposed'
      printFrame(w,'x',[w.pdf('sigmoddist_{0}_{1}'.format(k,j)),
                w.pdf('bkgmoddist_{0}_{1}'.format(k,j))],makePlotName(full,'trained',k,j,'dist'),['signal','bkg'])

  for k,c in enumerate(c0):
    for j,c_ in enumerate(c1):
      constructDensity(w,(k,j))
      #w.Print()
      # Save graphs
      #sigpdf.graphVizTree('sigpdfgraph.dot')
      #bkgpdf.graphVizTree('bkgpdfgraph.dot')
      
  constructDensity(w)

  # To calculate the ratio between single functions
  def singleRatio(x,f0,f1,val):
    x.setVal(val)
    if f0.getValV() < 10E-10:
      return 0.
    return f1.getValV() / f0.getValV()


  # To calculate the ratio between single functions
  def regFunc(x,f0,f1,val):
    x.setVal(val)
    if (f0.getValV() + f1.getValV()) < 10E-10:
      return 0.
    return f1.getValV() / (f0.getValV() + f1.getValV())



  # pair-wise ratios
  # and decomposition computation
  npoints = 100
  x = w.var('x')
  def evaluateDecomposedRatio(w,x,xarray,plotting=True, roc=False):
    npoints = xarray.shape[0]
    fullRatios = np.zeros(npoints)
    for k,c in enumerate(c0):
      innerRatios = np.zeros(npoints)
      if c == 0:
        continue
      for j,c_ in enumerate(c1):
        f0pdf = w.pdf('bkgmoddist_{0}_{1}'.format(k,j))
        f1pdf = w.pdf('sigmoddist_{0}_{1}'.format(k,j))
        f0 = w.pdf('f{0}'.format(k))
        f1 = w.pdf('f{0}'.format(j))
        pdfratios = [singleRatio(x,f0pdf,f1pdf,xs) for xs in xarray]
        # the cases in which both distributions are the same can be problematic
        # one will expect that the classifier gives same prob to both signal and bkg
        # but it can behave in weird ways, I will just avoid this for now 
        pdfratios = np.array(pdfratios) if k <> j else np.ones(npoints)
        innerRatios += (c_/c) * pdfratios
        ratios = [singleRatio(x,f0,f1,xs) for xs in xarray]
        if plotting == True:
          saveFig(xarray, [pdfratios,ratios], makePlotName('decomposed','trained',k,j,type='ratio'),
            ['trained','truth'])
        if roc == True:
          testdata, testtarget = loadData('data/{0}/testdata_{1}_{2}.dat'.format(model_g,k,j)) 
          clfRatios = [singleRatio(x,f0pdf,f1pdf,xs) for xs in testdata]
          trRatios = [singleRatio(x,f0,f1,xs) for xs in testdata]
          makeROC(np.array(trRatios), testtarget, makePlotName('decomposed','truth',k,j,type='roc'))
          makeROC(np.array(clfRatios), testtarget,makePlotName('decomposed','trained',k,j,type='roc'))
          
          # Scatter plot to compare regression function and classifier score
          reg = [regFunc(x,f0,f1,xs) for xs in testdata]
          clf = joblib.load('model/{0}/adaptive_{1}_{2}.pkl'.format(model_g,k,j))
          outputs = predict(clf,testdata.reshape(testdata.shape[0],1))
          saveFig(outputs,[reg], makePlotName('decomposed','trained',k,j,type='scatter'),scatter=True)

        #saveFig(xarray, ratios, makePlotName('decomposed','truth',k,j,type='ratio'))
      fullRatios += 1./innerRatios
    return fullRatios

  xarray = np.linspace(0,5,npoints)
   
  #testdata, testtarget = loadData('data/{0}/testdata_F0_F1.dat'.format(model_g)) 
  #xarray = np.sort(testdata)
 
  fullRatios = evaluateDecomposedRatio(w,x,xarray)

  saveFig(xarray, [fullRatios],  makePlotName('composite','trained',type='ratio')) 

  y2 = [singleRatio(x,w.pdf('F1'),w.pdf('F0'),xs) for xs in xarray]

  saveFig(xarray, [y2], makePlotName('full','truth',type='ratio'))
  saveFig(xarray, [np.array(y2) - fullRatios], makePlotName('composite','trained',type='diff'))

  # NN trained on complete model
  F0pdf = w.pdf('bkgmoddist_F0_F1')
  F1pdf = w.pdf('sigmoddist_F0_F1')
  pdfratios = [singleRatio(x,F1pdf,F0pdf,xs) for xs in xarray]
  pdfratios = np.array(pdfratios)
  saveFig(xarray, [pdfratios], makePlotName('full','trained',type='ratio'))
  saveFig(xarray, [np.array(y2) - pdfratios],makePlotName('full','trained',type='diff'))

  # ROC for ratios
  # load test data
  # check if ratios fulfill the requeriments of type
  testdata, testtarget = loadData('data/{0}/testdata_F0_F1.dat'.format(model_g)) 
  decomposedRatio = evaluateDecomposedRatio(w,x,testdata,plotting=False,roc=True)
  completeRatio = [singleRatio(x,F1pdf,F0pdf,xs) for xs in testdata]
  realRatio = [singleRatio(x,w.pdf('F1'),w.pdf('F0'),xs) for xs in testdata]

  makeROC(1.-np.array(realRatio), testtarget,makePlotName('full','truth',type='roc'))
  makeROC(1.-np.array(decomposedRatio), testtarget,makePlotName('composite','trained',type='roc'))
  makeROC(1.-np.array(completeRatio), testtarget,makePlotName('full','trained',type='roc'))

  #w.Print()

if __name__ == '__main__':
  classifiers = {'svc':svm.NuSVC(probability=True),'svr':svm.NuSVR(),
        'logistic': linear_model.LogisticRegression()}
  clf = None
  if (len(sys.argv) > 1):
    model_g = sys.argv[1]
    clf = classifiers.get(sys.argv[1])
  if clf == None:
    model_g = 'logistic'
    clf = classifiers['logistic']    
    print 'Not found classifier, Using logistic instead'

  # Set this value to False if only final plots are needed
  verbose_printing = False

  makeData(num_train=10000,num_test=3000) 
  trainClassifier(clf)
  classifierPdf()
  fitAdaptive()

