#!/usr/bin/env python
__author__ = "Pavez J. <juan.pavezs@alumnos.usm.cl>"


import ROOT
import numpy as np
from sklearn import svm, linear_model
from sklearn.externals import joblib

import sys

import os.path
import pdb

import matplotlib.pyplot as plt


''' 
 A simple example for the work on the section 
 5.4 of the paper 'Approximating generalized 
 likelihood ratio test with calibrated discriminative
 classifiers' by Kyle Cranmer
''' 

# Constants for each different model
c0 = [.1,.2, .7]
c1 = [.2,.5, .3]

def printFrame(w,obs,pdf):
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
    funcs.append(w.pdf(p))
    line_colors.append(ROOT.RooFit.LineColor(colors[i]))
  
  c1 = ROOT.TCanvas('c1')
  frame = x.frame()
  for i,f in enumerate(funcs):
      f.plotOn(frame,line_colors[i])
  frame.Draw()
  c1.SaveAs('plots/model{0}.pdf'.format('-'.join(pdf)))

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
   
  x = w.var('x')
  for k,c in enumerate(c0):
    for j,c_ in enumerate(c1):  
      traindata = np.zeros(num_train*2)
      targetdata = np.zeros(num_train*2)
      bkgpdf = w.pdf('f{0}'.format(k))
      sigpdf = w.pdf('f{0}'.format(j))
      bkgdata = bkgpdf.generate(ROOT.RooArgSet(x),num_train)
      sigdata = sigpdf.generate(ROOT.RooArgSet(x),num_train)
      
      traindata[:num_train] = [sigdata.get(i).getRealValue('x') 
          for i in range(num_train)]
      targetdata[:num_train].fill(1)

      traindata[num_train:] = [bkgdata.get(i).getRealValue('x')
          for i in range(num_train)]
      targetdata[num_train:].fill(0)
    
      np.savetxt('data/traindata_f{0}_f{1}.dat'.format(k,j),
              np.column_stack((traindata,targetdata)),fmt='%f')

  
def loadData(filename):
  traintarget = np.loadtxt(filename)
  traindata = traintarget[:,0]
  targetdata = traintarget[:,1]
  return (traindata, targetdata)


def trainClassifier():
  '''
    Train classifiers pair-wise on 
    datasets
  '''

  for k,c in enumerate(c0):
    for j,c_ in enumerate(c1):
      traindata,targetdata = loadData('data/traindata_f{0}_f{1}.dat'.format(k,j)) 

      print " Training SVM on f{0}/f{1}".format(k,j)
      #clf = svm.NuSVC(probability=True) #Why use a SVR??
      clf = linear_model.LogisticRegression()
      clf.fit(traindata.reshape(traindata.shape[0],1)
          ,targetdata)
      joblib.dump(clf, 'model/adaptive_f{0}_f{1}.pkl'.format(k,j))

def classifierPdf():
  ''' 
    Create pdfs for the classifier 
    score to be used later on the ratio 
    test
  '''

  bins = 30
  low = 0.
  high = 1.  

  f = ROOT.TFile('workspace_DecomposingTestOfMixtureModelsClassifiers.root')
  w = f.Get('w')
  f.Close()

  w.factory('score[{0},{1}]'.format(low,high))
  s = w.var('score')
  canvas = ROOT.TCanvas('c2','',400,400)

  for k,c in enumerate(c0):
    for j,c_ in enumerate(c1):
      traindata, targetdata = loadData('data/traindata_f{0}_f{1}.dat'.format(k,j))
      numtrain = traindata.shape[0]       

      clf = joblib.load('model/adaptive_f{0}_f{1}.pkl'.format(k,j))
      
      outputs = clf.predict_proba(traindata.reshape(traindata.shape[0],1)) 
      # Should I be using here test data?
         
      for l,name in enumerate(['sig','bkg']):
        data = ROOT.RooDataSet('{0}data_{1}_{2}'.format(name,k,j),"data",
              ROOT.RooArgSet(s))
        hist = ROOT.TH1F('{0}hist_{1}_{2}'.format(name,k,j),'hist',bins,low,high)
        # Check this use of data.add
        #[ (hist.Fill(val),data.add(val)) for val 
        #          in outputs[l*numtrain/2:(l+1)*numtrain/2] ]
        for val in outputs[l*numtrain/2:(l+1)*numtrain/2]:
          hist.Fill(val[1])
          s.setVal(val[1])
          data.add(ROOT.RooArgSet(s))

        hist.Draw()
        
        datahist = ROOT.RooDataHist('{0}datahist_{1}_{2}'.format(name,k,j),'hist',
              ROOT.RooArgList(s),hist)
        s.setBins(bins)
        histpdf = ROOT.RooHistPdf('{0}histpdf_{1}_{2}'.format(name,k,j),'hist',
              ROOT.RooArgSet(s), datahist, 1)
              
        histpdf.specialIntegratorConfig(ROOT.kTRUE).method1D().setLabel('RooBinIntegrator')

        getattr(w,'import')(data)
        getattr(w,'import')(datahist) # work around for morph = w.import(morph)
        getattr(w,'import')(histpdf) # work around for morph = w.import(morph)

        # Calculate the density of the classifier output using kernel density 
        # estimation technique
        w.factory('KeysPdf::{0}dist_{1}_{2}(score,{0}data_{1}_{2})'.format(name,k,j))

        printFrame(w,'score',['{0}histpdf_{1}_{2}'.format(name,k,j)])
        printFrame(w,'score',['{0}dist_{1}_{2}'.format(name,k,j)])

        canvas.SaveAs('plots/root_adaptive_hist_{0}_{1}_{2}.pdf'.format(name,k,j))     

  w.Print()

  w.writeToFile('workspace_DecomposingTestOfMixtureModelsClassifiers.root')

  
# this is not very efficient
def scikitlearnFunc(filename,x=0.):
  clf = joblib.load(filename)
  traindata = np.array((x))
  outputs = clf.predict_proba(traindata)[0][1]
  
  #if outputs[0] > 1:
  #  return 1.
  return outputs

class ScikitLearnCallback:
  def __init__(self,file):
    clf = joblib.load(file)

  def get(self,x = 0.):
    train = np.array((x))
    outputs = clf.predict_proba(train)[1]
    
    if outputs[0] > 1:
      return 1.
    return outputs[0]

def pdfRatio(w,x):
  sum = 0.
  w.var('x').setVal(x)
  for k,c in enumerate(c0):
    if (c==0):
      continue
    innerSum = 0.
    for j,c_ in enumerate(c1):
      # the cases in which both distributions are the same can be problematic
      # one will expect that the classifier gives same prob to both signal and bkg
      # but it can behave in weird ways, I will just avoid this for now
      if j == k:
        innerSum += c_/c
      else:
        # Just to avoid zero division, this need further checking
        if w.pdf('bkgmoddist_{0}_{1}'.format(k,j)).getValV() < 10E-10:
          continue
        innerSum += (c_/c *  
          (w.pdf('sigmoddist_{0}_{1}'.format(k,j)).getValV() /
             w.pdf('bkgmoddist_{0}_{1}'.format(k,j)).getValV()))
    if innerSum > 10E-10:
      sum += 1./innerSum
  return sum

def singlePdfRatio(w,f0,f1,x):
  w.var('x').setVal(x)
  if f0.getValV() < 10E-10:
    return 0.
  return f1.getValV() / f0.getValV()
  

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

  for k,c in enumerate(c0):
    for j,c_ in enumerate(c1):
      test = scikitlearnFunc('model/adaptive_f{0}_f{1}.pkl'.format(k,j),2.0)
      nn = ROOT.SciKitLearnWrapper('nn_{0}_{1}'.format(k,j),'nn_{0}_{1}'.format(k,j),x)
      nn.RegisterCallBack(lambda x: scikitlearnFunc('model/adaptive_f{0}_f{1}.pkl'.format(k,j),x))
      getattr(w,'import')(ROOT.RooArgSet(nn),ROOT.RooFit.RecycleConflictNodes()) 

      # I should find the way to use this method
      #callbck = ScikitLearnCallback('adaptive_f{0}_f{1}.pkl'.format(k,j))
      #nn.RegisterCallBack(lambda x: callbck.get(x))
      
      # Testing
      #canvas = ROOT.TCanvas('c1')
      #frame = x.frame()
      #nn.plotOn(frame)
      #frame.Draw()
      #canvas.SaveAs('classifierwrapper_f{0}_f{1}.pdf'.format(k,j))

      for l,name in enumerate(['sig','bkg']):
        w.factory('CompositeFunctionPdf::{0}template_{1}_{2}({0}dist_{1}_{2})'.
            format(name,k,j))
        w.factory('EDIT::{0}moddist_{1}_{2}({0}template_{1}_{2},score=nn_{1}_{2})'
                .format(name,k,j))
       

      sigpdf = w.pdf('sigmoddist_{0}_{1}'.format(k,j))
      bkgpdf = w.pdf('bkgmoddist_{0}_{1}'.format(k,j))

      canvas = ROOT.TCanvas('c1')
      frame = x.frame()
      bkgpdf.plotOn(frame)
      sigpdf.plotOn(frame,ROOT.RooFit.LineColor(ROOT.kRed))
      frame.Draw()
      canvas.SaveAs('plots/moddist_f{0}_f{1}.pdf'.format(k,j))

      #w.Print()

      #sigpdf.graphVizTree('sigpdfgraph.dot')
      #bkgpdf.graphVizTree('bkgpdfgraph.dot')
      
  def singleRatio(w,f0,f1,x):
    w.var('x').setVal(x)
    return f1.getValV() / f0.getValV()

  xarray = np.linspace(0,5,100)

  # pair-wise ratios
  for k,c in enumerate(c0):
    for j,c_ in enumerate(c1):
      f0pdf = w.pdf('bkgmoddist_{0}_{1}'.format(k,j))
      f1pdf = w.pdf('sigmoddist_{0}_{1}'.format(k,j))
      f0 = w.pdf('f{0}'.format(k))
      f1 = w.pdf('f{0}'.format(j))
      pdfratios = [singlePdfRatio(w,f0pdf,f1pdf,xs) for xs in xarray]
      ratios = [singleRatio(w,f0,f1,xs) for xs in xarray]
      plt.plot(xarray,pdfratios)
      plt.savefig('plots/pdf_ratio_{0}_{1}'.format(k,j))
      plt.clf()
      plt.plot(xarray,ratios)
      plt.savefig('plots/ratio_{0}_{1}'.format(k,j))
      plt.clf()

  # full ratios

  y = [pdfRatio(w,xs) for xs in xarray]

  plt.plot(xarray, y)
  plt.savefig('plots/ratio_classifier.png')

  def ratio(w,x):
    w.var('x').setVal(x)
    if w.pdf('F1').getValV() < 10E-10:
      return 0.
    return w.pdf('F0').getValV() / w.pdf('F1').getValV()

  y2 = [ratio(w,xs) for xs in xarray]

  plt.clf()
  plt.plot(xarray,y2)
  plt.savefig('plots/ratio.png')

  #w.Print()

makeData(num_train=100) 
trainClassifier()
classifierPdf()
fitAdaptive()

