#!/usr/bin/env python
__author__ = "Pavez J. <juan.pavezs@alumnos.usm.cl>"


import ROOT
import numpy as np
from sklearn import svm, linear_model
from sklearn.externals import joblib

import sys

import os.path
import pdb

''' 
 A simple example for the work on the section 
 5.4 of the paper 'Approximating generalized 
 likelihood ratio test with calibrated discriminative
 classifiers' by Kyle Cranmer
''' 

# Constants for each different model
c0 = [.2,.8, 0.]
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
  c1.SaveAs('model{0}.pdf'.format('-'.join(pdf)))

def makeData(num_train=500,num_test=100):
  '''
  RooFit statistical model for the data
  
  '''  
  # Statistical model
  w = ROOT.RooWorkspace('w')
  w.factory("EXPR::f0('exp(x*-1)',x[-5,5])")
  w.factory("EXPR::f1('cos(x)**2 + 0.1',x)")
  w.factory("EXPR::f2('exp(-(x-2)**2/2)',x)")
  w.factory("SUM::F0(c00[{0}]*f0,c01[{1}]*f1,f2)".format(c0[0],c0[1]))
  w.factory("SUM::F1(c10[{0}]*f0,c11[{1}]*f1,f2)".format(c1[0],c1[1]))
  
  # Check Model
  w.Print()
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
    
      np.savetxt('traindata_f{0}_f{1}.dat'.format(k,j),
              np.column_stack((traindata,targetdata)),fmt='%f')

  
def trainClassifier():
  '''
    Train classifiers pair-wise on 
    datasets
  '''

  for k,c in enumerate(c0):
    for j,c_ in enumerate(c1):
      traintarget = np.loadtxt('traindata_f{0}_f{1}.dat'.format(k,j))
      traindata = traintarget[:,0]
      targetdata = traintarget[:,1]
      
      print " Training SVM on f{0}/f{1}".format(k,j)
      clf = svm.NuSVR() #Why use a SVR??
      clf.fit(traindata.reshape(traindata.shape[0],1)
          ,targetdata)
      joblib.dump(clf, 'adaptive_f{0}_f{1}.pkl'.format(k,j))


makeData(num_train=250) 
trainClassifier()
