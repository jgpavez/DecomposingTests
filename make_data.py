#!/usr/bin/env python
__author__ = "Pavez J. <juan.pavezs@alumnos.usm.cl>"

import ROOT
import numpy as np

import sys

import os.path
import pdb

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''
  Functions to make model and data for the decomposed training 
  method
'''

# Default model parameters
# private coefficients
coeffs_g = [[ 0.28174199,0.46707738,0.25118062],[0.18294893,0.33386682,0.48318425],[ 0.25763285,0.28015834,0.46220881]]

# gaussians parameters
mu_g = []
cov_g = []
mu_g.append([5.,5.,4.,3.,5.,5.,4.5,2.5,4.,3.5])
mu_g.append([2.,4.5,0.6,5.,6.,4.5,4.2,0.2,4.1,3.3])
mu_g.append([1.,0.5,0.3,0.5,0.6,0.4,0.1,0.2,0.1,0.3])

cov_g.append([[3.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
         [0.,2.,0.,0.,0.,0.,0.,0.,0.,0.],
         [0.,0.,14.,0.,0.,0.,0.,0.,0.,0.],
         [0.,0.,0.,6.,0.,0.,0.,0.,0.,0.],
         [0.,0.,0.,0.,17.,0.,0.,0.,0.,0.],
         [0.,0.,0.,0.,0.,10.,0.,0.,0.,0.],
         [0.,0.,0.,0.,0.,0.,5.,0.,0.,0.],
         [0.,0.,0.,0.,0.,0.,0.,1.3,0.,0.],
         [0.,0.,0.,0.,0.,0.,0.,0.,1.,0.],
         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.3]])
cov_g.append([[3.5,0.,0.,0.,0.,0.,0.,0.,0.,0.],
         [0.,3.5,0.,0.,0.,0.,0.,0.,0.,0.],
         [0.,0.,3.5,0.,0.,0.,0.,0.,0.,0.],
         [0.,0.,0.,7.2,0.,0.,0.,0.,0.,0.],
         [0.,0.,0.,0.,4.5,0.,0.,0.,0.,0.],
         [0.,0.,0.,0.,0.,3.5,0.,0.,0.,0.],
         [0.,0.,0.,0.,0.,0.,8.2,0.,0.,0.],
         [0.,0.,0.,0.,0.,0.,0.,9.5,0.,0.],
         [0.,0.,0.,0.,0.,0.,0.,0.,0.5,0.],
         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.5]])
cov_g.append([[13.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
         [0.,12.,0.,0.,0.,0.,0.,0.,0.,0.],
         [0.,0.,14.,0.,0.,0.,0.,0.,0.,0.],
         [0.,0.,0.,6.,0.,0.,0.,0.,0.,0.],
         [0.,0.,0.,0.,1.,0.,0.,0.,0.,0.],
         [0.,0.,0.,0.,0.,10.,0.,0.,0.,0.],
         [0.,0.,0.,0.,0.,0.,15.,0.,0.,0.],
         [0.,0.,0.,0.,0.,0.,0.,6.3,0.,0.],
         [0.,0.,0.,0.,0.,0.,0.,0.,11.,0.],
         [0.,0.,0.,0.,0.,0.,0.,0.,0.,1.3]])


def makeModelPrivateND(vars_g,c0, c1, n_private=3, coeffs=coeffs_g,cov_l=cov_g, mu_l=mu_g, 
    workspace='workspace_DecomposingTestOfMixtureModelsClassifiers.root', 
    dir='/afs/cern.ch/user/j/jpavezse/systematics', 
    verbose_printing=False):
  '''
  RooFit statistical model for the data
  
  '''  
  # Statistical model
  w = ROOT.RooWorkspace('w')

  print 'Generating initial distributions'
  cov_m = []
  mu_m = []
  mu_str = []
  cov_root = []
  vec = []
  argus = ROOT.RooArgList() 

  # features
  for i,var in enumerate(vars_g):
    w.factory('{0}[{1},{2}]'.format(var,-25,30))
    argus.add(w.var(var))

  for glob in range(3):
    for priv in range(n_private):
      # generate covriance matrix
      cov_m.append(np.matrix(cov_l[glob]))
      cov_root.append(ROOT.TMatrixDSym(len(vars_g)))
      for i,var1 in enumerate(vars_g):
        for j,var2 in enumerate(vars_g):
          cov_root[-1][i][j] = cov_m[-1][i,j]
      getattr(w,'import')(cov_root[-1],'cov{0}'.format(glob*3 + priv))
      # generate mu vectors
      mu_m.append(np.array(mu_l[glob]) + meansum[glob][priv])
      vec.append(ROOT.TVectorD(len(vars_g)))
      for i, mu in enumerate(mu_m[-1]):
        vec[-1][i] = mu
      mu_str.append(','.join([str(mu) for mu in mu_m[-1]]))
      # create multivariate gaussian
      gaussian = ROOT.RooMultiVarGaussian('f{0}_{1}'.format(glob,priv),
            'f{0}_{1}'.format(glob,priv),argus,vec[-1],cov_root[-1])
      getattr(w,'import')(gaussian)
    # create private mixture model
    priv_coeffs = np.array(coeffs[glob])
    #print 'priv coef {0} {1}'.format(priv_coeffs, priv_coeffs.sum())
    sum_str = ','.join(['c_{0}_{1}[{2}]*f{0}_{1}'.format(glob,j,priv_coeffs[j]) for j in range(n_private)])
    w.factory('SUM::f{0}({1})'.format(glob,sum_str))
  #mixture model  
  w.factory("SUM::F0(c00[{0}]*f0,c01[{1}]*f1,f2)".format(c0[0],c0[1]))
  w.factory("SUM::F1(c10[{0}]*f0,c11[{1}]*f1,f2)".format(c1[0],c1[1]))
  
  # Check Model
  w.Print()

  w.writeToFile('{0}/{1}'.format(dir,workspace))
  if verbose_printing == True:
    printFrame(w,vars_g,[w.pdf('f0'),w.pdf('f1'),w.pdf('f2')],'decomposed_model',['f0','f1','f2']) 
    printFrame(w,vars_g,[w.pdf('F0'),w.pdf('F1')],'full_model',['F0','F1'])
    printFrame(w,vars_g,[w.pdf('F1'),'f0'],'full_signal', ['F1','f0'])

  return w

def makeModelND(vars_g,c0,c1,cov_l=cov_g,mu_l=mu_g,
    workspace='workspace_DecomposingTestOfMixtureModelsClassifiers.root', 
    dir='/afs/cern.ch/user/j/jpavezse/systematics',
    verbose_printing=False):
  '''
  RooFit statistical model for the data
  
  '''  
  # Statistical model
  w = ROOT.RooWorkspace('w')

  print 'Generating initial distributions'
  cov_m = []
  mu_m = []
  mu_str = []
  cov_root = []
  vec = []
  argus = ROOT.RooArgList() 
  #features
  for i,var in enumerate(vars_g):
    w.factory('{0}[{1},{2}]'.format(var,-25,30))
    argus.add(w.var(var))

  for glob in range(3):
    # generate covariance matrix
    cov_m.append(np.matrix(cov_l[glob]))
    cov_root.append(ROOT.TMatrixDSym(len(vars_g)))
    for i,var1 in enumerate(vars_g):
      for j,var2 in enumerate(vars_g):
        cov_root[-1][i][j] = cov_m[-1][i,j]
    getattr(w,'import')(cov_root[-1],'cov{0}'.format(glob))
    # generate mu vector
    mu_m.append(np.array(mu_l[glob]))
    vec.append(ROOT.TVectorD(len(vars_g)))
    for i, mu in enumerate(mu_m[-1]):
      vec[-1][i] = mu
    mu_str.append(','.join([str(mu) for mu in mu_m[-1]]))
    # multivariate gaussian
    gaussian = ROOT.RooMultiVarGaussian('f{0}'.format(glob),
          'f{0}'.format(glob),argus,vec[-1],cov_root[-1])
    getattr(w,'import')(gaussian)
  # mixture models
  w.factory("SUM::F0(c00[{0}]*f0,c01[{1}]*f1,f2)".format(c0[0],c0[1]))
  w.factory("SUM::F1(c10[{0}]*f0,c11[{1}]*f1,f2)".format(c1[0],c1[1]))
  
  # Check Model
  w.Print()

  w.writeToFile('{0}/{1}'.format(dir,workspace))
  if verbose_printing == True:
    printFrame(w,vars_g,[w.pdf('f0'),w.pdf('f1'),w.pdf('f2')],'decomposed_model',['f0','f1','f2']) 
    printFrame(w,vars_g,[w.pdf('F0'),w.pdf('F1')],'full_model',['F0','F1'])
    printFrame(w,vars_g,[w.pdf('F1'),'f0'],'full_signal', ['F1','f0'])

  return w

def makeData(vars_g,c0,c1, num_train=500,num_test=100,no_train=False,
  workspace='workspace_DecomposingTestOfMixtureModelsClassifiers.root',
  dir='/afs/cern.ch/user/j/jpavezse/systematics',
  c1_g='',model_g='mlp'):
  # Start generating data
  ''' 
    Each function will be discriminated pair-wise
    so n*n datasets are needed (maybe they can be reused?)
  ''' 

  f = ROOT.TFile('{0}/{1}'.format(dir,workspace))
  w = f.Get('w')
  f.Close()

  print 'Making Data'
  # Start generating data
  ''' 
    Each function will be discriminated pair-wise
    so n*n datasets are needed (maybe they can be reused?)
  ''' 
   
  # make data from root pdf
  def makeDataFi(x, pdf, num):
    traindata = np.zeros((num,len(vars_g))) 
    data = pdf.generate(x,num)
    traindata[:] = [[data.get(i).getRealValue(var) for var in vars_g]
        for i in range(num)]
    return traindata
  
  # features
  vars = ROOT.TList()
  for var in vars_g:
    vars.Add(w.var(var))
  x = ROOT.RooArgSet(vars)

  # make data from pdf and save to .dat in folder 
  # ./data/{model}/{c1}
  for k,c in enumerate(c0):
    print 'Making {0}'.format(k)
    if not no_train:
      traindata = makeDataFi(x,w.pdf('f{0}'.format(k)), num_train)
      np.savetxt('{0}/data/{1}/{2}/train_{3}.dat'.format(dir,'mlp',c1_g,k),
                        traindata,fmt='%f')
    testdata = makeDataFi(x, w.pdf('f{0}'.format(k)), num_test)
    np.savetxt('{0}/data/{1}/{2}/test_{3}.dat'.format(dir,'mlp',c1_g,k),
                      testdata,fmt='%f')
  if not no_train:
    traindata = makeDataFi(x,w.pdf('F0'), num_train)
    np.savetxt('{0}/data/{1}/{2}/train_F0.dat'.format(dir,'mlp',c1_g),
                      traindata,fmt='%f')
    traindata = makeDataFi(x,w.pdf('F1'), num_train)
    np.savetxt('{0}/data/{1}/{2}/train_F1.dat'.format(dir,'mlp',c1_g),
                      traindata,fmt='%f')
  testdata = makeDataFi(x, w.pdf('F0'), num_test)
  np.savetxt('{0}/data/{1}/{2}/test_F0.dat'.format(dir,'mlp',c1_g),
                    testdata,fmt='%f')
  testdata = makeDataFi(x, w.pdf('F1'), num_test)
  np.savetxt('{0}/data/{1}/{2}/test_F1.dat'.format(dir,'mlp',c1_g),
                    testdata,fmt='%f')


