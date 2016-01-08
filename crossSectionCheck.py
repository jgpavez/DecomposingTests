#!/usr/bin/env python
__author__ = "Pavez J. <juan.pavezs@alumnos.usm.cl>"

'''
  This code is used to build morhphed histograms 
  for single features.
'''


import ROOT
import numpy as np
from os import listdir
from os.path import isfile, join

import sys

import os.path
import pdb

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils import printMultiFrame, printFrame, saveFig, loadData,\
              makeROC, makeSigBkg, makePlotName, getWeights

from xgboost_wrapper import XGBoostClassifier
from pyMorphWrapper import MorphingWrapper


def findOutliers(x):
  q5, q95 = np.percentile(x, [5,95])  
  iqr = 2.0*(q95 - q5)
  outliers = (x <= q95 + iqr) & (x >= q5 - iqr)
  return outliers

def evalDist(x,f0,val):
  iter = x.createIterator()
  v = iter.Next()
  i = 0
  while v:
    v.setVal(val[i])
    v = iter.Next()
    i = i+1
  return f0.getVal(x)


def checkCrossSection(c1,cross_section,samples,target,dir,c1_g,model_g,feature=0,targetdata=None,samplesdata=None):
  '''
    Build morphed histograms for a feature
  '''

  w = ROOT.RooWorkspace('w')
  normalizer_abs = (np.abs(np.multiply(c1,cross_section))).sum()
  normalizer = (np.multiply(c1,cross_section)).sum()
  n_eff = normalizer / normalizer_abs
  print 'n_eff_ratio: {0}, n_tot: {0}'.format(n_eff,normalizer_abs)
  #normalizer = cross_section.sum()

  data_file = 'data'
  targetdata = targetdata[:2500,feature]
  fulldata = targetdata[:]
  targetdata = targetdata[targetdata <> -999.]
  targetdata = targetdata[findOutliers(targetdata)]

  bins = 300
  minimum = targetdata.min()
  maximum = targetdata.max()
  low = minimum - ((maximum - minimum) / bins)*10
  high = maximum + ((maximum - minimum) / bins)*10

  w.factory('score[{0},{1}]'.format(low,high))
  s = w.var('score')
  target_hist = ROOT.TH1F('targethist','targethist',bins,low,high)
  for val in targetdata:
    target_hist.Fill(val)
  norm = 1./target_hist.Integral()
  target_hist.Scale(norm) 

  # Creating samples histograms
  samples_hists = []
  sum_hist = ROOT.TH1F('sampleshistsum','sampleshistsum',bins,low,high)
  for i,sample in enumerate(samples):
    samples_hist = ROOT.TH1F('sampleshist{0}'.format(i),'sampleshist',bins,low,high)
    testdata = samplesdata[i]
    testdata = testdata[:2500,feature]
    testdata = testdata[testdata <> -999.]
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
  samples_datahist = ROOT.RooDataHist('{0}datahist'.format('samples'),'histsamples',
        ROOT.RooArgList(s),sum_hist)
  samples_histpdf = ROOT.RooHistFunc('{0}histpdf'.format('samples'),'histsamples',
        ROOT.RooArgSet(s), samples_datahist, 0)
  
  #printFrame(w,['score'],[target_histpdf,samples_histpdf],'check_cross_section_{0}'.format(feature),['real','weighted'],
  #  dir=dir, model_g=model_g,title='cross section check',x_text='x',y_text='dN')

   
  score = ROOT.RooArgSet(w.var('score'))
  # Now compute likelihood
  evalValues = np.array([evalDist(score,samples_histpdf,[xs]) for xs in fulldata])
  n_zeros = evalValues[evalValues <= 0.].shape[0]
  evalValues = evalValues[evalValues > 0.]
  print evalValues.shape
  likelihood = -np.log(evalValues).sum()
  print likelihood
  return likelihood,n_eff,n_zeros

def fullCrossSectionCheck(dir,c1_g,model_g,data_files,f1_dist,accept_list,c_min,c_max,npoints,n_eval):
  '''
    Likelihood of morphed distributions plots for all features
  '''

  csarray = np.linspace(c_min,c_max,npoints)

  # Loading morphed indexes, couplings and cross section
  all_indexes = np.loadtxt('2indexes_{0:.2f}_{1:.2f}_{2}.dat'.format(c_min,c_max,npoints)) 
  all_indexes = np.array([int(x) for x in all_indexes])
  #all_indexes = np.array([[int(x) for x in rows] for rows in all_indexes])
  all_couplings = np.loadtxt('2couplings_{0:.2f}_{1:.2f}_{2}.dat'.format(c_min,c_max,npoints)) 
  all_cross_sections = np.loadtxt('2crosssection_{0:.2f}_{1:.2f}_{2}.dat'.format(c_min,c_max,npoints)) 
  features = accept_list

  n_effs = np.zeros((len(features),all_couplings.shape[0]))
  n_zeros = np.zeros((len(features),all_couplings.shape[0]))
  likelihoods = []

  data_file='data'

  # Loading target and samples data
  targetdata = np.loadtxt('{0}/data/{1}/{2}/{3}_{4}.dat'.format(dir,'mlp',c1_g,data_file,f1_dist))
  basis_files = [data_files[i] for i in all_indexes]
  samplesdata = []
  data_file='data'
  for i,sample in enumerate(basis_files):
    samplesdata.append(np.loadtxt('{0}/data/{1}/{2}/{3}_{4}.dat'.format(dir,'mlp',c1_g,data_file,sample)))

  for k,couplings in enumerate(all_couplings):
    #samplesdata = []
    likelihoods.append([])
    for i,f in enumerate(features):
      likelihood,n_eff,n_zero = checkCrossSection(couplings,all_cross_sections[k],basis_files,f1_dist,
              dir,c1_g,model_g,feature=f,targetdata=targetdata,samplesdata=samplesdata)
      likelihoods[-1].append(likelihood)
      n_effs[i,k] = n_eff
      n_zeros[i,k] = n_zero
  print likelihoods

  likelihoods = np.array(likelihoods)
  likelihoods = likelihoods - np.abs(likelihoods.min(axis=0))
  likelihoods = likelihoods/likelihoods.max(axis=0)

  n_zeros_max = n_zeros.max(axis=1)
  n_zeros = n_zeros.transpose()/n_zeros.max(axis=1)
  n_zeros = n_zeros.transpose()

  for k,feat in enumerate(features):
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 12))
    ax1.plot(csarray, likelihoods[:,k])
    ax1.plot(csarray, n_effs[k])
    ax1.plot(csarray, n_zeros[k])
    plt.legend(['Likelihood','n_eff','n_zeros/{0}'.format(n_zeros_max[k])])
    fig.savefig('{0}/plots/{1}/{2}.png'.format(dir,model_g,'features_likelihood_g1_{0}'.format(feat)))
 

def CrossSectionCheck2D(dir,c1_g,model_g,data_files,f1_dist,accept_list,c_min,c_max,npoints,n_eval,feature):
  ''' 
    2D likelihood plots for a single feature
  '''

  # 2D version
  csarray = np.linspace(c_min[0],c_max[0],npoints)
  csarray2 = np.linspace(c_min[1], c_max[1], npoints)

  all_indexes = np.loadtxt('3indexes_{0:.2f}_{1:.2f}_{2:.2f}_{3:.2f}_{4}.dat'.format(c_min[0],c_min[1],c_max[0],c_max[1],npoints)) 
  all_indexes = np.array([int(x) for x in all_indexes])
  all_couplings = np.loadtxt('3couplings_{0:.2f}_{1:.2f}_{2:.2f}_{3:.2f}_{4}.dat'.format(c_min[0],c_min[1],c_max[0],c_max[1],npoints)) 
  all_cross_sections = np.loadtxt('3crosssection_{0:.2f}_{1:.2f}_{2:.2f}_{3:.2f}_{4}.dat'.format(c_min[0],c_min[1],c_max[0],c_max[1],npoints))

  basis_files = [data_files[i] for i in all_indexes]
  samplesdata = []
  data_file='data'
  for i,sample in enumerate(basis_files):
    samplesdata.append(np.loadtxt('{0}/data/{1}/{2}/{3}_{4}.dat'.format(dir,'mlp',c1_g,data_file,sample)))

  print all_indexes
  targetdata = np.loadtxt('{0}/data/{1}/{2}/{3}_{4}.dat'.format(dir,'mlp',c1_g,data_file,f1_dist))
 
  likelihoods = np.zeros((npoints,npoints))
  n_effs = np.zeros((npoints,npoints))
  n_zeros = np.zeros((npoints,npoints))

  for k,cs in enumerate(csarray):
    for j,cs2 in enumerate(csarray2):
      likelihood,n_eff,n_zero = checkCrossSection(all_couplings[k*npoints+j],all_cross_sections[k*npoints + j],basis_files,f1_dist,
              dir,c1_g,model_g,feature=feature,targetdata=targetdata,samplesdata=samplesdata)
      likelihoods[k,j] = likelihood
      n_effs[k,j] = n_eff
      n_zeros[k,j] = n_zero
  #print likelihoods
  saveFig(csarray,[csarray2,likelihoods],makePlotName('feature{0}'.format(25),'train',type='pixel_g1g2'),labels=['composed'],pixel=True,marker=True,dir=dir,model_g=model_g,marker_value=(1.0,0.5),print_pdf=True,contour=True,title='Feature for g1,g2')

