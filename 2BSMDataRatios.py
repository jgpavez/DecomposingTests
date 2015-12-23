#!/usr/bin/env python
__author__ = "Pavez J. <juan.pavezs@alumnos.usm.cl>"


import ROOT
import numpy as np
from sklearn import svm, linear_model
from sklearn.externals import joblib
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import GradientBoostingClassifier

from os import listdir
from os.path import isfile, join

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
from pyMorphWrapper import MorphingWrapper

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

def findOutliers(x):
  q5, q95 = np.percentile(x, [5,95])  
  iqr = 2.0*(q95 - q5)
  outliers = (x <= q95 + iqr) & (x >= q5 - iqr)
  return outliers


def checkCrossSection(c1,cross_section,samples,target,dir,c1_g,model_g,feature=0,testdata=None,samplesdata=None):
  w = ROOT.RooWorkspace('w')
  normalizer_abs = (np.abs(np.multiply(c1,cross_section))).sum()
  normalizer = (np.multiply(c1,cross_section)).sum()
  n_eff = normalizer / normalizer_abs
  print 'n_eff_ratio: {0}, n_tot: {0}'.format(n_eff,normalizer_abs)
  #normalizer = cross_section.sum()

  # load S(1,1.5) data
  data_file = 'data'
  testdata = testdata[:2500,feature]
  fulldata = testdata[:]
  testdata = testdata[testdata <> -999.]
  testdata = testdata[findOutliers(testdata)]
  bins = 300
  minimum = testdata.min()
  maximum = testdata.max()
  low = minimum - ((maximum - minimum) / bins)*10
  high = maximum + ((maximum - minimum) / bins)*10

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
    testdata = samplesdata[i]
    testdata = testdata[:2500,feature]
    testdata = testdata[testdata <> -999.]
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
  
  #printFrame(w,['score'],[target_histpdf,samples_histpdf],'check_cross_section_{0}'.format(feature),['real','weighted'],
  #  dir=dir, model_g=model_g,title='cross section check',x_text='x',y_text='dN')

   
  score = ROOT.RooArgSet(w.var('score'))
  # Now compute likelihood
  evalValues = np.array([evalDist(score,samples_histpdf,[xs]) for xs in fulldata])
  n_zeros = evalValues[evalValues <= 0.].shape[0]
  evalValues = evalValues[evalValues > 0.]
  print evalValues.shape
  #likelihood = -(1./evalValues.shape[0])*np.log(evalValues).sum()
  likelihood = -np.log(evalValues).sum()
  print likelihood
  return likelihood,n_eff,n_zeros

def fullCrossSectionCheck(dir,c1_g,model_g,data_files,f1_dist):

  npoints = 10
  c_eval = 2
  #c_min = [0.6,0.1]
  #c_max = [1.5,0.9]
  c_min = 0.1
  c_max = 0.9

  csarray = np.linspace(c_min,c_max,npoints)

  all_indexes = np.loadtxt('05indexes_{0:.2f}_{1:.2f}_{2}.dat'.format(c_min,c_max,npoints)) 
  all_indexes = np.array([int(x) for x in all_indexes])
  #all_indexes = np.array([[int(x) for x in rows] for rows in all_indexes])
  all_couplings = np.loadtxt('05couplings_{0:.2f}_{1:.2f}_{2}.dat'.format(c_min,c_max,npoints)) 
  all_cross_sections = np.loadtxt('05crosssection_{0:.2f}_{1:.2f}_{2}.dat'.format(c_min,c_max,npoints)) 
  features = [i for i in range(51) if i not in [32,39,50]]
  features = features

  n_effs = np.zeros((len(features),all_couplings.shape[0]))
  n_zeros = np.zeros((len(features),all_couplings.shape[0]))
  likelihoods = []

  data_file='data'

  testdata = np.loadtxt('{0}/data/{1}/{2}/{3}_{4}.dat'.format(dir,'mlp',c1_g,data_file,f1_dist))
  basis_files = [data_files[i] for i in all_indexes]
  samplesdata = []
  data_file='data'
  for i,sample in enumerate(basis_files):
    samplesdata.append(np.loadtxt('{0}/data/{1}/{2}/{3}_{4}.dat'.format(dir,'mlp',c1_g,data_file,sample)))



  '''
  c_min = [0.6,0.1]
  c_max = [1.5,0.9]
  npoints = 15
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
  testdata = np.loadtxt('{0}/data/{1}/{2}/{3}_{4}.dat'.format(dir,'mlp',c1_g,data_file,f1_dist))
  #basis_files = [data_files[i] for i in all_indexes]
  #for i,sample in enumerate(basis_files):
  #  samplesdata.append(np.loadtxt('{0}/data/{1}/{2}/{3}_{4}.dat'.format(dir,'mlp',c1_g,data_file,sample)))
 
  feature = 25
  likelihoods = np.zeros((npoints,npoints))
  n_effs = np.zeros((npoints,npoints))
  n_zeros = np.zeros((npoints,npoints))

  for k,cs in enumerate(csarray):
    for j,cs2 in enumerate(csarray2):
      likelihood,n_eff,n_zero = checkCrossSection(all_couplings[k*npoints+j],all_cross_sections[k*npoints + j],basis_files,f1_dist,
              dir,c1_g,model_g,feature=feature,testdata=testdata,samplesdata=samplesdata)
      likelihoods[k,j] = likelihood
      n_effs[k,j] = n_eff
      n_zeros[k,j] = n_zero
  #print likelihoods
  saveFig(csarray,[csarray2,likelihoods],makePlotName('feature{0}'.format(25),'train',type='pixel_g1g2'),labels=['composed'],pixel=True,marker=True,dir=dir,model_g=model_g,marker_value=(1.0,0.5),print_pdf=True,contour=True,title='Feature for g1,g2')
  '''

  for k,couplings in enumerate(all_couplings):
    #samplesdata = []
    likelihoods.append([])
    #basis_files = [data_files[i] for i in all_indexes[k]]
    #print all_indexes[k]
    #for i,sample in enumerate(basis_files):
    #  samplesdata.append(np.loadtxt('{0}/data/{1}/{2}/{3}_{4}.dat'.format(dir,'mlp',c1_g,data_file,sample)))
    for i,f in enumerate(features):
      likelihood,n_eff,n_zero = checkCrossSection(couplings,all_cross_sections[k],basis_files,f1_dist,
              dir,c1_g,model_g,feature=f,testdata=testdata,samplesdata=samplesdata)
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

  #fig, ax1 = plt.subplots(1, 1, figsize=(8, 12), subplot_kw={'projection': '3d'})

  for k,feat in enumerate(features):
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 12))
    ax1.plot(csarray, likelihoods[:,k])
    ax1.plot(csarray, n_effs[k])
    ax1.plot(csarray, n_zeros[k])
    plt.legend(['Likelihood','n_eff','n_zeros/{0}'.format(n_zeros_max[k])])
    fig.savefig('{0}/plots/{1}/{2}.png'.format(dir,model_g,'features_likelihood_g2_{0}'.format(feat)))
 
  '''
  #ax1.view_init(30,90)
  X,Y = np.meshgrid(csarray,features)
  ax1.plot_wireframe(X, Y, likelihoods.transpose(), rstride=1, cstride=10)
  ax1.set_title("Row stride 0")
  plt.tight_layout()
  ax1.set_xlabel('Khzz values')
  ax1.set_ylabel('Feature')
  #ax1.plot_wireframe(X, Y, likelihoods, rstride=10, cstride=0)
  #ax1.set_title("Column stride 0")
  '''
  #fig.savefig('{0}/plots/{1}/{2}.png'.format(dir,model_g,'features_likelihood.png'))
  pdb.set_trace()


if __name__ == '__main__':
  # Setting the classifier to use
  model_g = None
  classifiers = {'svc':svm.NuSVC(probability=True),'svr':svm.NuSVR(),
        'logistic': linear_model.LogisticRegression(), 
        'bdt':GradientBoostingClassifier(n_estimators=300, learning_rate=0.1,
        max_depth=4, random_state=0),
        'mlp':'',
        'xgboost': XGBoostClassifier(missing=-999.,num_class=2, nthread=4, silent=0,
          num_boost_round=50, eta=0.5, max_depth=11)}
  clf = None
  if (len(sys.argv) > 1):
    model_g = sys.argv[1]
    clf = classifiers.get(sys.argv[1])
  if clf == None:
    model_g = 'logistic'
    clf = classifiers['logistic']    
    print 'Not found classifier, Using logistic instead'

  # couplings and data files
  verbose_printing = True
  dir = '/afs/cern.ch/user/j/jpavezse/systematics'
  workspace_file = 'workspace_2BSMDataRatios.root'
  
  mypath = dir + '/data/mlp/2BSM'
  data_files = [f[5:-4] for f in listdir(mypath) if isfile(join(mypath, f)) and f.startswith('data')]
  print data_files

  def processString(file_str):
    file_str = file_str.split('_') 
    res = []
    for s_str in file_str[1:]:
      neg = 1.
      if s_str[0] == 'm':
        s_str = s_str[1:]
        neg = -1.
      if 'ov' in s_str:
        nums = s_str.split('ov')    
        res.append(neg * (float(nums[0])/float(nums[1])))
      else:
        res.append(neg * float(s_str))
    return res

  all_couplings = [processString(f) for f in data_files]
  basis_files = data_files[1:]
  basis = all_couplings[1:]
  f1_dist = data_files[0]
  f0_dist = data_files[3]#Using 1_1_0
  morphed = all_couplings[0]
  c1_g = '2BSM'

  #morph = MorphingWrapper()    
  #morph.setSampleData(nsamples=15,ncouplings=3,types=['S','S','S'],morphed=morphed,samples=all_couplings)
  #basis_indexes = morph.dynamicMorphing()
  #basis_files = [data_files[i] for i in basis_indexes]
  #basis = [all_couplings[i] for i in basis_indexes] 
  #couplings = np.array(morph.getWeights())
  #cross_section = np.array(morph.getCrossSections())
  #pdb.set_trace()  

  #basis_indexes = [ 1,  4,  5,  6,  8,  9, 10, 11, 14, 15, 17, 19, 21, 23, 24]
  basis_indexes = [ 1,  2,  3,  7, 10, 11, 12, 13, 15, 16, 18, 20, 22, 24, 25]
  basis_samples = [all_couplings[i] for i in basis_indexes]
  morph = MorphingWrapper()    
  morph.setSampleData(nsamples=15,ncouplings=3,types=['S','S','S'],morphed=morphed,samples=basis_samples)
  basis_files = [data_files[i] for i in basis_indexes]
  basis = [all_couplings[i] for i in basis_indexes] 
  couplings = np.array(morph.getWeights())
  cross_section = np.array(morph.getCrossSections())
  #pdb.set_trace()  

  '''
  basis_indexes = [15, 19, 6, 21, 20, 24, 15, 17, 23, 11, 12, 22, 16, 5, 9]
  basis_files = [basis_files[i] for i in basis_indexes]
  basis = [basis[i] for i in basis_indexes] 
  couplings = np.array([  8.0625    ,   0.04166675,  10.8125    ,   0.7782737 ,
        35.98204041,   7.71840429,   5.74050045,   5.25005674,
        19.89099693,  20.37686539,  10.0357542 ,  21.91056824,
         0.65301228,   3.71593142,  19.95527649])
  cross_section = np.array([  0.72753178,   0.17189144,   0.52275165,   3.90910227,
         2.50304355,   3.68695864,   0.72753178,   0.21662377,
        10.33017995,  58.08554602,  35.75995451,   3.90910227,
         0.36000962,   1.13599184,  29.05239118])
  '''

  #bkg_morphed = all_couplings[3]
  #morph = MorphingWrapper()    
  #morph.setSampleData(nsamples=15,ncouplings=3,types=['S','S','S'],morphed=bkg_morphed,samples=basis)
  #bkg_couplings = np.array(morph.getWeights())
  #bkg_cross_section = np.array(morph.getCrossSections())
  #pdb.set_trace()  
  ''' 

  # Considering background as 1,1,0
  c0 = np.array([  5.        ,  -0.33333349,  -4.39285707,  47.50322723,
         5.39236307,   6.67031336,   0.46575356,  -0.34352303,
         4.70266438,   4.09168863,   7.60000801,   6.89566898,
         3.80555463,   2.91366005,  10.47156906])  
  c0_cross_section = np.array([  0.72753178,   0.19756027,   1.13599184,   2.50304355,
         0.17189144,  10.33017995,   0.72753178,   0.36000962,
         3.90910227,  35.75995451,  58.08554602,   3.90910227,
         0.72753178,   5.05110374,   0.35444307])
  c0 = np.multiply(c0,c0_cross_section)

  c1 = couplings
  c0 = np.array([1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.])
  f0_dist = basis_files[0]
  print basis_files
  print f1_dist
  print f0_dist
  '''
  c1 = couplings
  c0 = np.ones(couplings.shape[0])
  #c0 = np.array([1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.])
  print f1_dist
  print f0_dist
  print c1
  print c0

  # features
  vars_g = ["Z1_E","Z1_pt","Z1_eta","Z1_phi","Z1_m","Z2_E","Z2_pt","Z2_eta","Z2_phi","Z2_m","higgs_E","higgs_pt","higgs_eta","higgs_phi","higgs_m","DelPhi_Hjj","mH","pT_Hjj","DelEta_jj","EtaProd_jj","DelY_jj","DelPhi_jj","DelR_jj","Mjj","Mjets","njets","jet1_E","jet1_eta","jet1_y","jet1_phi","jet1_pt","jet1_m","jet1_isPU","jet2_E","jet2_phi","jet2_eta","jet2_y","jet2_pt","jet2_m","jet2_isPU","DelPt_jj","minDelR_jZ","DelPt_ZZ","Zeppetaj3","ZeppetaZZ","jet3_E","jet3_eta","jet3_phi","jet3_pt","jet3_m","jet3_isPU"]
  null_list = [15,17,18,19,20,21,22,23,32,33,34,35,37,38,39,41,43,44,45,46,47,48,49,50]
  accept_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,24,25,26,27,28,29,30,31,36,40,42]
  #vars_g = [vars_g[i] for i,_ in enumerate(vars_g) if i in accept_list]

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
  set_size = np.zeros(len(data_files))

  #fullCrossSectionCheck(dir,c1_g,model_g,data_files,f1_dist)
  #pdb.set_trace()

  train_files = data_files
  train_n = len(train_files)
  scaler = None
  # train the pairwise classifiers
  #scaler = trainClassifiers(clf,train_n,dir=dir, model_g=model_g,
  #    c1_g=c1_g ,model_file='model',data_file='data',dataset_names=train_files,
  #    preprocessing=False,
  #    seed=random_seed, full_names=[f0_dist,f1_dist],vars_names=vars_g)
  #pdb.set_trace()

  # class which implement the decomposed method
  test = DecomposedTest(c0,c1,dir=dir,c1_g=c1_g,model_g=model_g,
          input_workspace=workspace_file, verbose_printing = verbose_printing,
          model_file='model',preprocessing=False,scaler=scaler, dataset_names=data_files,
          seed=random_seed, F1_dist=f1_dist,F0_dist=f0_dist, cross_section=cross_section,
          basis_indexes=basis_indexes,F1_couplings=morphed,all_couplings=all_couplings)
  #test.fit(data_file='data',importance_sampling=False, true_dist=False,vars_g=vars_g)
  #test.computeRatios(data_file='data',true_dist=False,vars_g=vars_g,use_log=False) 
  #pdb.set_trace()

  n_hist = 1050
  # compute likelihood for c0[0] and c0[1] values
  test.fitCValues(c0,c1,data_file='data', true_dist=False,vars_g=vars_g,use_log=False,
            n_hist=n_hist, num_pseudodata=5000,weights_func=getWeights)

  #plotCValues(c0,c1,dir=dir,c1_g=c1_g,model_g=model_g,true_dist=False,vars_g=vars_g,
  #    workspace=workspace_file,use_log=False,n_hist=n_hist,c_eval=1,range_min=1.1,
  #    range_max=1.9)


