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

from crossSectionCheck import fullCrossSectionCheck 

if __name__ == '__main__':
  # Setting the classifier to use
  model_g = None
  classifiers = {'svc':svm.NuSVC(probability=True),'svr':svm.NuSVR(),
        'logistic': linear_model.LogisticRegression(), 
        'bdt':GradientBoostingClassifier(n_estimators=300, learning_rate=0.1,
        max_depth=4, random_state=0),
        'mlp':'',
        'xgboost': XGBoostClassifier(missing=-999.,num_class=2, nthread=4, silent=0,
          num_boost_round=50, eta=0.5, max_depth=5)}
  clf = None
  if (len(sys.argv) > 1):
    model_g = sys.argv[1]
    clf = classifiers.get(sys.argv[1])
  if clf == None:
    model_g = 'logistic'
    clf = classifiers['logistic']    
    print 'Not found classifier, Using logistic instead'

  c1_g = '2BSM'

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
  # F1 distribution
  f1_idx = 19
  all_couplings = [processString(f) for f in data_files]
  morphed = all_couplings[f1_idx]
  f1_dist = data_files[f1_idx]
  basis_files = [data_files[i] for i,_ in enumerate(data_files) if i <> f1_idx]
  basis = [all_couplings[i] for i,_ in enumerate(all_couplings) if i <> f1_idx]
  # F0 distribution
  f0_dist = data_files[3]#Using 1_1_0

  # This is usefull only if ploting full ratio histograms
  basis_indexes = [ 1,  2,  3,  7, 10, 11, 12, 13, 15, 16, 18, 20, 22, 24, 25]
  basis_samples = [all_couplings[i] for i in basis_indexes]
  morph = MorphingWrapper()    
  morph.setSampleData(nsamples=15,ncouplings=3,types=['S','S','S'],morphed=morphed,samples=basis_samples)
  basis_files = [data_files[i] for i in basis_indexes]
  basis = [all_couplings[i] for i in basis_indexes] 
  couplings = np.array(morph.getWeights())
  cross_section = np.array(morph.getCrossSections())

  c1 = couplings
  c0 = np.zeros(couplings.shape[0])
  c0[0] = 1.
  print f1_dist
  print f0_dist
  print c1
  print c0

  # features
  vars_g = ["Z1_E","Z1_pt","Z1_eta","Z1_phi","Z1_m","Z2_E","Z2_pt","Z2_eta","Z2_phi","Z2_m","higgs_E","higgs_pt","higgs_eta","higgs_phi","higgs_m","DelPhi_Hjj","mH","pT_Hjj","DelEta_jj","EtaProd_jj","DelY_jj","DelPhi_jj","DelR_jj","Mjj","Mjets","njets","jet1_E","jet1_eta","jet1_y","jet1_phi","jet1_pt","jet1_m","jet1_isPU","jet2_E","jet2_phi","jet2_eta","jet2_y","jet2_pt","jet2_m","jet2_isPU","DelPt_jj","minDelR_jZ","DelPt_ZZ","Zeppetaj3","ZeppetaZZ","jet3_E","jet3_eta","jet3_phi","jet3_pt","jet3_m","jet3_isPU"]
  # This features were selected by inspection
  accept_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,20,24,25,26,27,28,29,30,31,36,40,42]
  vars_g = [vars_g[i] for i,_ in enumerate(vars_g) if i in accept_list]

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

  # Define range for fitting or cross section check
  npoints = 15
  c_eval = 1
  #c_min = [0.6,0.1]
  #c_max = [1.5,0.9]
  c_min = 0.6
  c_max = 1.5

  # Checking histograms of each feature
  #fullCrossSectionCheck(dir,c1_g,model_g,data_files,f1_dist,accept_list,c_min,c_max,npoints,c_eval)
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

  # Fitting distributions  

  #test.fit(data_file='data',importance_sampling=False, true_dist=False,vars_g=vars_g)

  # Compute likelihood ratios 
  #test.computeRatios(data_file='data',true_dist=False,vars_g=vars_g,use_log=False) 
  #pdb.set_trace()

  n_hist = 1050
  # Fit coupling values
  test.fitCValues(c0,c1,data_file='data', true_dist=False,vars_g=vars_g,use_log=False,
            n_hist=n_hist, num_pseudodata=5000,weights_func=getWeights)

  #plotCValues(c0,c1,dir=dir,c1_g=c1_g,model_g=model_g,true_dist=False,vars_g=vars_g,
  #    workspace=workspace_file,use_log=False,n_hist=n_hist,c_eval=1,range_min=1.1,
  #    range_max=1.9)


