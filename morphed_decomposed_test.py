
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
          loadData,makeSigBkg,makeROC, makeMultiROC,saveMultiFig,preProcessing
from train_classifiers import predict

from pyMorphWrapper import MorphingWrapper

from decomposed_test import DecomposedTest


class MorphedDecomposedTest(DecomposedTest):
  '''
    Inherited class which implement decomposed test with
    morhed bases
  '''

  def pre2DDoubleBasis(self,c_min,c_max,npoints):
    '''
      Compute the bases for the morphing method, 
      2 almost orthgonal bases are computed and weights 
      and cross section are calculated for npoints between 
      c_min and c_max. All the values are stored in files.
    '''

    csarray = np.linspace(c_min[0],c_max[0],npoints)
    csarray2 = np.linspace(c_min[1], c_max[1], npoints)
    n_eff_ratio = np.zeros((npoints,npoints))
    cross_section = None
    morph = MorphingWrapper()

    morph.setSampleData(nsamples=15,ncouplings=3,types=['S','S','S'],morphed=self.F1_couplings,samples=self.all_couplings)
    # This compute the morphed bases
    indexes = morph.dynamicMorphing2(self.F1_couplings,csarray,csarray2)
    target = self.F1_couplings[:]
    # Start computing couplings and cross sections
    for l,ind in enumerate(indexes): 
      ind = np.array(ind)
      morph.resetBasis([self.all_couplings[int(k)] for k in ind]) 
      sorted_indexes = np.argsort(ind)
      indexes[l] = ind[sorted_indexes]
      print ind
      for i,cs in enumerate(csarray):
        for j,cs2 in enumerate(csarray2):
          target[1] = cs
          target[2] = cs2
          morph.resetTarget(target)
          couplings = np.array(morph.getWeights())
          cross_section = np.array(morph.getCrossSections())
          couplings,cross_section = (couplings[sorted_indexes],
                            cross_section[sorted_indexes])
          all_couplings = np.vstack([all_couplings,couplings])  if i <> 0 or j <> 0 or l <> 0 else couplings
          all_cross_sections = np.vstack([all_cross_sections, cross_section]) if i <> 0 or j <> 0 or l <> 0 else cross_section
          c1s = np.multiply(couplings,cross_section)
          n_eff = c1s.sum()
          n_tot = np.abs(c1s).sum()
          n_eff_ratio[i] = n_eff/n_tot
          print '{0} {1} {2}'.format(l,i,j)
          print target
          print 'n_eff: {0}, n_tot: {1}, n_eff/n_tot: {2}'.format(n_eff, n_tot, n_eff/n_tot)

    np.savetxt('3doubleindexes_{0:.2f}_{1:.2f}_{2:.2f}_{3:.2f}_{4}.dat'.format(c_min[0],c_min[1],c_max[0],c_max[1],npoints),indexes) 
    np.savetxt('3doublecouplings_{0:.2f}_{1:.2f}_{2:.2f}_{3:.2f}_{4}.dat'.format(c_min[0],c_min[1],c_max[0],c_max[1],npoints),all_couplings) 
    np.savetxt('3doublecrosssection_{0:.2f}_{1:.2f}_{2:.2f}_{3:.2f}_{4}.dat'.format(c_min[0],c_min[1],c_max[0],c_max[1],npoints),all_cross_sections) 
 
  def evalDoubleC1C2Likelihood(self,w,testdata,c0,c1,c_eval=0,c_min=0.01,c_max=0.2,use_log=False,true_dist=False, vars_g=None, npoints=50,samples_ids=None,weights_func=None):
    '''
      Find minimum of likelihood on testdata using decomposed
      ratios and the weighted orthogonal morphing method to find the bases
    '''

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

    # Compute bases if they don't exist for this range
    if not os.path.isfile('3doubleindexes_{0:.2f}_{1:.2f}_{2:.2f}_{3:.2f}_{4}.dat'.format(c_min[0],c_min[1],c_max[0],c_max[1],npoints)): 
      self.pre2DDoubleBasis(c_min=c_min,c_max=c_max,npoints=npoints)   

    csarray = np.linspace(c_min[0],c_max[0],npoints)
    csarray2 = np.linspace(c_min[1], c_max[1], npoints)
    decomposedLikelihood = np.zeros((npoints,npoints))
    trueLikelihood = np.zeros((npoints,npoints))

    all_indexes = np.loadtxt('3doubleindexes_{0:.2f}_{1:.2f}_{2:.2f}_{3:.2f}_{4}.dat'.format(c_min[0],c_min[1],c_max[0],c_max[1],npoints)) 
    all_indexes = np.array([[int(x) for x in rows] for rows in all_indexes])
    all_couplings = np.loadtxt('3doublecouplings_{0:.2f}_{1:.2f}_{2:.2f}_{3:.2f}_{4}.dat'.format(c_min[0],c_min[1],c_max[0],c_max[1],npoints)) 
    all_cross_sections = np.loadtxt('3doublecrosssection_{0:.2f}_{1:.2f}_{2:.2f}_{3:.2f}_{4}.dat'.format(c_min[0],c_min[1],c_max[0],c_max[1],npoints))
    
    # Bkg used in the fit
    # TODO: Harcoded this have to be changed
    basis_value = 2
    
    n_eff_ratio = np.zeros((csarray.shape[0], csarray2.shape[0]))
    n_eff_1s = np.zeros((csarray.shape[0], csarray2.shape[0]))
    n_eff_2s = np.zeros((csarray.shape[0], csarray2.shape[0]))

    # Pre evaluate the values for each distribution
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
        # This save some time by only evaluating the needed samples
        if index_k <> basis_value:
          continue
        print 'Pre computing {0} {1}'.format(index_k,index_j)
        if k <> j:
          f0pdf = w.function('bkghistpdf_{0}_{1}'.format(index_k,index_j))
          f1pdf = w.function('sighistpdf_{0}_{1}'.format(index_k,index_j))
          data = testdata
          if self.preprocessing == True:
            data = preProcessing(testdata,self.dataset_names[min(k,j)],
            self.dataset_names[max(k,j)],self.scaler) 
          #outputs = predict('{0}/model/{1}/{2}/{3}_{4}_{5}.pkl'.format(self.dir,self.model_g,
          outputs = predict('/afs/cern.ch/work/j/jpavezse/private/{0}_{1}_{2}.pkl'.format(self.model_file,index_k,index_j),data,model_g=self.model_g)
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
    # Usefull values to inspect after the training 
    alpha = np.zeros([csarray.shape[0],csarray2.shape[0],2])
    n_eff_ratio = np.zeros((csarray.shape[0], csarray2.shape[0]))
    n_eff_1s = np.zeros((csarray.shape[0], csarray2.shape[0]))
    n_eff_2s = np.zeros((csarray.shape[0], csarray2.shape[0]))
    n_tot_1s = np.zeros((csarray.shape[0], csarray2.shape[0]))
    n_tot_2s = np.zeros((csarray.shape[0], csarray2.shape[0]))
    n_zeros = np.zeros((npoints,npoints))
    target = self.F1_couplings[:]
    for i,cs in enumerate(csarray):
      ratiosList.append([])
      for j, cs2 in enumerate(csarray2):
        target[1] = cs
        target[2] = cs2
        print '{0} {1}'.format(i,j)
        print target

        # Compute F1 couplings and cross sections
        c1s_1 = all_couplings[i*npoints + j]
        cross_section_1 = all_cross_sections[i*npoints + j] 
        c1s_1 = np.multiply(c1s_1,cross_section_1)
        n_eff = c1s_1.sum()
        n_tot = np.abs(c1s_1).sum()
        n_eff_1 = n_eff / n_tot
        n_eff_1s[i,j] = n_eff_1
        n_tot_1s[i,j] = n_tot
        print 'n_eff 1: {0}'.format(n_eff/n_tot)
        c1s_1 = c1s_1/c1s_1.sum()

        c1s_2 = all_couplings[npoints*npoints + i*npoints + j]
        cross_section_2 = all_cross_sections[npoints*npoints + i*npoints + j] 
        c1s_2 = np.multiply(c1s_2,cross_section_2)
        n_eff = c1s_2.sum()
        n_tot = np.abs(c1s_2).sum()
        n_eff_2 = n_eff / n_tot
        n_eff_2s[i,j] = n_eff_2
        n_tot_2s[i,j] = n_tot
        print 'n_eff 2: {0}'.format(n_eff/n_tot)
        c1s_2 = c1s_2/c1s_2.sum()

        # Compute weights for bases
        neff2 = 1./n_eff_2s[i,j]
        neff1 = 1./n_eff_1s[i,j] 
        alpha1 = np.exp(-np.power(neff1,1./3.))
        alpha2 = np.exp(-np.power(neff2,1./3.))
        alpha1 = np.exp(-np.sqrt(neff1))
        alpha2 = np.exp(-np.sqrt(neff2))
        #alpha1 = np.exp(n_eff_1s[i,j])
        #alpha2 = np.exp(n_eff_2s[i,j])
        alpha[i,j,0] = alpha1/(alpha1 + alpha2)
        alpha[i,j,1] = alpha2/(alpha1 + alpha2)

        # Compute Bkg weights
        c0_arr_1 = np.zeros(15)
        c0_arr_2 = np.zeros(15)
        c0_arr_1[np.where(all_indexes[0] == basis_value)[0][0]] = 1.
        c0_arr_2[np.where(all_indexes[1] == basis_value)[0][0]] = 1.

        c0_arr_1 = c0_arr_1/c0_arr_1.sum()
        c0_arr_2 = c0_arr_2/c0_arr_2.sum()

        c1s = np.append(alpha[i,j,0]*c1s_1,alpha[i,j,1]*c1s_2) 
        c0_arr = np.append(0.5*c0_arr_1,0.5*c0_arr_2)

        print c0_arr

        cross_section = np.append(cross_section_1,cross_section_2)
        indexes = np.append(all_indexes[0],all_indexes[1])
        completeRatios,trueRatios = evaluateRatio(w,testdata,x=x,
        plotting=False,roc=False,c0arr=c0_arr,c1arr=c1s,true_dist=true_dist,
        pre_dist=pre_dist,pre_evaluation=pre_pdf,cross_section=cross_section,
        indexes=indexes)
        completeRatios = 1./completeRatios

        print completeRatios[completeRatios < 0.].shape
        n_zeros[i,j] = completeRatios[completeRatios < 0.].shape[0]
        ratiosList[i].append(completeRatios)
        n_eff_ratio[i,j] = (alpha[i,j,0]*n_eff_1 + alpha[i,j,1]*n_eff_2) 

        print 'total eff: {0}'.format(n_eff_ratio[i,j])
        if n_eff_ratio[i,j] > 0.3:
          indices = np.logical_and(indices, completeRatios > 0.)
    for i,cs in enumerate(csarray):
      for j, cs2 in enumerate(csarray2):

        completeRatios = ratiosList[i][j]
        completeRatios = completeRatios[indices]
        if use_log == False:
            norm = completeRatios[completeRatios <> 0.].shape[0] 
            if n_eff_ratio[i,j] < 0.3:
              #TODO: Harcoded number
              decomposedLikelihood[i,j] = 20000
            else:
              decomposedLikelihood[i,j] = -np.log(completeRatios).sum() 
        else:
          decomposedLikelihood[i,j] = completeRatios.sum()
          trueLikelihood[i,j] = trueRatios.sum()
    decomposedLikelihood[decomposedLikelihood == 20000] = decomposedLikelihood[decomposedLikelihood <> 20000].max()
    decomposedLikelihood = decomposedLikelihood - decomposedLikelihood.min()
    decMin = np.unravel_index(decomposedLikelihood.argmin(), decomposedLikelihood.shape)

    # Plotting 
    # pixel plots
    saveFig(csarray,[csarray2,n_eff_1s/n_eff_2s],makePlotName('comp','train',type='n_eff_ratio'),labels=['composed'],pixel=True,marker=True,dir=self.dir,model_g=self.model_g,marker_value=(c1[0],c1[1]),print_pdf=True,contour=True,title='n_rat_1/n_rat_2 values for g1,g2')
  
    saveFig(csarray,[csarray2,n_eff_ratio],makePlotName('comp','train',type='n_eff'),labels=['composed'],pixel=True,marker=True,dir=self.dir,model_g=self.model_g,marker_value=(c1[0],c1[1]),print_pdf=True,contour=True,title='n_eff/n_tot sum values for g1,g2')

    saveFig(csarray,[csarray2,n_eff_1s],makePlotName('comp','train',type='n_eff1'),labels=['composed'],pixel=True,marker=True,dir=self.dir,model_g=self.model_g,marker_value=(c1[0],c1[1]),print_pdf=True,contour=True,title='n_eff_1 ratio values for g1,g2')

    saveFig(csarray,[csarray2,n_eff_2s],makePlotName('comp','train',type='n_eff2'),labels=['composed'],pixel=True,marker=True,dir=self.dir,model_g=self.model_g,marker_value=(c1[0],c1[1]),print_pdf=True,contour=True,title='n_eff_2 ratiovalues for g1,g2')


    saveFig(csarray,[csarray2,alpha[:,:,0]],makePlotName('comp','train',type='alpha1'),labels=['composed'],pixel=True,marker=True,dir=self.dir,model_g=self.model_g,marker_value=(c1[0],c1[1]),print_pdf=True,contour=True,title='weights_1 ratio values for g1,g2')

    saveFig(csarray,[csarray2,alpha[:,:,1]],makePlotName('comp','train',type='alpha2'),labels=['composed'],pixel=True,marker=True,dir=self.dir,model_g=self.model_g,marker_value=(c1[0],c1[1]),print_pdf=True,contour=True,title='weights_2 ratiovalues for g1,g2')


    saveFig(csarray,[csarray2,n_tot_1s],makePlotName('comp','train',type='n_tot1'),labels=['composed'],pixel=True,marker=True,dir=self.dir,model_g=self.model_g,marker_value=(c1[0],c1[1]),print_pdf=True,contour=True,title='n_tot_1 values for g1,g2')

    saveFig(csarray,[csarray2,n_tot_2s],makePlotName('comp','train',type='n_tot2'),labels=['composed'],pixel=True,marker=True,dir=self.dir,model_g=self.model_g,marker_value=(c1[0],c1[1]),print_pdf=True,contour=True,title='n_tot_2 values for g1,g2')


    saveFig(csarray,[csarray2,n_zeros],makePlotName('comp','train',type='n_zeros'),labels=['composed'],pixel=True,marker=True,dir=self.dir,model_g=self.model_g,marker_value=(c1[0],c1[1]),print_pdf=True,contour=True,title='n_zeros values for g1,g2')

    saveFig(csarray,[csarray2,decomposedLikelihood],makePlotName('comp','train',type='pixel_g1g2'),labels=['composed'],pixel=True,marker=True,dir=self.dir,model_g=self.model_g,marker_value=(c1[0],c1[1]),print_pdf=True,contour=True,title='Likelihood fit for g1,g2')

    #decMin = [np.sum(decomposedLikelihood,1).argmin(),np.sum(decomposedLikelihood,0).argmin()] 
    X,Y = np.meshgrid(csarray, csarray2)

    saveFig(X,[Y,decomposedLikelihood],makePlotName('comp','train',type='multilikelihood_{0:.2f}_{1:.2f}'.format(self.F1_couplings[1],self.F1_couplings[2])),labels=['composed'],contour=True,marker=True,dir=self.dir,model_g=self.model_g,marker_value=(self.F1_couplings[1],self.F1_couplings[2]),print_pdf=True,min_value=(csarray[decMin[0]],csarray2[decMin[1]]))
    #print decMin
    print [csarray[decMin[0]],csarray2[decMin[1]]]
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
    c_min = [-1.1,-1.1]
    c_max = [-0.1,-0.1]

    f = ROOT.TFile('{0}/{1}'.format('/afs/cern.ch/work/j/jpavezse/private/',self.workspace))
    w = f.Get('w')
    f.Close()
    assert w 

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
  
    testdata = np.loadtxt('{0}/data/{1}/{2}/{3}_{4}.dat'.format(self.dir,'mlp',self.c1_g,data_file,self.F1_dist))[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,20,24,25,26,27,28,29,30,31,36,40,42]]
    print self.F1_dist
    print testdata.shape

    fil1 = open('{0}/fitting_values_c1.txt'.format(self.dir),'a')
    for i in range(n_hist):
      indexes = rng.choice(testdata.shape[0], num_pseudodata) 
      dataset = testdata[indexes]
      ((c1_true,c2_true),(c1_dec,c2_dec)) = self.evalDoubleC1C2Likelihood(w,dataset, c0,c1,c_eval=c_eval,c_min=c_min,
      c_max=c_max,true_dist=true_dist,vars_g=vars_g,weights_func=weights_func,
                    npoints=npoints,use_log=use_log)  
      print '2: {0} {1} {2} {3}'.format(c1_true, c1_dec, c2_true, c2_dec)
      fil1.write('{0} {1} {2} {3}\n'.format(c1_true, c1_dec, c2_true, c2_dec))
    fil1.close()  
 
