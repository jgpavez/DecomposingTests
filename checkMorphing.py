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
from morphed_decomposed_test import MorphedDecomposedTest

from xgboost_wrapper import XGBoostClassifier
from pyMorphWrapper import MorphingWrapper

def evalDist(x, f0, val):
    iter = x.createIterator()
    v = iter.Next()
    i = 0
    while v:
        v.setVal(val[i])
        v = iter.Next()
        i = i + 1
    return f0.getVal(x)

def checkCrossSection(
        c1,
        cross_section,
        samples,
        feature=0,
        targetdata=None,
        samplesdata=None):
    '''
      Build morphed histograms for a feature
    '''

    w = ROOT.RooWorkspace('w')
    normalizer_abs = (np.abs(np.multiply(c1, cross_section))).sum()
    normalizer = (np.multiply(c1, cross_section)).sum()
    n_eff = normalizer / normalizer_abs
    print 'n_eff_ratio: {0}, n_tot: {0}'.format(n_eff, normalizer_abs)
    #normalizer = cross_section.sum()

    data_file = 'data'
    targetdata = targetdata[:80000, feature]
    targetdata = targetdata[targetdata != -999.]
    targetdata = np.abs(targetdata)
    #targetdata = targetdata[findOutliers(targetdata)]

    bins = 20
    minimum = targetdata.min()
    maximum = targetdata.max()
    low = minimum
    high = maximum
    q5, q95 = np.percentile(targetdata, [5, 95])
    low_hist = q5
    high_hist = q95
    #low = minimum - ((maximum - minimum) / bins) * 10
    #high = maximum + ((maximum - minimum) / bins) * 10
    low_hist = minimum
    high_hist = maximum
    x_range = (low,high)
    x_values = np.linspace(low, high, bins)
    plt.figure(1)
    plt.clf()
    plt.subplot(212)

    w.factory('score[{0},{1}]'.format(low, high))
    s = w.var('score')
    score = ROOT.RooArgSet(w.var('score'))

    target_hist = ROOT.TH1F('targethist', 'targethist', bins, low_hist, high_hist)
    for val in targetdata:
        target_hist.Fill(val)
    norm = 1. / target_hist.Integral()
    target_hist.Scale(norm)
    # Creating samples histograms
    samples_hists = []
    samples_pdfs = []
    samples_datas = []
    samples_hists = []
    sum_hist = ROOT.TH1F('sampleshistsum', 'sampleshistsum', bins, low_hist, high_hist)
    
    for i, sample in enumerate(samples):
        samples_hists.append(ROOT.TH1F(
            'sampleshist{0}'.format(i),
            'sampleshist',
            bins,
            low_hist,
            high_hist))
        samples_hist = samples_hists[-1]
        testdata = samplesdata[i]
        testdata = testdata[:80000, feature]
        testdata = testdata[testdata != -999.]
        testdata = np.abs(testdata)
        #testdata = testdata[findOutliers(testdata)]
        weight = (c1[i] * cross_section[i]) / normalizer
        for val in testdata:
            samples_hist.Fill(val)
            # samples_hist.Fill(val,weight)
        norm = 1. / samples_hist.Integral()
        samples_hist.Scale(norm)
        samples_hists.append(samples_hist)
        sum_hist.Add(samples_hist, weight)
        
        samples_datas.append(ROOT.RooDataHist('{0}datahist{1}'.format(
            'samples',i), 'histsamples', ROOT.RooArgList(s), samples_hists[-1]))
        samples_pdfs.append(ROOT.RooHistFunc(
            '{0}histpdf{1}'.format('samples',i),
            'histsamples',
            ROOT.RooArgSet(s),
            samples_datas[-1],
            0))
        print weight
    
        hist_values = np.array(
            [evalDist(score, samples_pdfs[-1], [xs]) for xs in x_values])
        plt.plot(x_values, hist_values,linewidth=0.5,alpha=0.8)
        
    target_datahist = ROOT.RooDataHist(
        '{0}datahist'.format('target'),
        'histtarget',
        ROOT.RooArgList(s),
        target_hist)
    target_histpdf = ROOT.RooHistFunc(
        '{0}histpdf'.format('target'),
        'histtarget',
        ROOT.RooArgSet(s),
        target_datahist,
        0)
    samples_datahist = ROOT.RooDataHist('{0}datahist'.format(
        'samples'), 'histsamples', ROOT.RooArgList(s), sum_hist)
    samples_histpdf = ROOT.RooHistFunc(
        '{0}histpdf'.format('samples'),
        'histsamples',
        ROOT.RooArgSet(s),
        samples_datahist,
        0)
    #printMultiFrame(w,['score','score'],[[target_histpdf,samples_histpdf],samples_pdfs],
    #                                'check_cross_section_{0}_1'.format(feature),[['real_0','weighted_0'],
    #                                ['sample_{0}'.format(k) for k,_ in enumerate(samples_pdfs)]], dir=dir,
    #                                model_g=model_g,title='cross section check, Base 2',x_text='x',y_text='dN')
    
    plt.subplot(211)
    hist_values = np.array(
            [evalDist(score, samples_histpdf, [xs]) for xs in x_values])
    plt.plot(x_values, hist_values, 'b', label='Weighted')
    hist_values = np.array(
            [evalDist(score, target_histpdf, [xs]) for xs in x_values])
    plt.plot(x_values, hist_values, 'r', label='Real')
    plt.legend(loc=2)
    #plt.title('Base 1, features {0}'.format(vars_g[feature]))
    plt.title('Base 1, features {0}'.format(feature))
    
    plt.savefig('plots/morph/cross_section_{0}'.format(feature))
    
    # Now compute likelihood
    evalValues = np.array(
        [evalDist(score, samples_histpdf, [xs]) for xs in targetdata])
    n_zeros = evalValues[evalValues <= 0.].shape[0]
    evalValues = evalValues[evalValues > 0.]
    print evalValues.shape
    likelihood = -np.log(evalValues).sum()
    print likelihood
    return likelihood, n_eff, n_zeros


if __name__ == "__main__":
    # Compute both bases
    np.random.seed(1234)
    
    # Define fitting ranges
    g1_range = (-0.3557,0.2646)
    g2_range = (-0.34467,0.34467)
    g_ranges = (g1_range, g2_range)
    
    nsamples = 15
    ncomb = 18
    npoints = 15
    
    base = 0
    point = 0
    dim = 0
    c_min, c_max = g_ranges[dim]

    morph = MorphingWrapper()

    # List of availables basis samples
    # Samples used in the evolutionary algorithm
    theta = [[ 0.0230769230769, -0.253846153846 ],
             [-0.0230769230769, -0.853846153846 ],
             [ 0.0692307692308, -0.0692307692308],
             [ 0.0692307692308,  0.207692307692 ],
             [-0.115384615385,  -0.9            ],
             [-0.161538461538,   0.346153846154 ],
             [-0.207692307692,   0.0230769230769],
             [-0.207692307692,   0.669230769231 ],
             [ 0.253846153846,   0.253846153846 ],
             [-0.253846153846,  -0.3            ],
             [ 0.253846153846,   0.853846153846 ],
             [-0.346153846154,   0.9            ],
             [-0.3,              0.115384615385 ],
             [ 0.3,             -0.115384615385 ],
             [ 0.3,             -0.484615384615 ],
             [ 0.3,              0.761538461538 ],
             [-0.438461538462,  -0.484615384615 ],
             [ 0.438461538462,  -0.853846153846 ],
             [ 0.530769230769,  -0.0230769230769],
             [ 0.530769230769,   0.253846153846 ],
             [-0.530769230769,   0.3            ],
             [-0.576923076923,  -0.9            ],
             [-0.623076923077,  -0.161538461538 ],
             [-0.623076923077,   0.669230769231 ],
             [ 0.669230769231,  -0.669230769231 ],
             [ 0.761538461538,   0.392307692308 ],
             [-0.761538461538,  -0.761538461538 ],
             [ 0.761538461538,  -0.761538461538 ],
             [-0.807692307692,   0.0230769230769],
             [-0.807692307692,   0.161538461538 ]] 
    theta = [[1.,s[0],s[1]] for s in theta]

    # Using half of range as initial target (used only to make computation faster)
    target = [1.,1.,1.]
    morph.setSampleData(nsamples=nsamples,ncouplings=3,types=['S','S','S'],samples=theta,
          ncomb=ncomb)
    
    indexes = [range(15), range(15,30)]


    # Save cross sections and couplings for each one of the points on the fitting space
    # Also compute the weighted n_eff
    npoints = 10
    csarray1 = np.linspace(g1_range[0],g1_range[1],npoints)
    csarray2 = np.linspace(g2_range[0], g2_range[1], npoints)
    n_effs_1 = np.zeros((csarray1.shape[0], csarray2.shape[0]))
    all_couplings = np.zeros((2,npoints,npoints))
    for l,ind in enumerate(indexes): 
        ind = np.array(ind)
        morph.resetBasis([theta[int(k)] for k in ind]) 
        sorted_indexes = np.argsort(ind)
        indexes[l] = ind[sorted_indexes]
        for i,cs in enumerate(csarray1):
            for j,cs2 in enumerate(csarray2):
                target[1] = cs
                target[2] = cs2 
                morph.resetTarget(target)
                # Compute weights and cross section of each sample
                couplings = np.array(morph.getWeights())
                cross_section = np.array(morph.getCrossSections())

                couplings,cross_section = (couplings[sorted_indexes],
                            cross_section[sorted_indexes])
                # Save list of cross sections and weights for each samples and orthogonal bases
                all_couplings = np.vstack([all_couplings,couplings])  if i <> 0 or j <> 0 or l <> 0 else couplings

                all_cross_sections = np.vstack([all_cross_sections, cross_section]) if i <> 0 or j <> 0 or l <> 0 else cross_section
    all_couplings = np.array(all_couplings)
    all_cross_sections = np.array(all_cross_sections)
    print(all_couplings.shape)
    print(all_cross_sections.shape)
    
    all_indexes = indexes[base]
    all_couplings = all_couplings[base*15 + point]
    all_cross_sections = all_cross_sections[base*15 + point]

    data_file = 'data'
    # Loading target and samples data  
    samplesdata = []
    
    for i, sample in enumerate(all_indexes):
        samplesdata.append(np.loadtxt(
            '/afs/cern.ch/work/j/jpavezse/public/samples_cyril/newer_samples/data/{0}_{1}.dat'.format(data_file,sample)))

    targetdata = np.loadtxt(
        '/afs/cern.ch/work/j/jpavezse/public/samples_cyril/newer_samples/data/{0}_{1}.dat'.format(data_file, "sm"))

    accept_list = range(samplesdata[0].shape[1])    
    features = accept_list

    n_effs = np.zeros(len(features))
    n_zeros = np.zeros(len(features))
    likelihoods = []
    #samplesdata = []
    for i, f in enumerate(features):
        likelihood, n_eff, n_zero = checkCrossSection(all_couplings, all_cross_sections, all_indexes,
                                                      feature=f, targetdata=targetdata, samplesdata=samplesdata)
        likelihoods.append(likelihood)
        n_effs[i] = n_eff
        n_zeros[i] = n_zero

    print likelihoods
    
    likelihoods = np.array(likelihoods)
    likelihoods = likelihoods - np.abs(likelihoods.min(axis=0))
    likelihoods = likelihoods / likelihoods.max(axis=0)

    n_zeros_max = n_zeros.max(axis=0)
    n_zeros = n_zeros.transpose() / n_zeros.max(axis=0)
    n_zeros = n_zeros.transpose()
    
    
    
    

