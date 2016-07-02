#!/usr/bin/env python
__author__ = "Pavez J. <juan.pavezs@alumnos.usm.cl>"


import ROOT
import numpy as np

import sys

import os.path

from utils import printMultiFrame, printFrame, saveFig, loadData, printFrame, makePlotName,\
    makeSigBkg, makeROC, makeMultiROC, saveMultiFig
    
from make_data import makeData
from train_classifiers import predict


class DecomposedTest:
    '''
      Class which implement the decomposed test on
      the data
    '''

    def __init__(
            self,
            c0,
            c1,
            model_file='adaptive',
            input_workspace=None,
            output_workspace='workspace_DecomposingTestOfMixtureModelsClassifiers.root',
            dir='/afs/cern.ch/user/j/jpavezse/systematics',
            c1_g='',
            model_g='mlp',
            verbose_printing=False,
            dataset_names=None,
            seed=1234,
            F1_dist='F1',
            F0_dist='F0',
            all_couplings=None,
            F1_couplings=None,
            basis_indexes=None,
            clf=None):
        self.input_workspace = input_workspace
        self.workspace = output_workspace
        self.c0 = c0
        self.c1 = c1
        self.model_file = model_file
        self.dir = dir
        self.c1_g = c1_g
        self.model_g = model_g
        self.verbose_printing = verbose_printing
        self.seed = seed
        self.F1_dist = F1_dist
        self.F0_dist = F0_dist
        self.dataset_names = dataset_names
        self.basis_indexes = basis_indexes if basis_indexes is not None else range(
            len(dataset_names))
        self.F1_couplings = F1_couplings
        self.all_couplings = all_couplings
        self.nsamples = len(dataset_names)
        self.clf = clf

    def fit(
            self,
            data_file='test',
            true_dist=True,
            vars_g=None):
        '''
          Create pdfs for the classifier
          score to be used later on the ratio
          test, input workspace only needed in case
          there exist true pdfs for the distributions
          the models being used are ./model/{model_g}/{c1_g}/{model_file}_i_j.pkl
          and the data files are ./data/{model_g}/{c1_g}/{data_file}_i_j.dat
        '''

        bins = 40
        low = 0.
        high = 1.

        if self.input_workspace is not None:
            f = ROOT.TFile('{0}/{1}'.format(self.dir, self.workspace))
            w = f.Get('w')
            w = ROOT.RooWorkspace('w') if w is None else w
            f.Close()
        else:
            w = ROOT.RooWorkspace('w')
        w.Print()

        print 'Generating Score Histograms'

        w.factory('score[{0},{1}]'.format(low, high))
        s = w.var('score')

        bins_full = 40
        low_full = 0.
        high_full = 1.
        w.factory('scoref[{0},{1}]'.format(low_full, high_full))
        s_full = w.var('scoref')
        
        histos = []
        histos_names = []
        inv_histos = []
        inv_histos_names = []
        sums_histos = []

        def saveHistos(
                w,
                outputs,
                s,
                bins,
                low,
                high,
                pos=None):
            if pos is not None:
                k, j = pos
            else:
                k, j = ('F0', 'F1')
            print 'Estimating {0} {1}'.format(k, j)
            for l, name in enumerate(['sig', 'bkg']):
                data = ROOT.RooDataSet(
                    '{0}data_{1}_{2}'.format(
                        name, k, j), "data", ROOT.RooArgSet(s))
                hist = ROOT.TH1F(
                    '{0}hist_{1}_{2}'.format(
                        name, k, j), 'hist', bins, low, high)
                values = outputs[l]
                for val in values:
                    hist.Fill(val)
                    s.setVal(val)
                    data.add(ROOT.RooArgSet(s))
                norm = 1. / hist.Integral()
                hist.Scale(norm)

                s.setBins(bins)
                datahist = ROOT.RooDataHist(
                    '{0}datahist_{1}_{2}'.format(
                        name, k, j), 'hist', ROOT.RooArgList(s), hist)

                histpdf = ROOT.RooHistFunc('{0}histpdf_{1}_{2}'.format(
                    name, k, j), 'hist', ROOT.RooArgSet(s), datahist, 1)

                getattr(w, 'import')(hist)
                getattr(w, 'import')(data)
                getattr(w, 'import')(datahist)
                getattr(w, 'import')(histpdf)
                score_str = 'scoref' if pos is None else 'score'
                # Calculate the density of the classifier output using kernel density
                # w.factory('KeysPdf::{0}dist_{1}_{2}({3},{0}data_{1}_{2},RooKeysPdf::NoMirror,2)'.format(name,k,j,score_str))

                # Print histograms pdfs and estimated densities
                if self.verbose_printing and name == 'bkg' and k != j:
                    full = 'full' if pos is None else 'dec'
                    if k < j and k != 'F0':
                        histos.append([w.function('sighistpdf_{0}_{1}'.format(
                            k, j)), w.function('bkghistpdf_{0}_{1}'.format(k, j))])
                        histos_names.append(
                            ['f{0}-f{1}_f{1}(signal)'.format(k, j), 'f{0}-f{1}_f{0}(background)'.format(k, j)])
                    if j < k and k != 'F0':
                        inv_histos.append([w.function('sighistpdf_{0}_{1}'.format(
                            k, j)), w.function('bkghistpdf_{0}_{1}'.format(k, j))])
                        inv_histos_names.append(
                            ['f{0}-f{1}_f{1}(signal)'.format(k, j), 'f{0}-f{1}_f{0}(background)'.format(k, j)])

        for k in range(self.nsamples):
            for j in range(self.nsamples):
                if k == j:
                    continue
                if self.dataset_names is not None:
                    name_k, name_j = (
                        self.dataset_names[k], self.dataset_names[j])
                else:
                    name_k, name_j = (k, j)
                print 'Loading {0}:{1} {2}:{3}'.format(k, name_k, j, name_j)
                traindata, targetdata = loadData(data_file, name_k, name_j, dir=self.dir, c1_g=self.c1_g)

                numtrain = traindata.shape[0]
                size2 = traindata.shape[1] if len(traindata.shape) > 1 else 1
                output = [
                    predict(
                        '{0}/model/{1}/{2}/{3}_{4}_{5}.pkl'.format(
                            self.dir, self.model_g, self.c1_g, self.model_file, k, j), traindata[
                            targetdata == 1], model_g=self.model_g, clf=self.clf), predict(
                        '{0}/model/{1}/{2}/{3}_{4}_{5}.pkl'.format(
                            self.dir, self.model_g, self.c1_g, self.model_file, k, j), traindata[
                            targetdata == 0], model_g=self.model_g, clf=self.clf)]
                saveHistos(w, output, s, bins, low, high, (k, j))
                w.writeToFile('{0}/{1}'.format(self.dir, self.workspace))

        if self.verbose_printing:
            for ind in range(1, (len(histos) / 3 + 1)):
                print_histos = histos[(ind - 1) * 3:(ind - 1) * 3 + 3]
                print_histos_names = histos_names[
                    (ind - 1) * 3:(ind - 1) * 3 + 3]
                printMultiFrame(
                    w,
                    ['score'] * len(print_histos),
                    print_histos,
                    makePlotName(
                        'dec{0}'.format(
                            ind - 1),
                        'all',
                        type='hist',
                        dir=self.dir,
                        c1_g=self.c1_g,
                        model_g=self.model_g),
                    print_histos_names,
                    dir=self.dir,
                    model_g=self.model_g,
                    y_text='score(x)',
                    print_pdf=True,
                    title='Pairwise score distributions')
        # Full model
        traindata, targetdata = loadData(data_file, self.F0_dist, self.F1_dist, dir=self.dir, c1_g=self.c1_g)
        numtrain = traindata.shape[0]
        size2 = traindata.shape[1] if len(traindata.shape) > 1 else 1
        outputs = [predict('{0}/model/{1}/{2}/{3}_F0_F1.pkl'.format(self.dir,
                                                                    self.model_g,
                                                                    self.c1_g,
                                                                    self.model_file),
                           traindata[targetdata == 1],
                           model_g=self.model_g,
                           clf=self.clf),
                   predict('{0}/model/{1}/{2}/{3}_F0_F1.pkl'.format(self.dir,
                                                                    self.model_g,
                                                                    self.c1_g,
                                                                    self.model_file),
                           traindata[targetdata == 0],
                           model_g=self.model_g,
                           clf=self.clf)]
        saveHistos(
            w,
            outputs,
            s_full,
            bins_full,
            low_full,
            high_full)
        if self.verbose_printing:
            printFrame(w,
                       ['scoref'],
                       [w.function('sighistpdf_F0_F1'),
                        w.function('bkghistpdf_F0_F1')],
                       makePlotName('full',
                                    'all',
                                    type='hist',
                                    dir=self.dir,
                                    c1_g=self.c1_g,
                                    model_g=self.model_g),
                       ['signal',
                           'bkg'],
                       dir=self.dir,
                       model_g=self.model_g,
                       y_text='score(x)',
                       print_pdf=True,
                       title='Pairwise score distributions')

        w.writeToFile('{0}/{1}'.format(self.dir, self.workspace))

        w.Print()

    # To calculate the ratio between single functions
    def singleRatio(self, f0, f1):
        ratio = f1 / f0
        ratio[~np.isfinite(ratio)] = 0
        return ratio

    def evalDist(self, x, f0, val):
        iter = x.createIterator()
        v = iter.Next()
        i = 0
        while v:
            v.setVal(val[i])
            v = iter.Next()
            i = i + 1
        return f0.getVal(x)

    # To calculate the ratio between single functions
    def __regFunc(self, x, f0, f1, val):
        iter = x.createIterator()
        v = iter.Next()
        i = 0
        while v:
            v.setVal(val[i])
            v = iter.Next()
            i = i + 1
        if (f0.getVal(x) + f1.getVal(x)) == 0.:
            return 0.
        return f1.getVal(x) / (f0.getVal(x) + f1.getVal(x))

    def evaluateDecomposedRatio(
            self,
            w,
            evalData,
            x=None,
            roc=False,
            c0arr=None,
            c1arr=None,
            true_dist=False,
            pre_evaluation=None,
            pre_dist=None,
            data_type='test',
            indexes=None,
            plotting=None):
        '''
          Compute composed ratio for dataset 'evalData'.
          Single ratios can be precomputed in pre_evaluation
        '''

        plotting = plotting if plotting <> None else self.verbose_printing
        if indexes is None:
            indexes = self.basis_indexes

        score = ROOT.RooArgSet(w.var('score'))
        npoints = evalData.shape[0]
        fullRatios = np.zeros(npoints)
        fullRatiosReal = np.zeros(npoints)
        c0arr = self.c0 if c0arr is None else c0arr
        c1arr = self.c1 if c1arr is None else c1arr

        true_score = []
        train_score = []
        all_targets = []
        all_positions = []
        all_ratios = []
        for k, c in enumerate(c0arr):
            innerRatios = np.zeros(npoints)
            innerTrueRatios = np.zeros(npoints)
            if c == 0:
                continue
            for j, c_ in enumerate(c1arr):
                index_k, index_j = (indexes[k], indexes[j])
                if index_k != index_j:
                    f0pdf = w.function(
                    'bkghistpdf_{0}_{1}'.format(
                        index_k, index_j))
                    f1pdf = w.function(
                    'sighistpdf_{0}_{1}'.format(
                        index_k, index_j))
                    if pre_evaluation is None:
                        traindata = evalData
                        outputs = predict(
                            '{0}/model/{1}/{2}/{3}_{4}_{5}.pkl'.format(
                                self.dir,
                                self.model_g,
                                self.c1_g,
                                self.model_file,
                                k,
                                j),
                            traindata,
                            model_g=self.model_g,
                            clf=self.clf)

                        f0pdfdist = np.array(
                            [self.evalDist(score, f0pdf, [xs]) for xs in outputs])
                        f1pdfdist = np.array(
                            [self.evalDist(score, f1pdf, [xs]) for xs in outputs])
                    else:
                        f0pdfdist = pre_evaluation[0][index_k][index_j]
                        f1pdfdist = pre_evaluation[1][index_k][index_j]
                    pdfratios = self.singleRatio(f0pdfdist, f1pdfdist)
                else:
                    pdfratios = np.ones(npoints)
                all_ratios.append(pdfratios)
                innerRatios += (c_ / c) * pdfratios
                if true_dist:
                    if pre_dist is None:
                        f0 = w.pdf('f{0}'.format(index_k))
                        f1 = w.pdf('f{0}'.format(index_j))
                        if len(evalData.shape) > 1:
                            f0dist = np.array(
                                [self.evalDist(x, f0pdf, xs) for xs in evalData])
                            f1dist = np.array(
                                [self.evalDist(x, f1pdf, xs) for xs in evalData])
                        else:
                            f0dist = np.array(
                                [self.evalDist(x, f0pdf, [xs]) for xs in evalData])
                            f1dist = np.array(
                                [self.evalDist(x, f1pdf, [xs]) for xs in evalData])
                    else:
                        f0dist = pre_dist[0][index_k][index_j]
                        f1dist = pre_dist[1][index_k][index_j]
                    ratios = self.singleRatio(f0dist, f1dist)
                    innerTrueRatios += (c_ / c) * ratios
                # ROC curves for pair-wise ratios
                if (roc or plotting) and k < j:
                    all_positions.append((k, j))
                    if roc:
                        if self.dataset_names is not None:
                            name_k, name_j = (
                                self.dataset_names[index_k], self.dataset_names[index_j])
                        else:
                            name_k, name_j = (index_k, index_j)
                        testdata, testtarget = loadData(data_type, name_k, name_j, dir=self.dir, c1_g=self.c1_g)
                    else:
                        testdata = evalData
                    size2 = testdata.shape[1] if len(testdata.shape) > 1 else 1
                    outputs = predict(
                        '{0}/model/{1}/{2}/{3}_{4}_{5}.pkl'.format(
                            self.dir,
                            self.model_g,
                            self.c1_g,
                            self.model_file,
                            k,
                            j),
                        testdata,
                        model_g=self.model_g,
                        clf=self.clf)
                    f0pdfdist = np.array(
                        [self.evalDist(score, f0pdf, [xs]) for xs in outputs])
                    f1pdfdist = np.array(
                        [self.evalDist(score, f1pdf, [xs]) for xs in outputs])
                    clfRatios = self.singleRatio(f0pdfdist, f1pdfdist)
                    train_score.append(clfRatios)
                    if roc:
                        all_targets.append(testtarget)
                    if true_dist:
                        if len(evalData.shape) > 1:
                            f0dist = np.array(
                                [self.evalDist(x, f0pdf, xs) for xs in testdata])
                            f1dist = np.array(
                                [self.evalDist(x, f1pdf, xs) for xs in testdata])
                        else:
                            f0dist = np.array(
                                [self.evalDist(x, f0pdf, [xs]) for xs in testdata])
                            f1dist = np.array(
                                [self.evalDist(x, f1pdf, [xs]) for xs in testdata])

                        trRatios = self.singleRatio(f0dist, f1dist)

                        true_score.append(trRatios)

            innerRatios = 1. / innerRatios
            innerRatios[np.abs(innerRatios) == np.inf] = 0.
            fullRatios += innerRatios
            if true_dist:
                innerTrueRatios = 1. / innerTrueRatios
                innerTrueRatios[np.abs(innerTrueRatios) == np.inf] = 0.
                fullRatiosReal += innerTrueRatios
        if roc:
            for ind in range(1, (len(train_score) / 3 + 1)):
                print_scores = train_score[(ind - 1) * 3:(ind - 1) * 3 + 3]
                print_targets = all_targets[(ind - 1) * 3:(ind - 1) * 3 + 3]
                print_positions = all_positions[
                    (ind - 1) * 3:(ind - 1) * 3 + 3]
                if true_dist:
                    makeMultiROC(
                        print_scores,
                        print_targets,
                        makePlotName(
                            'all{0}'.format(
                                ind - 1),
                            'comparison',
                            type='roc',
                            dir=self.dir,
                            model_g=self.model_g,
                            c1_g=self.c1_g),
                        dir=self.dir,
                        model_g=self.model_g,
                        true_score=true_score,
                        print_pdf=True,
                        title='ROC for pairwise trained classifier',
                        pos=print_positions)
                else:
                    makeMultiROC(
                        print_scores,
                        print_targets,
                        makePlotName(
                            'all{0}'.format(
                                ind - 1),
                            'comparison',
                            type='roc',
                            dir=self.dir,
                            model_g=self.model_g,
                            c1_g=self.c1_g),
                        dir=self.dir,
                        model_g=self.model_g,
                        print_pdf=True,
                        title='ROC for pairwise trained classifier',
                        pos=print_positions)

        return fullRatios, fullRatiosReal

    def findOutliers(self, x):
        q5, q95 = np.percentile(x, [5, 95])
        iqr = 2.0 * (q95 - q5)
        outliers = (x <= q95 + iqr) & (x >= q5 - iqr)
        return outliers

    def computeRatios(self, true_dist=False, vars_g=None,
                      data_file='test', use_log=False):
        '''
          Use the computed score densities to compute
          the decomposed ratio test.
          set true_dist to True if workspace have the true distributions to
          make plots, in that case vars_g also must be provided
          Final result is histogram for ratios and signal - bkf rejection curves
        '''

        f = ROOT.TFile('{0}/{1}'.format(self.dir, self.workspace))
        w = f.Get('w')
        f.Close()

        c1 = self.c1
        c0 = self.c0
        c1 = c1 / c1.sum()
        c0 = c0 / c0.sum()

        print 'Calculating ratios'

        npoints = 50

        if true_dist:
            vars = ROOT.TList()
            for var in vars_g:
                vars.Add(w.var(var))
            x = ROOT.RooArgSet(vars)

        if use_log:
            evaluateRatio = self.evaluateLogDecomposedRatio
            post = 'log'
        else:
            evaluateRatio = self.evaluateDecomposedRatio
            post = ''

        score = ROOT.RooArgSet(w.var('score'))
        scoref = ROOT.RooArgSet(w.var('scoref'))

        if use_log:
            getRatio = self.singleLogRatio
        else:
            getRatio = self.singleRatio

        # NN trained on complete model
        F0pdf = w.function('bkghistpdf_F0_F1')
        F1pdf = w.function('sighistpdf_F0_F1')

        # TODO Here assuming that signal is first dataset
        testdata, testtarget = loadData(
            data_file, self.F0_dist, 0, dir=self.dir, c1_g=self.c1_g)
        if len(vars_g) == 1:
            xarray = np.linspace(0, 5, npoints)
            fullRatios, _ = evaluateRatio(
                w, xarray, x=x, roc=False, true_dist=True)

            F1dist = np.array([self.evalDist(x, w.pdf('F1'), [xs])
                               for xs in xarray])
            F0dist = np.array([self.evalDist(x, w.pdf('F0'), [xs])
                               for xs in xarray])
            y2 = getRatio(F1dist, F0dist)

            # NN trained on complete model
            outputs = predict(
                '{0}/model/{1}/{2}/adaptive_F0_F1.pkl'.format(
                    self.dir, self.model_g, self.c1_g), xarray.reshape(
                    xarray.shape[0], 1), model_g=self.model_g, clf=self.clf)
            F1fulldist = np.array(
                [self.evalDist(scoref, F1pdf, [xs]) for xs in outputs])
            F0fulldist = np.array(
                [self.evalDist(scoref, F0pdf, [xs]) for xs in outputs])

            pdfratios = getRatio(F1fulldist, F0fulldist)

            saveFig(xarray,
                    [fullRatios,
                     y2,
                     pdfratios],
                    makePlotName('all',
                                 'train',
                                 type='ratio' + post),
                    title='Likelihood Ratios',
                    labels=['Approx. Decomposed',
                            'Exact',
                            'Approx.'],
                    print_pdf=True,
                    dir=self.dir)

        if true_dist:
            decomposedRatio, _ = evaluateRatio(
                w, testdata, x=x, roc=self.verbose_printing, true_dist=True)
        else:
            decomposedRatio, _ = evaluateRatio(
                w, testdata, c0arr=c0, c1arr=c1, roc=True, data_type=data_file)
        if len(testdata.shape) > 1:
            outputs = predict(
                '{0}/model/{1}/{2}/{3}_F0_F1.pkl'.format(
                    self.dir,
                    self.model_g,
                    self.c1_g,
                    self.model_file),
                testdata,
                model_g=self.model_g,
                clf=self.clf)

        else:
            outputs = predict(
                '{0}/model/{1}/{2}/{3}_F0_F1.pkl'.format(
                    self.dir, self.model_g, self.c1_g, self.model_file), testdata.reshape(
                    testdata.shape[0], 1), model_g=self.model_g, clf=self.clf)

        F1fulldist = np.array(
            [self.evalDist(scoref, F1pdf, [xs]) for xs in outputs])
        F0fulldist = np.array(
            [self.evalDist(scoref, F0pdf, [xs]) for xs in outputs])

        completeRatio = getRatio(F1fulldist, F0fulldist)
        if true_dist:
            if len(testdata.shape) > 1:
                F1dist = np.array([self.evalDist(x, w.pdf('F1'), xs)
                                   for xs in testdata])
                F0dist = np.array([self.evalDist(x, w.pdf('F0'), xs)
                                   for xs in testdata])
            else:
                F1dist = np.array(
                    [self.evalDist(x, w.pdf('F1'), [xs]) for xs in testdata])
                F0dist = np.array(
                    [self.evalDist(x, w.pdf('F0'), [xs]) for xs in testdata])

            realRatio = getRatio(F1dist, F0dist)

        decomposed_target = testtarget
        complete_target = testtarget
        real_target = testtarget

        # Removing outliers
        numtest = decomposedRatio.shape[0]

        if true_dist:
            real_outliers = np.zeros(numtest, dtype=bool)
            real_outliers = self.findOutliers(realRatio)


        all_ratios_plots = []
        all_names_plots = []
        bins = 70
        low = 0.6
        high = 1.2
        if use_log:
            low = -1.0
            high = 1.0
        low = []
        high = []
        low = []
        high = []
        ratios_vars = []
        for l, name in enumerate(['sig', 'bkg']):
            if true_dist:
                ratios_names = ['true', 'full', 'composed']
                ratios_vec = [realRatio, completeRatio, decomposedRatio]
                target_vec = [real_target, complete_target, decomposed_target]

                minimum = min([realRatio[real_target == 1 - l].min(),
                               completeRatio[complete_target == 1 - l].min(),
                               decomposedRatio[decomposed_target == 1 - l].min()])
                maximum = max([realRatio[real_target == 1 - l].max(),
                               completeRatio[complete_target == 1 - l].max(),
                               decomposedRatio[decomposed_target == 1 - l].max()])

            else:
                ratios_names = ['full', 'composed']
                ratios_vec = [completeRatio, decomposedRatio]
                target_vec = [complete_target, decomposed_target]
                minimum = min([completeRatio[complete_target == 1 - l].min(),
                               decomposedRatio[decomposed_target == 1 - l].min()])
                maximum = max([completeRatio[complete_target == 1 - l].max(),
                               decomposedRatio[decomposed_target == 1 - l].max()])

            low.append(minimum - ((maximum - minimum) / bins) * 10)
            high.append(maximum + ((maximum - minimum) / bins) * 10)
            w.factory('ratio{0}[{1},{2}]'.format(name, low[l], high[l]))
            ratios_vars.append(w.var('ratio{0}'.format(name)))
        for curr, curr_ratios, curr_targets in zip(
                ratios_names, ratios_vec, target_vec):
            numtest = curr_ratios.shape[0]
            for l, name in enumerate(['sig', 'bkg']):
                hist = ROOT.TH1F(
                    '{0}_{1}hist_F0_f0'.format(
                        curr, name), 'hist', bins, low[l], high[l])
                for val in curr_ratios[curr_targets == 1 - l]:
                    hist.Fill(val)
                datahist = ROOT.RooDataHist(
                    '{0}_{1}datahist_F0_f0'.format(
                        curr, name), 'hist', ROOT.RooArgList(
                        ratios_vars[l]), hist)
                ratios_vars[l].setBins(bins)
                histpdf = ROOT.RooHistFunc(
                    '{0}_{1}histpdf_F0_f0'.format(
                        curr, name), 'hist', ROOT.RooArgSet(
                        ratios_vars[l]), datahist, 0)

                histpdf.specialIntegratorConfig(
                    ROOT.kTRUE).method1D().setLabel('RooBinIntegrator')
                getattr(w, 'import')(hist)
                getattr(w, 'import')(datahist)
                getattr(w, 'import')(histpdf)
                if name == 'bkg':
                    all_ratios_plots.append([w.function('{0}_sighistpdf_F0_f0'.format(
                        curr)), w.function('{0}_bkghistpdf_F0_f0'.format(curr))])
                    all_names_plots.append(
                        ['sig_{0}'.format(curr), 'bkg_{0}'.format(curr)])

        all_ratios_plots = [[all_ratios_plots[j][i] for j, _ in enumerate(
            all_ratios_plots)] for i, _ in enumerate(all_ratios_plots[0])]
        all_names_plots = [[all_names_plots[j][i] for j, _ in enumerate(
            all_names_plots)] for i, _ in enumerate(all_names_plots[0])]

        printMultiFrame(w,
                        ['ratiosig',
                         'ratiobkg'],
                        all_ratios_plots,
                        makePlotName('ratio',
                                     'comparison',
                                     type='hist' + post,
                                     dir=self.dir,
                                     model_g=self.model_g,
                                     c1_g=self.c1_g),
                        all_names_plots,
                        setLog=True,
                        dir=self.dir,
                        model_g=self.model_g,
                        y_text='Count',
                        title='Histograms for ratios',
                        x_text='ratio value',
                        print_pdf=True)


        if use_log:
            decomposedRatio = np.exp(decomposedRatio)
            completeRatio = np.exp(completeRatio)
            if true_dist:
                realRatio = np.exp(realRatio)
        if true_dist:
            ratios_list = [decomposedRatio / decomposedRatio.max(),
                           completeRatio / completeRatio.max(),
                           realRatio / realRatio.max()]
            targets_list = [decomposed_target, complete_target, real_target]
            legends_list = ['Approx. Decomposed', 'Approx.', 'Exact']
        else:

            indices = (decomposedRatio > 0.)
            decomposedRatio = decomposedRatio[indices]
            decomposed_target = decomposed_target[indices]
            indices = (completeRatio > 0.)
            completeRatio = completeRatio[indices]
            complete_target = complete_target[indices]

            completeRatio = np.log(completeRatio)
            decomposedRatio = np.log(decomposedRatio)
            decomposedRatio = decomposedRatio + np.abs(decomposedRatio.min())
            completeRatio = completeRatio + np.abs(completeRatio.min())
            ratios_list = [decomposedRatio / decomposedRatio.max(),
                           completeRatio / completeRatio.max()]
            targets_list = [decomposed_target, complete_target]
            legends_list = ['Approx. Decomposed', 'Approx.']
        makeSigBkg(
            ratios_list,
            targets_list,
            makePlotName(
                'comp',
                'all',
                type='sigbkg' + post,
                dir=self.dir,
                model_g=self.model_g,
                c1_g=self.c1_g),
            dir=self.dir,
            model_g=self.model_g,
            print_pdf=True,
            legends=legends_list,
            title='Signal-Background rejection curves')


    def evalC1C2Likelihood(
            self,
            c0,
            c1,
            dir='/afs/cern.ch/user/j/jpavezse/systematics',
            workspace='workspace_DecomposingTestOfMixtureModelsClassifiers.root',
            c1_g='',
            model_g='mlp',
            use_log=False,
            true_dist=False,
            vars_g=None,
            clf=None,
            verbose_printing=False):

        f = ROOT.TFile('{0}/{1}'.format(dir, workspace))
        w = f.Get('w')
        f.Close()
        if true_dist:
            vars = ROOT.TList()
            for var in vars_g:
                vars.Add(w.var(var))
            x = ROOT.RooArgSet(vars)
        else:
            x = None

        score = ROOT.RooArgSet(w.var('score'))
        if use_log:
            evaluateRatio = self.evaluateLogDecomposedRatio
            post = 'log'
        else:
            evaluateRatio = self.evaluateDecomposedRatio
            post = ''

        npoints = 25
        csarray = np.linspace(0.01, 0.2, npoints)
        cs2array = np.linspace(0.1, 0.4, npoints)
        testdata = np.loadtxt(
            '{0}/data/{1}/{2}/{3}_{4}.dat'.format(dir, model_g, c1_g, 'test', 'F1'))

        decomposedLikelihood = np.zeros((npoints, npoints))
        trueLikelihood = np.zeros((npoints, npoints))
        c1s = np.zeros(c1.shape[0])
        c0s = np.zeros(c1.shape[0])
        pre_pdf = []
        pre_dist = []
        pre_pdf.extend([[], []])
        pre_dist.extend([[], []])
        for k, c0_ in enumerate(c0):
            pre_pdf[0].append([])
            pre_pdf[1].append([])
            pre_dist[0].append([])
            pre_dist[1].append([])
            for j, c1_ in enumerate(c1):
                if k != j:
                    f0pdf = w.function('bkghistpdf_{0}_{1}'.format(k, j))
                    f1pdf = w.function('sighistpdf_{0}_{1}'.format(k, j))
                    outputs = predict(
                        '{0}/model/{1}/{2}/{3}_{4}_{5}.pkl'.format(
                            dir,
                            model_g,
                            c1_g,
                            'adaptive',
                            k,
                            j),
                        testdata,
                        model_g=model_g,
                        clf=clf)
                    f0pdfdist = np.array(
                        [self.evalDist(score, f0pdf, [xs]) for xs in outputs])
                    f1pdfdist = np.array(
                        [self.evalDist(score, f1pdf, [xs]) for xs in outputs])
                    pre_pdf[0][k].append(f0pdfdist)
                    pre_pdf[1][k].append(f1pdfdist)
                else:
                    pre_pdf[0][k].append(None)
                    pre_pdf[1][k].append(None)
                if true_dist:
                    f0 = w.pdf('f{0}'.format(k))
                    f1 = w.pdf('f{0}'.format(j))
                    if len(testdata.shape) > 1:
                        f0dist = np.array([self.evalDist(x, f0, xs)
                                           for xs in testdata])
                        f1dist = np.array([self.evalDist(x, f1, xs)
                                           for xs in testdata])
                    else:
                        f0dist = np.array([self.evalDist(x, f0, [xs])
                                           for xs in testdata])
                        f1dist = np.array([self.evalDist(x, f1, [xs])
                                           for xs in testdata])
                    pre_dist[0][k].append(f0dist)
                    pre_dist[1][k].append(f1dist)

        # Evaluate Likelihood in different c1[0] and c1[1] values
        self
        for i, cs in enumerate(csarray):
            for j, cs2 in enumerate(cs2array):
                c1s[:] = c1[:]
                c1s[0] = cs
                c1s[1] = cs2
                c1s[2] = 1. - cs - cs2
                decomposedRatios, trueRatios = evaluateRatio(w, testdata,
                                                             x=x, roc=False, plotting=False,
                                                             c0arr=c0, c1arr=c1s, true_dist=true_dist,
                                                             pre_evaluation=pre_pdf,
                                                             pre_dist=pre_dist)

                if not use_log:
                    decomposedLikelihood[i, j] = np.log(decomposedRatios).sum()
                    trueLikelihood[i, j] = np.log(trueRatios).sum()
                else:
                    decomposedLikelihood[i, j] = decomposedRatios.sum()
                    trueLikelihood[i, j] = trueRatios.sum()
        decomposedLikelihood = 2. * decomposedLikelihood
        if true_dist:
            trueLikelihood = 2. * trueLikelihood
        decomposedLikelihood = decomposedLikelihood - decomposedLikelihood.min()
        X, Y = np.meshgrid(csarray, cs2array)
        decMin = np.unravel_index(
            decomposedLikelihood.argmin(),
            decomposedLikelihood.shape)
        min_value = [csarray[decMin[0]], cs2array[decMin[1]]]
        if true_dist:
            trueLikelihood = trueLikelihood - trueLikelihood.min()
            trueMin = np.unravel_index(
                trueLikelihood.argmin(),
                trueLikelihood.shape)
            if verbose_printing:
                saveFig(X,
                        [Y,
                         decomposedLikelihood,
                         trueLikelihood],
                        makePlotName('comp',
                                     'train',
                                     type='multilikelihood'),
                        labels=['Approx. Decomposed',
                                'Exact'],
                        type='contour2',
                        marker=True,
                        dir=dir,
                        marker_value=(c1[0],
                                      c1[1]),
                        print_pdf=True,
                        min_value=min_value)
            return [[csarray[trueMin[0]], cs2array[trueMin[1]]],
                    [csarray[decMin[0]], cs2array[decMin[1]]]]
        else:
            return [[0., 0.], [csarray[decMin[0]], cs2array[decMin[1]]]]


    def fitCValues(
            self,
            c0,
            c1,
            dir='/afs/cern.ch/user/j/jpavezse/systematics',
            c1_g='',
            model_g='mlp',
            true_dist=False,
            vars_g=None,
            workspace='workspace_DecomposingTestOfMixtureModelsClassifiers.root',
            use_log=False,
            clf=None,
            n_hist_c=100):
        if use_log:
            post = 'log'
        else:
            post = ''
        keys = ['true', 'dec']
        c1_ = dict((key, np.zeros(n_hist_c)) for key in keys)
        c1_values = dict((key, np.zeros(n_hist_c)) for key in keys)
        c2_values = dict((key, np.zeros(n_hist_c)) for key in keys)

        fil2 = open('{0}/fitting_values_c1c2{1}.txt'.format(dir, post), 'w')

        for i in range(n_hist_c):

            makeData(vars_g, c0, c1, num_train=200000, num_test=500, no_train=True,
                     workspace=workspace, dir=dir, c1_g=c1_g)

            if i == 0:
                verbose_printing = True
            else:
                verbose_printing = False

            ((c1_true,
              c2_true),
             (c1_dec,
              c2_dec)) = self.evalC1C2Likelihood(c0,
                                            c1,
                                            dir=dir,
                                            c1_g=c1_g,
                                            model_g=model_g,
                                            true_dist=true_dist,
                                            vars_g=vars_g,
                                            workspace=workspace,
                                            use_log=use_log,
                                            clf=clf,
                                            verbose_printing=verbose_printing)
            print '2: {0} {1} {2} {3}'.format(c1_true, c1_dec, c2_true, c2_dec)

            fil2.write(
                '{0} {1} {2} {3}\n'.format(
                    c1_true,
                    c1_dec,
                    c2_true,
                    c2_dec))

        fil2.close()
