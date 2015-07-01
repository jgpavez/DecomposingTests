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
import pylab as plt

from mlp import make_predictions, train_mlp

''' 
 A simple example for the work on the section 
 5.4 of the paper 'Approximating generalized 
 likelihood ratio test with calibrated discriminative
 classifiers' by Kyle Cranmer
''' 

# Constants for each different model
c0 = np.array([.0,.3, .7])
c1 = np.array([.1,.3, .7])
c1_g = ''
#c1 = c1 / c1.sum()

#c1 = [.1,.5, .4]
verbose_printing = True
model_g = None
dir = '/afs/cern.ch/user/j/jpavezse/systematics'

vars_g = ['x0','x1','x2','x3','x4','x5','x6','x7','x8','x9']
mu1_g = [5.,5.,4.,3.,5.,5.,4.5,2.5,4.,3.5]
mu2_g = [2.,4.5,0.6,5.,6.,4.5,4.2,0.2,4.1,3.3]
cov1_g =[[3.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
         [0.,2.,0.,0.,0.,0.,0.,0.,0.,0.],
         [0.,0.,14.,0.,0.,0.,0.,0.,0.,0.],
         [0.,0.,0.,6.,0.,0.,0.,0.,0.,0.],
         [0.,0.,0.,0.,17.,0.,0.,0.,0.,0.],
         [0.,0.,0.,0.,0.,10.,0.,0.,0.,0.],
         [0.,0.,0.,0.,0.,0.,5.,0.,0.,0.],
         [0.,0.,0.,0.,0.,0.,0.,1.3,0.,0.],
         [0.,0.,0.,0.,0.,0.,0.,0.,1.,0.],
         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.3]]
cov2_g =[[3.5,0.,0.,0.,0.,0.,0.,0.,0.,0.],
         [0.,3.5,0.,0.,0.,0.,0.,0.,0.,0.],
         [0.,0.,3.5,0.,0.,0.,0.,0.,0.,0.],
         [0.,0.,0.,7.2,0.,0.,0.,0.,0.,0.],
         [0.,0.,0.,0.,4.5,0.,0.,0.,0.,0.],
         [0.,0.,0.,0.,0.,3.5,0.,0.,0.,0.],
         [0.,0.,0.,0.,0.,0.,8.2,0.,0.,0.],
         [0.,0.,0.,0.,0.,0.,0.,9.5,0.,0.],
         [0.,0.,0.,0.,0.,0.,0.,0.,0.5,0.],
         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.5]]

mu3_g = [1.,0.5,0.3,0.5,0.6,0.4,0.1,0.2,0.1,0.3]
cov3_g =[[13.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
         [0.,12.,0.,0.,0.,0.,0.,0.,0.,0.],
         [0.,0.,14.,0.,0.,0.,0.,0.,0.,0.],
         [0.,0.,0.,6.,0.,0.,0.,0.,0.,0.],
         [0.,0.,0.,0.,1.,0.,0.,0.,0.,0.],
         [0.,0.,0.,0.,0.,10.,0.,0.,0.,0.],
         [0.,0.,0.,0.,0.,0.,15.,0.,0.,0.],
         [0.,0.,0.,0.,0.,0.,0.,6.3,0.,0.],
         [0.,0.,0.,0.,0.,0.,0.,0.,11.,0.],
         [0.,0.,0.,0.,0.,0.,0.,0.,0.,1.3]]

'''
mu1_g = [5.,3.8,4.2,4.,5.,5.,4.5,3.5,5.,3.5]
mu2_g = [2.,3.5,2.6,4.,2.,3.5,2.2,4.2,3.1,2.3]
mu3_g = [-2.,-0.5,-2.3,-1.3,-3.6,-2.4,-1.1,-2.2,-3.1,-1.3]
cov1_g =[[3.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
         [0.,2.,0.,0.,0.,0.,0.,0.,0.,0.],
         [0.,0.,5.,0.,0.,0.,0.,0.,0.,0.],
         [0.,0.,0.,6.,0.,0.,0.,0.,0.,0.],
         [0.,0.,0.,0.,11.,0.,0.,0.,0.,0.],
         [0.,0.,0.,0.,0.,10.,0.,0.,0.,0.],
         [0.,0.,0.,0.,0.,0.,5.,0.,0.,0.],
         [0.,0.,0.,0.,0.,0.,0.,1.3,0.,0.],
         [0.,0.,0.,0.,0.,0.,0.,0.,1.,0.],
         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.3]]
cov2_g =[[3.5,0.,0.,0.,0.,0.,0.,0.,0.,0.],
         [0.,3.5,0.,0.,0.,0.,0.,0.,0.,0.],
         [0.,0.,3.5,0.,0.,0.,0.,0.,0.,0.],
         [0.,0.,0.,7.2,0.,0.,0.,0.,0.,0.],
         [0.,0.,0.,0.,4.5,0.,0.,0.,0.,0.],
         [0.,0.,0.,0.,0.,3.5,0.,0.,0.,0.],
         [0.,0.,0.,0.,0.,0.,5.2,0.,0.,0.],
         [0.,0.,0.,0.,0.,0.,0.,4.5,0.,0.],
         [0.,0.,0.,0.,0.,0.,0.,0.,0.5,0.],
         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.5]]

cov3_g =[[13.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
         [0.,12.,0.,0.,0.,0.,0.,0.,0.,0.],
         [0.,0.,14.,0.,0.,0.,0.,0.,0.,0.],
         [0.,0.,0.,6.,0.,0.,0.,0.,0.,0.],
         [0.,0.,0.,0.,5.,0.,0.,0.,0.,0.],
         [0.,0.,0.,0.,0.,10.,0.,0.,0.,0.],
         [0.,0.,0.,0.,0.,0.,15.,0.,0.,0.],
         [0.,0.,0.,0.,0.,0.,0.,9.3,0.,0.],
         [0.,0.,0.,0.,0.,0.,0.,0.,11.,0.],
         [0.,0.,0.,0.,0.,0.,0.,0.,0.,10.3]]

'''

def printMultiFrame(w,obs,all_pdfs,name,legends,setLog=False):
  '''
    This just print a bunch of pdfs 
    in a Canvas
  ''' 

  # Hope I don't need more colors ...
  colors = [ROOT.kBlack,ROOT.kRed,ROOT.kBlue,ROOT.kGreen,
    ROOT.kYellow]
  style = [ROOT.kSolid,ROOT.kSolid,ROOT.kDashed,ROOT.kDashed]
  can = ROOT.TCanvas('c1')
  can.Divide(1,len(all_pdfs))
  x = w.var(obs)
  for curr,pdf in enumerate(all_pdfs): 
    can.cd(curr+1)
    if setLog == True:
      ROOT.gPad.SetLogy(1)
    funcs = []
    line_colors = []
    line_styles = []
    for i,p in enumerate(pdf):
      funcs.append(p)
      line_colors.append(ROOT.RooFit.LineColor(colors[i]))
      line_styles.append(ROOT.RooFit.LineStyle(style[i]))
    frame = x.frame(ROOT.RooFit.Name(legends[curr][0]),ROOT.RooFit.
        Title(legends[curr][0].split('_')[0]))
    for i,f in enumerate(funcs):
      if isinstance(f,str):
        funcs[0].plotOn(frame, ROOT.RooFit.Components(f),ROOT.RooFit.Name(legends[curr][i]), line_colors[i],
        line_styles[i])
      else:
        f.plotOn(frame,ROOT.RooFit.Name(legends[curr][i]),line_colors[i],line_styles[i])
    leg = ROOT.TLegend(0.65, 0.73, 0.86, 0.87)
    #leg.SetFillColor(ROOT.kWhite)
    #leg.SetLineColor(ROOT.kWhite)
    # TODO This is just a quick fix because is now working how it should
    for i,l in enumerate(legends[curr]):
      if i == 0:
        leg.AddEntry(frame.findObject(legends[curr][i]), l.split('_')[1], 'l')
      else:
        leg.AddEntry(frame.findObject(legends[curr][i]), l.split('_')[1], 'l')
    
    frame.Draw()
    leg.Draw()
    ROOT.gPad.Update()
    can.Modified()
    can.Update()
  can.SaveAs('{0}/plots/{1}/{2}.png'.format(dir,model_g,name))

def printFrame(w,obs,pdf,name,legends):
  '''
    This just print a bunch of pdfs 
    in a Canvas
  ''' 

  # Hope I don't need more colors ...
  colors = [ROOT.kBlack,ROOT.kRed,ROOT.kGreen,ROOT.kBlue,
    ROOT.kYellow]
  x = []
  for var in obs:
    x.append(w.var(var))
  funcs = []
  line_colors = []
  for i,p in enumerate(pdf):
    funcs.append(p)
    line_colors.append(ROOT.RooFit.LineColor(colors[i]))
  
  can = ROOT.TCanvas('c1')
  can.Divide(1,len(obs))
  frame = []
  for var in x:
    frame.append(var.frame())
  for j,fra in enumerate(frame):    
    can.cd(j+1)
    for i,f in enumerate(funcs):
        if isinstance(f,str):
          funcs[0].plotOn(fra, ROOT.RooFit.Components(f),ROOT.RooFit.Name(legends[i]), line_colors[i])
        else:
          f.plotOn(fra,ROOT.RooFit.Name(legends[i]),line_colors[i])
    leg = ROOT.TLegend(0.65, 0.73, 0.86, 0.87)
    #leg.SetFillColor(ROOT.kWhite)
    #leg.SetLineColor(ROOT.kWhite)
    for i,l in enumerate(legends):
      if i == 0:
        leg.AddEntry(fra.findObject(legends[i]), l, 'l')
      else:
        leg.AddEntry(fra.findObject(legends[i]), l, 'l')
    fra.Draw()
    leg.Draw()
  can.SaveAs('{0}/plots/{1}/{2}.png'.format(dir,model_g,name))

def makeModelND():
  '''
  RooFit statistical model for the data
  
  '''  
  # Statistical model
  w = ROOT.RooWorkspace('w')
  vars_string = ','.join(['{0}[0,5]'.format(var) for var in vars_g]) 
  gaus_vars_string = ','.join(vars_g)
  exponential_string = '+'.join(vars_g) 

  print 'Generating initial distributions'

  # Gaussian 1
  cov1_m = np.matrix(cov1_g)
  #cov1_m = np.dot(cov1_m,cov1_m.transpose()) 
  cov1 = ROOT.TMatrixDSym(len(vars_g))
  for i,var1 in enumerate(vars_g):
    for j,var2 in enumerate(vars_g):
      cov1[i][j] = cov1_m[i,j]
  getattr(w,'import')(cov1,'cov1')
  cov2_m = np.matrix(cov2_g)
  #cov2_m = np.dot(cov2_m,cov2_m.transpose())
  cov2 = ROOT.TMatrixDSym(len(vars_g))
  for i,var1 in enumerate(vars_g):
    for j,var2 in enumerate(vars_g):
      print cov2_m[i,j],
      cov2[i][j] = cov2_m[i,j]
    print
  getattr(w,'import')(cov2,'cov2')

  mu1 = ','.join([str(mu) for mu in mu1_g])
  mu2 = ','.join([str(mu) for mu in mu2_g])
  
  cov3_m = np.matrix(cov3_g)
  cov3 = ROOT.TMatrixDSym(len(vars_g))
  for i,var1 in enumerate(vars_g):
    for j,var2 in enumerate(vars_g):
      cov3[i][j] = cov3_m[i,j]
  getattr(w,'import')(cov3,'cov3')

  mu3 = ','.join([str(mu) for mu in mu3_g])

  #w.factory("EXPR::f2('exp(-0.4*({0}))',{1})".format(exponential_string,vars_string))
  # Making Multi Var Gaussian, factory not working
  #w.factory("MultiVarGaussian::f1({{{0}}},{{{1}}},cov1)".format(gaus_vars_string,mu1))
  #w.factory("MultiVarGaussian::f0({{{0}}},{{{1}}},cov2)".format(gaus_vars_string,mu2))
  argus = ROOT.RooArgList() 
  for i,var in enumerate(vars_g):
    w.factory('{0}[{1},{2}]'.format(var,0,5))
    argus.add(w.var(var))

  vec1 = ROOT.TVectorD(len(vars_g))
  for i,mu in enumerate(mu1_g):
    vec1[i] = mu
  gaussian1 = ROOT.RooMultiVarGaussian('f1','f1',argus,vec1,cov1)
  getattr(w,'import')(gaussian1)

  vec2 = ROOT.TVectorD(len(vars_g))
  for i,mu in enumerate(mu2_g):
    vec2[i] = mu
  gaussian2 = ROOT.RooMultiVarGaussian('f0','f0',argus,vec2,cov2)
  getattr(w,'import')(gaussian2)

  vec3 = ROOT.TVectorD(len(vars_g))
  for i,mu in enumerate(mu3_g):
    vec3[i] = mu
  gaussian3 = ROOT.RooMultiVarGaussian('f2','f2',argus,vec3,cov3)
  getattr(w,'import')(gaussian3)

  w.factory("SUM::F0(c00[{0}]*f0,c01[{1}]*f1,f2)".format(c0[0],c0[1]))
  w.factory("SUM::F1(c10[{0}]*f0,c11[{1}]*f1,f2)".format(c1[0],c1[1]))
  
  # Check Model
  w.Print()

  w.writeToFile('{0}/workspace_DecomposingTestOfMixtureModelsClassifiers.root'.format(dir))
  if verbose_printing == True:
    printFrame(w,vars_g,[w.pdf('f0'),w.pdf('f1'),w.pdf('f2')],'decomposed_model',['f0','f1','f2']) 
    printFrame(w,vars_g,[w.pdf('F0'),w.pdf('F1')],'full_model',['F0','F1'])
    printFrame(w,vars_g,[w.pdf('F1'),'f0'],'full_signal', ['F1','f0'])


def makeModel2D():
  '''
  RooFit statistical model for the data
  
  '''  
  # Statistical model
  w = ROOT.RooWorkspace('w')
  #w.factory("EXPR::f1('cos(x)**2 + .01',x)")
  w.factory("EXPR::f2('exp(-0.4*(x+y))',x[0,5],y[0,5])")
  w.factory("EXPR::f1('0.3 + exp(-(x-5)**2/3.)+exp(-(y-3)**2/2.)',x,y)")
  w.factory("EXPR::f0('exp(-(x-2.5)**2/0.5)+exp(-(y-2)**2/1.5)',x,y)")
  #w.factory("EXPR::f2('exp(x*-1)',x[0,5])")
  #w.factory("EXPR::f1('0.3 + exp(-(x-5)**2/5.)',x)")
  #w.factory("EXPR::f0('exp(-(x-2.5)**2/1.)',x)")
  #w.factory("EXPR::f2('exp(-(x-2)**2/2)',x)")
  w.factory("SUM::F0(c00[{0}]*f0,c01[{1}]*f1,f2)".format(c0[0],c0[1]))
  w.factory("SUM::F1(c10[{0}]*f0,c11[{1}]*f1,f2)".format(c1[0],c1[1]))
  
  # Check Model
  w.Print()
  w.writeToFile('{0}/workspace_DecomposingTestOfMixtureModelsClassifiers.root'.format(dir))
  if verbose_printing == True:
    printFrame(w,['x','y'],[w.pdf('f0'),w.pdf('f1'),w.pdf('f2')],'decomposed_model',['f0','f1','f2']) 
    printFrame(w,['x','y'],[w.pdf('F0'),w.pdf('F1')],'full_model',['F0','F1'])
    printFrame(w,['x','y'],[w.pdf('F1'),'f0'],'full_signal', ['F1','f0'])


def makeData(num_train=500,num_test=100):
  # Start generating data
  ''' 
    Each function will be discriminated pair-wise
    so n*n datasets are needed (maybe they can be reused?)
  ''' 

  f = ROOT.TFile('{0}/workspace_DecomposingTestOfMixtureModelsClassifiers.root'.format(dir))
  w = f.Get('w')
  f.Close()

  print 'Making Data'
  # Start generating data
  ''' 
    Each function will be discriminated pair-wise
    so n*n datasets are needed (maybe they can be reused?)
  ''' 
   
  def makeDataFi(x, pdf, num):
    traindata = np.zeros((num,len(vars_g))) 
    data = pdf.generate(x,num)
    traindata[:] = [[data.get(i).getRealValue(var) for var in vars_g]
        for i in range(num)]
    return traindata

  vars = ROOT.TList()
  for var in vars_g:
    vars.Add(w.var(var))
  x = ROOT.RooArgSet(vars)

  for k,c in enumerate(c0):
    print 'Making {0}'.format(k)
    traindata = makeDataFi(x,w.pdf('f{0}'.format(k)), num_train)
    np.savetxt('{0}/data/{1}/{2}/train_{3}.dat'.format(dir,'mlp',c1_g,k),
                      traindata,fmt='%f')
    testdata = makeDataFi(x, w.pdf('f{0}'.format(k)), num_test)
    np.savetxt('{0}/data/{1}/{2}/test_{3}.dat'.format(dir,'mlp',c1_g,k),
                      testdata,fmt='%f')

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



def loadData(type,k,j,folder=None):
  if folder <> None:
    fk = np.loadtxt('{0}/{1}_{2}.dat'.format(folder,type,k))
    fj = np.loadtxt('{0}/{1}_{2}.dat'.format(folder,type,j))
  else:
    fk = np.loadtxt('{0}/data/{1}/{2}/{3}_{4}.dat'.format(dir,'mlp',c1_g,type,k))
    fj = np.loadtxt('{0}/data/{1}/{2}/{3}_{4}.dat'.format(dir,'mlp',c1_g,type,j))
  num = fk.shape[0]
  traindata = np.zeros((num*2,fk.shape[1]))
  targetdata = np.zeros(num*2)
  traindata[:num] = fj[:]
  traindata[num:] = fk[:]
  targetdata[:num].fill(1)
  targetdata[num:].fill(0)
  return (traindata, targetdata)

def predict(filename, traindata):
  sig = 1
  sfilename,k,j = filename.split('_')
  j = j.split('.')[0]
  sig = 1
  if k <> 'F0':
    k = int(k)
    j = int(j)
    sig = 1 if k < j else 0
    filename = '{0}_{1}_{2}.pkl'.format(sfilename,min(k,j),max(k,j))
  if model_g == 'mlp':
    return make_predictions(dataset=traindata, model_file=filename)[:,sig]
  else:
    clf = joblib.load(filename)
    if clf.__class__.__name__ == 'NuSVR':
      output = clf.predict(traindata)
      return np.clip(output,0.,1.)
    else:
      return clf.predict_proba(traindata)[:,sig]


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
  np.savetxt('{0}/plots/{1}/results/{2}.txt'.format(dir,model_g,label),np.column_stack((fpr,tpr)))
  plt.savefig('{0}/plots/{1}/{2}.png'.format(dir,model_g,label))
  plt.close(fig)
  plt.clf()


def makeSigBkg(outputs, target, label):
  '''
    make plots for ROC curve of classifier and 
    test data.
  '''
  #fpr, tpr, _  = roc_curve(target.ravel(),outputs.ravel())
  #tnr = 1. - fpr
  fnr, tnr, thresholds = roc_curve(1.-target.ravel(), outputs.ravel())
  fnr = np.array([float(np.sum((outputs > tr) * (target == 0)))/float(np.sum(target == 0)) for tr in thresholds])
  fnr = fnr.ravel()
  tpr = np.array([float(np.sum((outputs < tr) * (target == 1)))/float(np.sum(target == 1)) for tr in thresholds])
  tpr = tpr.ravel()
  roc_auc = auc(tpr,fnr)
  fig = plt.figure()
  plt.plot(tpr, fnr, label='Signal Eff Bkg Rej (area = %0.2f)' % roc_auc)
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('Signal Efficiency')
  plt.ylabel('Background Rejection')
  plt.title('{0}'.format(label))
  plt.legend(loc="lower right")
  np.savetxt('{0}/plots/{1}/results/{2}.txt'.format(dir,model_g,label),np.column_stack((tpr,fnr)))
  plt.savefig('{0}/plots/{1}/{2}.png'.format(dir,model_g,label))
  plt.close(fig)
  plt.clf()

def makePlotName(full, truth, f0 = None, f1 = None, type=None):
  if full == 'dec':
    return '{0}_{1}_f{2}_f{3}_{4}_{5}'.format(full, truth, f0, f1, model_g,type)
  else:
    return '{0}_{1}_{2}_{3}'.format(full, truth, model_g,type)

def trainClassifier(clf):
  '''
    Train classifiers pair-wise on 
    datasets
  '''

  print 'Training classifier'

  for k,c in enumerate(c0):
    for j,c_ in enumerate(c1):
      if k==j or k > j:
        continue
      #if k == j:
      print " Training Classifier on f{0}/f{1}".format(k,j)
      #clf = svm.NuSVC(probability=True) #Why use a SVR??
      if model_g == 'mlp':
        train_mlp(datatype='train',kpos=k,jpos=j,dir='{0}/data/{1}/{2}'.format(dir,'mlp',c1_g),
          save_file='{0}/model/{1}/{2}/adaptive_{3}_{4}.pkl'.format(dir,model_g,c1_g,k,j))
      else:
        traindata,targetdata = loadData('train',k,j) 
        clf.fit(traindata.reshape(traindata.shape[0],traindata.shape[1])
            ,targetdata)
        joblib.dump(clf, '{0}/model/{1}/{2}/adaptive_{3}_{4}.pkl'.format(dir,model_g,c1_g,k,j))
      #makeROC(outputs, testtarget, makePlotName('decomposed','trained',k,j,'roc'))
  
  print " Training Classifier on F0/F1"
  if model_g == 'mlp':
    train_mlp(datatype='train',kpos='F0',jpos='F1',dir='{0}/data/{1}/{2}'.format(dir,'mlp',c1_g), 
        save_file='{0}/model/{1}/{2}/adaptive_F0_F1.pkl'.format(dir,model_g,c1_g))
  else:
    traindata,targetdata = loadData('train','F0','F1') 
    #clf = svm.NuSVC(probability=True) #Why use a SVR??
    clf.fit(traindata.reshape(traindata.shape[0],traindata.shape[1])
        ,targetdata)
    joblib.dump(clf, '{0}/model/{1}/{2}/adaptive_F0_F1.pkl'.format(dir,model_g,c1_g))

  #testdata, testtarget = loadData('data/{0}/testdata_F0_F1.dat'.format(model_g)) 
  #outputs = predict(clf,testdata.reshape(testdata.shape[0],1))
  #makeROC(outputs, testtarget, makePlotName('full','trained',type='roc'))

def classifierPdf():
  ''' 
    Create pdfs for the classifier 
    score to be used later on the ratio 
    test
  '''

  bins = 40
  low = -7.
  high = 7.  

  f = ROOT.TFile('{0}/workspace_DecomposingTestOfMixtureModelsClassifiers.root'.format(dir))
  w = f.Get('w')
  f.Close()
  
  print 'Generating Score Histograms'

  w.factory('score[{0},{1}]'.format(low,high))
  s = w.var('score')
  
  #This is because most of the data of the full model concentrate around 0 
  bins_full = 40
  low_full = -1.0
  high_full = 1.0
  w.factory('scoref[{0},{1}]'.format(low_full, high_full))
  s_full = w.var('scoref')
  histos = []
  histos_names = []

  def saveHistos(w,ouputs,s,bins,low,high,pos=None):
    numtrain = outputs.shape[0]
    if pos <> None:
      k,j = pos
    else:
      k,j = ('F0','F1')
    for l,name in enumerate(['sig','bkg']):
      data = ROOT.RooDataSet('{0}data_{1}_{2}'.format(name,k,j),"data",
          ROOT.RooArgSet(s))
      #low = outputs.min()
      #high = outputs.max() 
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
    
      histpdf.specialIntegratorConfig(ROOT.kTRUE).method1D().setLabel('RooBinIntegrator')

      getattr(w,'import')(hist)
      getattr(w,'import')(data)
      getattr(w,'import')(datahist) # work around for morph = w.import(morph)
      getattr(w,'import')(histpdf) # work around for morph = w.import(morph)

      score_str = 'scoref' if pos == None else 'score'
      #w.factory('KeysPdf::{0}dist_{1}_{2}({3},{0}data_{1}_{2},RooKeysPdf::NoMirror,2)'.format(name,k,j,score_str))

      # Calculate the density of the classifier output using kernel density 
      # estimation technique
      
      # Print histograms pdfs and estimated densities
      if verbose_printing == True and name == 'bkg' and k <> j:
        full = 'full' if pos == None else 'dec'
        # print histograms
        printFrame(w,[score_str],[w.pdf('sighistpdf_{0}_{1}'.format(k,j)), w.pdf('bkghistpdf_{0}_{1}'.format(k,j))], makePlotName(full,'train',k,j,type='hist'),['signal','bkg'])
        # print histogram and density estimation together
        #printFrame(w,score_str,[w.pdf('sighistpdf_{0}_{1}'.format(k,j)), w.pdf('bkghistpdf_{0}_{1}'.format(k,j)),w.pdf('sigdist_{0}_{1}'.format(k,j)),w.pdf('bkgdist_{0}_{1}'.format(k,j))], makePlotName(full,'train',k,j,type='hist'),['signal_hist','bkg_hist','signal_est','bkg_est'])
        # print density estimation
        #printFrame(w,'score',[w.pdf('sigdist_{0}_{1}'.format(k,j)), w.pdf('bkgdist_{0}_{1}'.format(k,j))], makePlotName(full,'trained',k,j,type='density'),['signal','bkg'])
        if k < j and k <> 'F0':
          histos.append([w.pdf('sighistpdf_{0}_{1}'.format(k,j)), w.pdf('bkghistpdf_{0}_{1}'.format(k,j))])
          histos_names.append(['f{0}-f{1}_signal'.format(k,j), 'f{0}-f{1}_background'.format(k,j)])

  for k,c in enumerate(c0):
    for j,c_ in enumerate(c1):
      if k == j: 
        continue
      traindata, targetdata = loadData('train',k,j)
      numtrain = traindata.shape[0]       

      # Should I be using test data here?
      outputs = predict('{0}/model/{1}/{2}/adaptive_{3}_{4}.pkl'.format(dir,model_g,c1_g,k,j),traindata.reshape(traindata.shape[0],traindata.shape[1]))
      #outputs = clf.predict_proba(traindata.reshape(traindata.shape[0],1)) 
      saveHistos(w,outputs,s,bins,low,high,(k,j))

  printMultiFrame(w,'score',histos, makePlotName('decomp','all',type='hist'),histos_names)
  traindata, targetdata = loadData('train','F0','F1')
  numtrain = traindata.shape[0]       

  # Should I be using test data here?
  outputs = predict('{0}/model/{1}/{2}/adaptive_F0_F1.pkl'.format(dir,model_g,c1_g),traindata.reshape(traindata.shape[0],traindata.shape[1]))
  #outputs = clf.predict_proba(traindata.reshape(traindata.shape[0],1)) 
  saveHistos(w,outputs,s_full, bins_full, low_full, high_full)
     
  w.Print()

  w.writeToFile('{0}/workspace_DecomposingTestOfMixtureModelsClassifiers.root'.format(dir))

# this is not very efficient
def scikitlearnFunc(filename,x=0.):
  '''
    Needed for the scikit-learn wrapper
  '''
  traindata = np.array(([x])) if model_g == 'mlp' else np.array((x))
  outputs = predict(filename,traindata)[0]

  #if outputs[0] > 1:
  #  return 1.
  return outputs

class ScikitLearnCallback:
  def __init__(self,file):
    filename = file

  def __call__(self,x = 0.):
    train = np.array((x))
    outputs = predict(filename,train)[0]
    
    return outputs


def saveFig(x,y,file,labels=None,scatter=False,axis=None):
  fig,ax = plt.subplots()
  if scatter == True:
    if len(y) == 1: 
      ax.scatter(x,y[0],s=2)
      ax.set_xlabel(axis[0])
      ax.set_ylabel(axis[1])
    else:
      sc1 = ax.scatter(x,y[0],color='black')
      sc2 = ax.scatter(x,y[1],color='red')
      ax.legend((sc1,sc2),(labels[0],labels[1]))
      ax.set_xlabel('x')
      ax.set_ylabel('regression(score)')
  else:
    if len(y) == 1:
      ax.plot(x,y[0],'b')
    else:
      #Just supporting two plots for now
      ax.plot(x,y[0],'b-',label=labels[0]) 
      ax.plot(x,y[1],'r-',label=labels[1])
      ax.legend()
    ax.set_ylabel('LR')
    ax.set_xlabel('x')
  ax.set_title(file)
  if (len(y) > 1):
    # This breaks the naming convention for plots, I will solve
    # it later
    for i,l in enumerate(labels):
      np.savetxt('{0}/plots/{1}/results/{2}_{3}.txt'.format(dir,model_g,file,l),y[i])
  else:
    np.savetxt('{0}/plots/{1}/results/{2}.txt'.format(dir,model_g,file),y[0])
  fig.savefig('{0}/plots/{1}/{2}.png'.format(dir,model_g,file))
  plt.close(fig)
  plt.clf()


def saveFig3D(x,y,z,file,labels=None,scatter=False,axis=None):
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  if scatter == True:
    if len(z) == 1: 
      ax.scatter(x,y,z[0],s=2)
      ax.set_xlabel(axis[0])
      ax.set_ylabel(axis[1])
      ax.set_zlabel(axis[2])
    else:
      sc1 = ax.scatter(x,y,z[0],color='black')
      sc2 = ax.scatter(x,y,z[1],color='red')
      ax.legend((sc1,sc2),(labels[0],labels[1]))
      ax.set_xlabel('x')
      ax.set_ylabel('y')
      ax.set_zlabel('regression(score)')
  else:
    if len(z) == 1:
      ax.plot_wireframe(x,y,z[0],color='red')
    else:
      #Just supporting two plots for now
      ax.plot_wireframe(x,y,z[0],color='red',label=labels[0]) 
      ax.plot_wireframe(x,y,z[1],color='blue',label=labels[1])
      ax.legend()
    ax.set_zlabel('LR')
    ax.set_ylabel('y')
    ax.set_xlabel('x')
  ax.set_title(file)
  if (len(z) > 1):
    # This breaks the naming convention for plots, I will solve
    # it later
    for i,l in enumerate(labels):
      np.savetxt('{0}/plots/{1}/{2}_{3}.txt'.format(dir,model_g,file,l),z[i])
  else:
    np.savetxt('{0}/plots/{1}/{2}.txt'.format(dir,model_g,file),z[0])
  fig.savefig('{0}/plots/{1}/{2}.png'.format(dir,model_g,file))
  plt.close(fig)
  plt.clf()


def fitAdaptive(use_log=False):
  '''
    Use the computed score densities to compute 
    the decompose ratio test
  '''
  ROOT.gSystem.Load('{0}/parametrized-learning/SciKitLearnWrapper/libSciKitLearnWrapper'.format(dir))
  ROOT.gROOT.ProcessLine('.L CompositeFunctionPdf.cxx+')


  f = ROOT.TFile('{0}/workspace_DecomposingTestOfMixtureModelsClassifiers.root'.format(dir))
  w = f.Get('w')
  f.Close()
  
  print 'Calculating ratios'

  #x = w.var('x[-5,5]')
  #x = ROOT.RooRealVar('x','x',0.2,0.,5.)
  #getattr(w,'import')(ROOT.RooArgSet(x),ROOT.RooFit.RecycleConflictNodes()) 

  # To calculate the ratio between single functions
  def singleRatio(x,f0,f1,val):
    iter = x.createIterator()
    v = iter.Next()
    i = 0
    while v:
      v.setVal(val[i])
      v = iter.Next()
      i = i+1
    if f0.getVal(x) < 10E-25:
      return 0.
    return f1.getVal(x) / f0.getVal(x)
    #return f0.getVal(ROOT.RooArgSet(x))

  def evalDist(x,f0,val):
    iter = x.createIterator()
    v = iter.Next()
    i = 0
    while v:
      v.setVal(val[i])
      v = iter.Next()
      i = i+1
    return f0.getVal(x)

  # To calculate the ratio between single functions
  def regFunc(x,f0,f1,val):
    iter = x.createIterator()
    v = iter.Next()
    i = 0
    while v:
      v.setVal(val[i])
      v = iter.Next()
      i = i+1
    if (f0.getVal(x) + f1.getVal(x)) < 10E-25:
      return 0.
    return f1.getVal(x) / (f0.getVal(x) + f1.getVal(x))

  # Functions for Log Ratios
  def singleLogRatio(x, f0, f1, val):
    iter = x.createIterator()
    v = iter.Next()
    i = 0
    while v:
      v.setVal(val[i])
      v = iter.Next()
      i = i+1
    rat = np.log(f1.getVal(x)) - np.log(f0.getVal(x))
    return rat
  def computeLogKi(x, f0, f1, c0, c1, val):
    iter = x.createIterator()
    v = iter.Next()
    i = 0
    while v:
      v.setVal(val[i])
      v = iter.Next()
      i = i+1
    k_ij = np.log(c1*f1.getVal(x)) - np.log(c0*f0.getVal(x))
    return k_ij
  # ki is a vector
  def computeAi(k0, ki):
    ai = -k0 - np.log(1. + np.sum(np.exp(ki - k0),0))
    return ai

  # pair-wise ratios
  # and decomposition computation
  npoints = 50

  vars = ROOT.TList()
  for var in vars_g:
    vars.Add(w.var(var))
  x = ROOT.RooArgSet(vars)

  def evaluateDecomposedRatio(w,x,evalData,plotting=True, roc=False,gridsize=None):
    score = ROOT.RooArgSet(w.var('score'))
    npoints = evalData.shape[0]
    fullRatios = np.zeros(npoints)
    for k,c in enumerate(c0):
      innerRatios = np.zeros(npoints)
      if c == 0:
        continue
      for j,c_ in enumerate(c1):
        #testdata, testtarget = loadData('data/{0}/testdata_{1}_{2}.dat'.format(model_g, k, j)) 
        #xarray = np.sort(testdata)
        #xarray = np.sort(testdata)

        f0pdf = w.pdf('bkghistpdf_{0}_{1}'.format(k,j))
        f1pdf = w.pdf('sighistpdf_{0}_{1}'.format(k,j))
        f0 = w.pdf('f{0}'.format(k))
        f1 = w.pdf('f{0}'.format(j))
        if k <> j:
          outputs = predict('{0}/model/{1}/{2}/adaptive_{3}_{4}.pkl'.format(dir,model_g,c1_g,k,j),
                  evalData)
          pdfratios = [singleRatio(score,f0pdf,f1pdf,[xs]) for xs in outputs]
          pdfratios = np.array(pdfratios)
        else:
          pdfratios = np.ones(npoints)
        # the cases in which both distributions are the same can be problematic
        # one will expect that the classifier gives same prob to both signal and bkg
        # but it can behave in weird ways, I will just avoid this for now 
        innerRatios += (c_/c) * pdfratios
        ratios = [singleRatio(x,f0,f1,xs) for xs in evalData]
        #if k == 1 and  j == 2:
          #pdb.set_trace()
        if roc == True and k <> j:
          testdata, testtarget = loadData('test',k,j) 
          outputs = predict('{0}/model/{1}/{2}/adaptive_{3}_{4}.pkl'.format(dir,model_g,c1_g,k,j),
                    testdata.reshape(testdata.shape[0],testdata.shape[1]))
          clfRatios = np.array([singleRatio(score,f0pdf,f1pdf,[xs]) for xs in outputs])
          trRatios = np.array([singleRatio(x,f0,f1,xs) for xs in testdata])
          makeROC(trRatios/trRatios.max(), testtarget, makePlotName('dec','truth',k,j,type='roc'))
          makeROC(clfRatios/clfRatios.max(), testtarget,makePlotName('dec','train',k,j,type='roc'))
          # Scatter plot to compare regression function and classifier score
          reg = np.array([regFunc(x,f0,f1,xs) for xs in testdata])
          #reg = reg/np.max(reg)
          saveFig(outputs,[reg], makePlotName('dec','train',k,j,type='scat'),scatter=True, axis=['score','regression'])
          #saveFig(testdata, [reg, outputs],  makePlotName('dec','train',k,j,type='mul_scat'),scatter=True,labels=['regression', 'score'])

          # Scatter of distributions
          #f0data = [evalDist(x,f0,xs) for xs in testdata]
          #f1data = [evalDist(x,f1,xs) for xs in testdata]
          #saveFig(testdata,[f0data,f1data],makePlotName('dec','truth',k,j,type='scat'),scatter=True,axis=['x','y'])

        #saveFig(xarray, ratios, makePlotName('decomposed','truth',k,j,type='ratio'))
      fullRatios += 1./innerRatios
    return fullRatios

  if use_log == True:
    evaluateRatio = evaluateLogDecomposedRatio
    post = 'log'
  else:
    evaluateRatio = evaluateDecomposedRatio
    post = ''

  score = ROOT.RooArgSet(w.var('score'))
  scoref = ROOT.RooArgSet(w.var('scoref'))

  if use_log == True:
    getRatio = singleLogRatio
  else:
    getRatio = singleRatio
 

  # NN trained on complete model
  F0pdf = w.pdf('bkghistpdf_F0_F1')
  F1pdf = w.pdf('sighistpdf_F0_F1')

  # ROC for ratios
  # load test data
  # check if ratios fulfill the requeriments of type
  testdata, testtarget = loadData('test','F0',0) 
  decomposedRatio = evaluateDecomposedRatio(w,x,testdata,plotting=False,roc=True)
  outputs = predict('{0}/model/{1}/{2}/adaptive_F0_F1.pkl'.format(dir,model_g,c1_g),testdata.reshape(testdata.shape[0],testdata.shape[1]))
  completeRatio = np.array([getRatio(scoref,F1pdf,F0pdf,[xs]) for xs in outputs])
  realRatio = np.array([getRatio(x,w.pdf('F1'),w.pdf('F0'),xs) for xs in testdata])
  
  #Histogram F0-f0 for composed, full and true
  all_ratios_plots = []
  all_names_plots = []
  bins = 70
  low = 0.6
  high = 1.2
  if use_log == True:
    low = -1.0
    high = 1.0
  minimum = min([realRatio.min(), completeRatio.min(), decomposedRatio.min()])
  maximum = max([realRatio.max(), completeRatio.max(), decomposedRatio.max()]) 
  low = minimum - ((maximum - minimum) / bins)*10
  high = maximum + ((maximum - minimum) / bins)*10
  w.factory('ratio[{0},{1}]'.format(low, high))
  ratio = w.var('ratio')
  for curr, curr_ratios in zip(['composed','full','truth'],[realRatio, completeRatio, decomposedRatio]):
    numtest = curr_ratios.shape[0] 
    for l,name in enumerate(['sig','bkg']):
      hist = ROOT.TH1F('{0}_{1}hist_F0_f0'.format(curr,name),'hist',bins,low,high)
      for val in curr_ratios[l*numtest/2:(l+1)*numtest/2]:
        hist.Fill(val)
      datahist = ROOT.RooDataHist('{0}_{1}datahist_F0_f0'.format(curr,name),'hist',
            ROOT.RooArgList(ratio),hist)
      ratio.setBins(bins)
      histpdf = ROOT.RooHistPdf('{0}_{1}histpdf_F0_f0'.format(curr,name),'hist',
            ROOT.RooArgSet(ratio), datahist, 0)

      histpdf.specialIntegratorConfig(ROOT.kTRUE).method1D().setLabel('RooBinIntegrator')
      getattr(w,'import')(hist)
      getattr(w,'import')(datahist) # work around for morph = w.import(morph)
      getattr(w,'import')(histpdf) # work around for morph = w.import(morph)
      if name == 'bkg':
        all_ratios_plots.append([w.pdf('{0}_sighistpdf_F0_f0'.format(curr)),
              w.pdf('{0}_bkghistpdf_F0_f0'.format(curr))])
        #all_names_plots.append(['{0}_signal'.format(curr),'{0}_bkg'.format(curr)])
      
  all_ratios_plots = [[all_ratios_plots[0][0],all_ratios_plots[1][0],all_ratios_plots[2][0]],
                    [all_ratios_plots[0][1],all_ratios_plots[1][1],all_ratios_plots[2][1]]]
  all_names_plots = [['sig_truth','sig_full','sig_composed'],
                    ['bkg_truth','bkg_full','bkg_composed']]
  printMultiFrame(w,'ratio',all_ratios_plots, makePlotName('ratio','comparison',type='hist'+post),all_names_plots,setLog=True)

  saveFig(completeRatio,[realRatio], makePlotName('full','train',type='scat'+post),scatter=True,axis=['full trained ratio','true ratio'])
  saveFig(decomposedRatio,[realRatio], makePlotName('comp','train',type='scat'+post),scatter=True, axis=['composed trained ratio','true ratio'])

  makeSigBkg(np.array(decomposedRatio),testtarget,makePlotName('comp','train',type='sigbkg'+post))
  makeSigBkg(np.array(realRatio), testtarget,makePlotName('full','truth',type='sigbkg'+post))
  makeSigBkg(np.array(completeRatio), testtarget,makePlotName('full','train',type='sigbkg'+post))

  testdata, testtarget = loadData('test','F0','F1') 
  # Scatter plot to compare regression function and classifier score
  reg = np.array([regFunc(x,w.pdf('F0'),w.pdf('F1'),xs) for xs in testdata])
  #reg = reg/np.max(reg)
  outputs = predict('{0}/model/{1}/{2}/adaptive_F0_F1.pkl'.format(dir,model_g,c1_g),testdata.reshape(testdata.shape[0],testdata.shape[1]))
  saveFig(outputs,[reg], makePlotName('full','train',type='scat'),scatter=True,axis=['score','regression'])
  #saveFig(testdata, [reg, outputs],  makePlotName('full','train',type='mul_scat'),scatter=True,labels=['regression', 'score'])

  #w.Print()

if __name__ == '__main__':
  classifiers = {'svc':svm.NuSVC(probability=True),'svr':svm.NuSVR(),
        'logistic': linear_model.LogisticRegression(), 
        'bdt':GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
        max_depth=1, random_state=0),
        'mlp':''}
  clf = None
  if (len(sys.argv) > 1):
    model_g = sys.argv[1]
    clf = classifiers.get(sys.argv[1])
  if clf == None:
    model_g = 'logistic'
    clf = classifiers['logistic']    
    print 'Not found classifier, Using logistic instead'

  c1[0] = sys.argv[2]
  c1_g = "%.2f"%c1[0]
  c1 = c1 / c1.sum()
  print c0
  print c1
  print c1_g
  
  ROOT.gROOT.SetBatch(ROOT.kTRUE)
  ROOT.RooAbsPdf.defaultIntegratorConfig().setEpsRel(1E-15)
  ROOT.RooAbsPdf.defaultIntegratorConfig().setEpsAbs(1E-15)
  # Set this value to False if only final plots are needed
  verbose_printing = True
  
  makeModelND()
  makeData(num_train=100000,num_test=30000) 
  trainClassifier(clf)
  classifierPdf()
  fitAdaptive(use_log=False)

