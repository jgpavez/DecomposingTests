#!/usr/bin/env python
__author__ = "Pavez J. <juan.pavezs@alumnos.usm.cl>"


import ROOT
import numpy as np
from sklearn.metrics import roc_curve, auc

import sys

import os.path
import pdb

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


''' 
  Some usefull functions to print and load data
'''

def makePlotName(full, truth, f0 = None, f1 = None, type=None,
        dir='/afs/cern.ch/user/j/jpavezse/systematics',
        c1_g='',model_g='mlp'):
  if full == 'dec':
    return '{0}_{1}_f{2}_f{3}_{4}_{5}'.format(full, truth, f0, f1, model_g,type)
  else:
    return '{0}_{1}_{2}_{3}'.format(full, truth, model_g,type)

def loadData(type,k,j,folder=None,dir='',c1_g=''):
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


def printMultiFrame(w,obs,all_pdfs,name,legends,
              dir='/afs/cern.ch/user/j/jpavezse/systematics',
              model_g='mlp',setLog=False):
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

def printFrame(w,obs,pdf,name,legends,
              dir='/afs/cern.ch/user/j/jpavezse/systematics',model_g='mlp',
      ):
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

def saveFig(x,y,file,labels=None,scatter=False,contour=False,axis=None, 
            dir='/afs/cern.ch/user/j/jpavezse/systematics',
            model_g='mlp',marker=False, hist=False, marker_value=None, x_range=None):
  fig,ax = plt.subplots()
  if contour == True: 
    cs1 = plt.contour(x,y[0],y[1],[0.,0.1,0.5,1.,5.,10.,50.,100.])
    cs2 = plt.contour(x,y[0],y[2],[0.,0.1,0.5,1.,5.,10.,50.,100.],linestyles="dashed")
    plt.clabel(cs1, inline=1, fontsize=10)
    lines = [cs1.collections[0],cs2.collections[0]]
    plt.legend(lines,labels)
    ax.set_title('c1[0]-c2[0] -ln(L)')
    ax.set_xlabel('c1[0]') 
    ax.set_ylabel('c1[1]')
    print 'c1s: {0} {1}'.format(c1[0],c1[1])
    if marker == True: 
      plt.axvline(c1[0], color='black')
      plt.axhline(c1[1], color='black')
    #ax.plot([c1[0]],[c1[1]],'o')
    #ax.annotate('min',xy=(c1[0],c1[1]),xytext=(0.,0.))
  else:
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
      if hist == True:
        if len(y) == 1:
          ax.hist(y[0],color='black')
        else:
          #Just supporting two plots for now
          ax.hist(y[0],color='blue',label=labels[0],bins=15, range=[x_range[0],x_range[1]],histtype='step',normed=1, alpha=0.5) 
          ax.hist(y[1],color='red',label=labels[1],bins=15, range=[x_range[0],x_range[1]],histtype='step',normed=1,alpha=0.5)
          ax.legend()
        if axis <> None:
          ax.set_xlabel(axis[0]) 
        else:
          ax.set_xlabel('x')
        if marker == True:
          plt.axvline(marker_value, color='black')
      else:
        if len(y) == 1:
          ax.plot(x,y[0],'b')
        else:
          #Just supporting two plots for now
          ax.plot(x,y[0],'b-',label=labels[0]) 
          ax.plot(x,y[1],'r-',label=labels[1])
          ax.legend()
        if axis <> None:
          ax.set_ylabel(axis[1])
          ax.set_xlabel(axis[0]) 
        else:
          ax.set_ylabel('LR')
          ax.set_xlabel('x')
        if marker == True:
          plt.axvline(c1[0], color='black')
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

def saveFig3D(x,y,z,file,labels=None,scatter=False,
              dir='/afs/cern.ch/user/j/jpavezse/systematics',
              model_g='mlp',axis=None):
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


def makeROC(outputs, target, label,
           dir='/afs/cern.ch/user/j/jpavezse/systematics',model_g='mlp'):
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


def makeSigBkg(outputs, target, label,
              dir='/afs/cern.ch/user/j/jpavezse/systematics',model_g='mlp'):
  '''
  make plots for ROC curve of classifier and
  test data.
  '''

  thresholds = np.linspace(0,1.0,150) 
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

