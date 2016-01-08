#!/usr/bin/env python
__author__ = "Pavez J. <juan.pavezs@alumnos.usm.cl>"


import ROOT
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.externals import joblib

import sys

import os.path
import pdb

from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing


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

def preProcessing(traindata,k,j, scaler):
  assert scaler <> None
  if (k,j) not in scaler:
    scaler[(k,j)] = preprocessing.StandardScaler().fit(traindata)
  traindata = scaler[(k,j)].transform(traindata)
  return traindata

def loadData(type,k,j,folder=None,dir='',c1_g='',preprocessing=False,scaler=None, persist=False,
      model_g='mlp'):
  if folder <> None:
    fk = np.loadtxt('{0}/{1}_{2}.dat'.format(folder,type,k))
    fj = np.loadtxt('{0}/{1}_{2}.dat'.format(folder,type,j))
  else:
    fk = np.loadtxt('{0}/data/{1}/{2}/{3}_{4}.dat'.format(dir,model_g,c1_g,type,k))
    fj = np.loadtxt('{0}/data/{1}/{2}/{3}_{4}.dat'.format(dir,model_g,c1_g,type,j))
  rng = np.random.RandomState(1111)
  # HARCODED RIGHT NOW

  data_num = 19900
  #data_num = 5000
  indices = rng.choice(fk.shape[0],data_num,replace=False)
  fk = fk[indices]
  indices = rng.choice(fj.shape[0],data_num,replace=False)
  fj = fj[indices] 

  fk = fk[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,20,24,25,26,27,28,29,30,31,36,40,42]]
  fj = fj[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,20,24,25,26,27,28,29,30,31,36,40,42]]

  num1 = fj.shape[0]
  num0 = fk.shape[0]
  traindata = np.zeros((num0+num1,fk.shape[1])) if len(fk.shape) > 1 else \
                  np.zeros(num0+num1)
  targetdata = np.zeros(num0+num1)
  traindata[:num1] = fj[:]
  traindata[num1:] = fk[:]
  targetdata[:num1].fill(1)
  targetdata[num1:].fill(0)
  if preprocessing == True:
    traindata = preProcessing(traindata,k,j, scaler)
    # this should be in the preProcessing?
    if persist == True:
      joblib.dump(scaler[(k,j)],'{0}/model/{1}/{2}/{3}_{4}_{5}.dat'.format(dir,'mlp',c1_g,'scaler',k,j))
  return (traindata, targetdata)


def printMultiFrame(w,obs,all_pdfs,name,legends,
              dir='/afs/cern.ch/user/j/jpavezse/systematics',
              model_g='mlp',setLog=False,y_text='',print_pdf=False,title='',
              x_text='x'):
  '''
    This just print a bunch of pdfs 
    in a Canvas
  ''' 

  # Preliminaries
  ROOT.gROOT.SetStyle('Plain')
  #ROOT.gStyle.SetOptTitle(0)
  ROOT.gStyle.SetOptStat(0)
  ROOT.gStyle.SetOptFit(1)
  ROOT.gStyle.SetPalette(1)

  ROOT.gStyle.SetTitleX(0.5)
  ROOT.gStyle.SetTitleAlign(23)
  ROOT.gStyle.SetTitleBorderSize(0)

  # Hope I don't need more colors ...
  colors = [ROOT.kBlue,ROOT.kRed,ROOT.kBlack,ROOT.kGreen,
    ROOT.kYellow]
  style = [ROOT.kSolid,ROOT.kSolid,ROOT.kDashed,ROOT.kDashed]
  can = ROOT.TCanvas('c1')
  can.Divide(1,len(all_pdfs))
  legs = []
  frames = []
  for curr,pdf in enumerate(all_pdfs): 
    x = w.var(obs[curr])
    can.cd(curr+1)
    if curr <> len(pdf) - 1:
      #ROOT.gPad.SetBottomMargin(0.01)
      if curr == 0:
        ROOT.gPad.SetTopMargin(0.09)
      else:
        ROOT.gPad.SetTopMargin(0.01)
      ROOT.gPad.SetRightMargin(0.01)
    else:
      ROOT.gPad.SetTopMargin(0.01)
      ROOT.gPad.SetRightMargin(0.01)
    ROOT.gPad.SetLeftMargin(0.04)
    if setLog == True:
      ROOT.gPad.SetLogy(1)
    funcs = []
    line_colors = []
    line_styles = []
    for i,p in enumerate(pdf):
      funcs.append(p)
      line_colors.append(ROOT.RooFit.LineColor(colors[i]))
      line_styles.append(ROOT.RooFit.LineStyle(style[i]))
    frames.append(x.frame(ROOT.RooFit.Name(legends[curr][0]),ROOT.RooFit.
        Title(legends[curr][0].split('_')[0])))
    for i,f in enumerate(funcs):
      if isinstance(f,str):
        funcs[0].plotOn(frames[-1], ROOT.RooFit.Components(f),ROOT.RooFit.Name(legends[curr][i]), line_colors[i],
        line_styles[i])
      else:
        f.plotOn(frames[-1],ROOT.RooFit.Name(legends[curr][i]),line_colors[i],line_styles[i])
    legs.append(ROOT.TLegend(0.79, 0.73, 0.90, 0.87))
    #leg.SetFillColor(ROOT.kWhite)
    #leg.SetLineColor(ROOT.kWhite)
    # TODO This is just a quick fix because is now working how it should
    for i,l in enumerate(legends[curr]):
      if i == 0:
        legs[-1].AddEntry(frames[-1].findObject(legends[curr][i]), l.split('_')[1], 'l')
      else:
        legs[-1].AddEntry(frames[-1].findObject(legends[curr][i]), l.split('_')[1], 'l')
    legs[-1].SetFillColor(0)
    legs[-1].SetBorderSize(0)
    legs[-1].SetTextSize(0.06)
    legs[-1].SetFillColor(0)
    legs[-1].SetBorderSize(0)
    frames[-1].SetTitleSize(0.06,"Y")
    frames[-1].SetTitleSize(0.06,"X")
    frames[-1].GetYaxis().CenterTitle(1)
    frames[-1].GetYaxis().SetTitleOffset(0.35)
    if curr == 0:
      frames[-1].SetTitle("{0};;{1}".format(title,y_text))
    elif curr == len(all_pdfs)-1:
      frames[-1].SetTitle(";{0};{1}".format(x_text,y_text))
      frames[-1].GetXaxis().SetTitleOffset(0.25)
    else:
      frames[-1].SetTitle(";;{0}".format(y_text))
        
    frames[-1].Draw()
    legs[-1].Draw()
    ROOT.gPad.Update()
    can.Modified()
    can.Update()
  can.SaveAs('{0}/plots/{1}/{2}.png'.format(dir,model_g,name))
  if print_pdf == True:
    can.SaveAs('{0}/plots/{1}/{2}.pdf'.format(dir,model_g,name))

def printFrame(w,obs,pdf,name,legends,
              dir='/afs/cern.ch/user/j/jpavezse/systematics',model_g='mlp',
              title='',y_text='',x_text='',range=None,print_pdf=False
      ):
  '''
    This just print a bunch of pdfs 
    in a Canvas
  ''' 
  # Preliminaries
  ROOT.gROOT.SetStyle('Plain')
  if len(obs) > 1:
    ROOT.gStyle.SetOptTitle(0)
  ROOT.gStyle.SetOptStat(0)
  ROOT.gStyle.SetOptFit(1)
  ROOT.gStyle.SetPalette(1)

  ROOT.gStyle.SetTitleX(0.5)
  ROOT.gStyle.SetTitleAlign(23)
  ROOT.gStyle.SetTitleBorderSize(0)

  # Hope I don't need more colors ...
  colors = [ROOT.kBlue,ROOT.kRed,ROOT.kGreen,ROOT.kBlack,
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
  legs = []
  for j,fra in enumerate(frame):    
    can.cd(j+1)
    if len(obs) == 1:
      ROOT.gPad.SetRightMargin(0.01)
    else:
      if j <> len(obs) - 1:
        #ROOT.gPad.SetBottomMargin(0.01)
        if j == 0:
          ROOT.gPad.SetTopMargin(0.09)
        else:
          ROOT.gPad.SetTopMargin(0.01)
        ROOT.gPad.SetRightMargin(0.01)
      else:
        ROOT.gPad.SetTopMargin(0.01)
        ROOT.gPad.SetRightMargin(0.01)
      ROOT.gPad.SetLeftMargin(0.04)
    for i,f in enumerate(funcs):
        if isinstance(f,str):
          funcs[0].plotOn(fra, ROOT.RooFit.Components(f),ROOT.RooFit.Name(legends[i]), line_colors[i])
        else:
          f.plotOn(fra,ROOT.RooFit.Name(legends[i]),line_colors[i])
    legs.append(ROOT.TLegend(0.69, 0.73, 0.89, 0.87))
    #leg.SetFillColor(ROOT.kWhite)
    #leg.SetLineColor(ROOT.kWhite)
    for i,l in enumerate(legends):
      if i == 0:
        legs[-1].AddEntry(fra.findObject(legends[i]), l, 'l')
      else:
        legs[-1].AddEntry(fra.findObject(legends[i]), l, 'l')
    legs[-1].SetFillColor(0)
    legs[-1].SetBorderSize(0)
    #legs[-1].SetTextSize(0.06)
    legs[-1].SetFillColor(0)
    legs[-1].SetBorderSize(0)
    #fra.SetTitleSize(0.06,"Y")
    #fra.SetTitleSize(0.06,"X")
    fra.GetYaxis().CenterTitle(1)
    #fra.GetYaxis().SetTitleOffset(0.35)
    if len(obs) == 1:
      fra.SetTitle("{0};{1};{2}".format(title,x_text,y_text))
    else:
      fra.SetTitle(";;{0}".format(y_text))
    if range <> None:
      fra.GetXaxis().SetRangeUser(range[0],range[1])
    fra.Draw()
    legs[-1].Draw()
  can.SaveAs('{0}/plots/{1}/{2}.png'.format(dir,model_g,name))
  if print_pdf == True:
    can.SaveAs('{0}/plots/{1}/{2}.pdf'.format(dir,model_g,name))

def saveFig(x,y,file,labels=None,scatter=False,contour=False,axis=None, 
            dir='/afs/cern.ch/user/j/jpavezse/systematics',
            model_g='mlp',marker=False, hist=False, marker_value=None, x_range=None,title='',multi=False,print_pdf=False,hist2D=False,pixel=False,min_value=None):
  fig,ax = plt.subplots()
  colors = ['b-','r-','k-']
  colors_rgb = ['blue','red','black']
  if contour == True: 
    if pixel == True:
      vals = np.flipud(y[1]) 
      #vals = y[1]
      im = plt.imshow(vals, extent=(y[0].min(), y[0].max(), x.min(),x.max()),
                 interpolation='nearest', cmap=cm.gist_rainbow_r)

      CB = plt.colorbar(im, shrink=0.8, extend='both')
      #plt.legend(lines,labels,frameon=False,fontsize=11)
      ax.set_title(title)
      ax.set_xlabel('g2',fontsize=11) 
      ax.set_ylabel('g1',fontsize=11)
    else:
      levels = [2.,10.,40.,60.,80.,100.,200.,400.,600.]
      #levels = [-800.,-600.,-400.,-200.,0.,200,1000.,10000.,50000.]
      #im = plt.imshow(y[1], interpolation='bilinear', origin='lower',
      #            cmap=cm.gray, extent=(-3,3,-2,2))
      #cs1 = plt.contour(x,y[0],y[1],[0.,0.1,0.5,1.,5.,10.,50.,100.])
      cs1 = plt.contour(x,y[0],y[1].transpose(),levels,origin='lower',alpha=.5, colors=('#ff0000', '#ff9900', '#999900', 'w', '#009999', '#0099ff', '#0000ff','#00ffff'))
      #cs2 = plt.contour(x,y[0],y[2],[0.,0.1,0.5,1.,5.,10.,50.,100.],linestyles="dashed")
      plt.clabel(cs1, inline=1, fontsize=10)
      #CB = plt.colorbar(cs1, shrink=0.8, extend='both')
      #lines = [cs1.collections[0],cs2.collections[0]]
      lines = [cs1.collections[0]]
      #plt.legend(lines,labels,frameon=False,fontsize=11)
      ax.set_title('Likelihood ratio values for Kazz,KHzz, target=({0:.2f},{1:.2f})'.
          format(marker_value[0],marker_value[1]))
      ax.set_xlabel('Kazz',fontsize=11) 
      ax.set_ylabel('Khazz',fontsize=11)
      ax.annotate('min=({0:.2f},{1:.2f})'.format(min_value[0],min_value[1]), xy=(min_value[0]+0.01
      ,min_value[1]+0.01),xytext=(min_value[0]+0.03,min_value[1]+0.03),arrowprops=dict(facecolor='red'))
      if marker == True: 
        plt.axvline(marker_value[0], color='black')
        plt.axhline(marker_value[1], color='black')
      #ax.plot([c1[0]],[c1[1]],'o')
      #ax.annotate('min',xy=(c1[0],c1[1]),xytext=(0.,0.))
  else:
    if scatter == True:
      if len(y) == 1: 
        ax.scatter(x,y[0],marker='*',color='g')
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
        if hist2D == True:
          H, xedges, yedges, img = plt.hist2d(y[0][:450], y[1][:450],bins=12)
          pdb.set_trace()
          extent = [xedges[0], xedges[-1], yedges[-1], yedges[0]]
          fig = plt.figure()
          plt.xlim([xedges[0],xedges[-1]])
          plt.ylim([1.2,yedges[-1]])
          ax = fig.add_subplot(1, 1, 1)
          ax.set_xlabel(axis[0])
          ax.set_ylabel(axis[1])
          im = ax.imshow(H, cmap=plt.cm.jet,interpolation='nearest',extent=extent)
          mean1,mean2 = (y[0][:450].mean(),y[1][:450].mean())
          std1, std2 = (y[0][:450].std(),y[1][:450].std())
          ax.annotate('mean=[{0:.2f},{1:.2f}]\nstd=[{2:.2f},{3:.2f}]'.format(mean1,mean2,std1,std2)
              ,xy=(mean1,mean2),xytext=(mean1-0.01,mean2-0.1),arrowprops=dict(facecolor='red'),color='white')
          fig.colorbar(im, ax=ax)
          if marker == True:
            plt.axvline(marker_value[0], color='black')
            plt.axhline(marker_value[1], color='black')
        else:
          if len(y) == 1:
            if x_range <> None:
              ax.hist(y[0],color='red', label=labels[0],bins=16, range=[x_range[0], x_range[1]],
            histtype='step', alpha=0.5)
            else:
              ax.hist(y[0],color='red', label=labels[0],bins=20,
            histtype='step', alpha=0.5)
          else:
            #Just supporting two plots for now
            if x_range <> None:
              for i,ys in enumerate(y): 
                ax.hist(ys,color=colors_rgb[i],label=labels[i],bins=15, range=[x_range[0],x_range[1]],histtype='step',normed=1, alpha=0.5) 
            else:
              for i,ys in enumerate(y): 
                ax.hist(ys,color=colors_rgb[i],label=labels[i],bins=15,histtype='step',normed=1, alpha=0.5) 
            ax.legend(frameon=False,fontsize=11)
          if axis <> None:
            ax.set_xlabel(axis[0]) 
          else:
            ax.set_xlabel('x')
          ax.set_ylabel('Count')
          if marker == True:
            plt.axvline(marker_value, color='black')
      else:
        if len(y) == 1:
          ax.plot(x,y[0],'b')
          ax.annotate('fit min 1.5285', xy=(1.52857,0),xytext=(1.8,200.),arrowprops=dict(facecolor='red'))
        else:
          #Just supporting two plots for now
          linestyles=['--','--']
          markers = ['+','x']
          for k,ys in enumerate(y):
            ax.plot(x,ys,colors[k],label=labels[k],linestyle=linestyles[k],marker=markers[k]) 
          ax.legend(frameon=False,fontsize=11)
        if axis <> None:
          ax.set_ylabel(axis[1])
          ax.set_xlabel(axis[0]) 
        else:
          ax.set_ylabel('LR')
          ax.set_xlabel('x')
        if marker == True:
          plt.axvline(marker_value, color='black')
    ax.set_title(title)
    #if (len(y) > 1):
      # This breaks the naming convention for plots, I will solve
      # it later
    #  for i,l in enumerate(labels):
    #    np.savetxt('{0}/plots/{1}/results/{2}_{3}.txt'.format(dir,model_g,file,l),y[i])
    #else:
    #  np.savetxt('{0}/plots/{1}/results/{2}.txt'.format(dir,model_g,file),y[0])
  if print_pdf == True:
    fig.savefig('{0}/plots/{1}/{2}.pdf'.format(dir,model_g,file))
  fig.savefig('{0}/plots/{1}/{2}.png'.format(dir,model_g,file))
  plt.close(fig)
  plt.clf()

def saveMultiFig(x,y,file,labels=None, 
            dir='/afs/cern.ch/user/j/jpavezse/systematics',
            model_g='mlp',title='',print_pdf=False):
  plot_num = 311
  for k,ys in enumerate(y):
    # Fix later
    plt.subplot(plot_num)
    plt.plot(x,ys[0],'b-',label=labels[k][0])
    plt.plot(x,ys[1],'r-',label=labels[k][1])
    if k == len(y)-1:
      plt.xlabel('x',fontsize=11)
    plt.ylabel('Ratio',fontsize=11)
    plt.tick_params(axis='both', labelsize=10) 
    plt.legend(frameon=False, fontsize=11)
    if plot_num == 311:
      plt.title('{0}'.format(title))
    plot_num = plot_num + 1
  if print_pdf == True:
    plt.savefig('{0}/plots/{1}/{2}.pdf'.format(dir,model_g,file))
  plt.savefig('{0}/plots/{1}/{2}.png'.format(dir,model_g,file))
  plt.close()
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


def makeMultiROC(all_outputs, targets, label,
           dir='/afs/cern.ch/user/j/jpavezse/systematics',model_g='mlp',
            true_score=None,print_pdf=False,title='',pos=[(0,1),(0,2),(1,2)]):
  '''
    make plots for ROC curve of classifier and 
    test data.
  '''
  plot_n = 311
  fprs = []
  tprs = []
  fpr_trues = []
  tpr_trues = []
  roc_aucs = []
  roc_auc_trues = []
  for k,(outputs,target) in enumerate(zip(all_outputs,targets)):
    fpr, tpr, _  = roc_curve(target.ravel(),outputs.ravel())
    fprs.append(fpr)
    tprs.append(tpr)
    roc_auc = auc(fpr, tpr)
    roc_aucs.append(roc_auc)
    if true_score <> None:
      fpr_true, tpr_true, _  = roc_curve(target.ravel(),true_score[k].ravel())
      fpr_trues.append(fpr_true)
      tpr_trues.append(tpr_true)
      roc_auc_true = auc(fpr_true, tpr_true)
      roc_auc_trues.append(roc_auc_true)
    plt.subplot(plot_n)
    plt.plot(fprs[-1], tprs[-1],'b-', label='ROC curve trained (area = %0.2f)' % roc_aucs[-1])
    if true_score <> None:
      plt.plot(fpr_trues[-1], tpr_trues[-1],'r-', label='ROC curve true (area = %0.2f)' % roc_auc_trues[-1])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    if k == len(targets)-1:
      plt.xlabel('Sensitivity',fontsize=11)
    plt.ylabel('1-Specificity',fontsize=11)
    plt.tick_params(axis='both', labelsize=10)
    if plot_n == 311:
      plt.title('{0}'.format(title))
    plt.legend(loc="lower right",frameon=False,fontsize=11)
    plt.text(0.62,0.42,'f{0}-f{1}'.format(pos[k][0],pos[k][1]))
    plot_n = plot_n + 1
  #np.savetxt('{0}/plots/{1}/results/{2}.txt'.format(dir,model_g,label),np.column_stack((fpr,tpr)))
  plt.savefig('{0}/plots/{1}/{2}.png'.format(dir,model_g,label))
  if print_pdf == True:
    plt.savefig('{0}/plots/{1}/{2}.pdf'.format(dir,model_g,label))
  plt.close()
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



def makeSigBkg(all_outputs, targets, label,
              dir='/afs/cern.ch/user/j/jpavezse/systematics',model_g='mlp',
              print_pdf=False,legends=None, title=''):
  '''
  make plots for ROC curve of classifier and
  test data.
  '''
  tprs = []
  fnrs = []
  aucs = []
  thresholds = np.linspace(0,1.0,150) 
  fig = plt.figure()
  for k,(outputs,target) in enumerate(zip(all_outputs,targets)):
    fnrs.append(np.array([float(np.sum((outputs > tr) * (target == 0)))/float(np.sum(target == 0)) for tr in thresholds]))
    fnrs[-1] = fnrs[-1].ravel()
    tprs.append(np.array([float(np.sum((outputs < tr) * (target == 1)))/float(np.sum(target == 1)) for tr in thresholds]))
    tprs[-1] = tprs[-1].ravel()
    aucs.append(auc(tprs[-1],fnrs[-1]))
    plt.plot(tprs[-1], fnrs[-1], label='ROC {0} (area = {1:.2f})'.format(
       legends[k],aucs[-1]))
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('Signal Efficiency',fontsize=11)
  plt.ylabel('Background Rejection',fontsize=11)
  plt.tick_params(axis='both', labelsize=10)
  plt.title('{0}'.format(title))
  plt.legend(loc="lower left",frameon=False, fontsize=11)
  #np.savetxt('{0}/plots/{1}/results/{2}.txt'.format(dir,model_g,label),np.column_stack((tpr,fnr)))
  plt.savefig('{0}/plots/{1}/{2}.png'.format(dir,model_g,label))
  if print_pdf == True:
    plt.savefig('{0}/plots/{1}/{2}.pdf'.format(dir,model_g,label))
  plt.close(fig)
  plt.clf()


def plotCValues(c0,c1,dir='/afs/cern.ch/user/j/jpavezse/systematics',
            c1_g='',model_g='mlp',true_dist=False,vars_g=None,
            workspace='workspace_DecomposingTestOfMixtureModelsClassifiers.root',
            use_log=False, n_hist=150,c_eval=0, range_min=-1.0,range_max=0.):

  ''' 
    Plot histogram of fitted values
  '''
  if use_log == True:
    post = 'log'
  else:
    post = ''

  keys = ['true','dec']
  c1_ = dict((key,np.zeros(n_hist)) for key in keys)
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

  size = min(vals.shape[0],vals1.shape[0])
  # 1D
  #saveFig([],[vals1], 
  #    makePlotName('g2','train',type='hist_g1g2'),hist=True, 
  #    axis=['g2'],marker=True,marker_value=c1[c_eval],
  #    labels=labels,x_range=[range_min,range_max],dir=dir,
  #    model_g=model_g,title='Histogram for fitted g2', print_pdf=True)
  # 2D
  saveFig([],[vals,vals1], 
      makePlotName('g1g2','train',type='hist'),hist=True,hist2D=True, 
      axis=['g1','g2'],marker=True,marker_value=c1,
      labels=labels,dir=dir,model_g=model_g,title='2D Histogram for fitted g1,g2', print_pdf=True,
      x_range=[[0.5,1.4],[1.1,1.9]])



def getWeights(g_1=1.,g_2=1.5):
  #Set the basis
  basis_g1 = np.array((1.,1.,1.,1.,0.))
  basis_g2 = np.array((0.,2.,1.,3.,1.))
  
  #basis_g1 = np.array((0.,1.,1.,1.,1.))
  #basis_g2 = np.array((1.,0.,1.,2.,3.))
 
  #define the formulae
  g1_t4 = lambda x,y: x*x*x*x + 0*y
  g1_t2_g2_t2  = lambda x,y: x*x*y*y
  g1_t3_g2 = lambda x,y: x*x*x*y
  g1_g2_t3 = lambda x,y: y*y*y*x
  g2_t4 = lambda x,y: y*y*y*y + 0*x

  a = np.zeros((5,5))

  for i in range(5):
      a[0,i] = g1_t4(basis_g1[i],basis_g2[i])
      a[2,i] = g1_t2_g2_t2(basis_g1[i],basis_g2[i])
      a[1,i] = g1_t3_g2(basis_g1[i],basis_g2[i])
      a[3,i] = g1_g2_t3(basis_g1[i],basis_g2[i])
      a[4,i] = g2_t4(basis_g1[i],basis_g2[i])

  #print a
  
  b = np.linalg.inv(a)
  #print b
  
  weight = np.zeros(5)
  #identify the weight

  for j in range(5):
      weight[j] = b[j,0]*g1_t4(g_1,g_2) + b[j,1]*g1_t3_g2(g_1,g_2) + b[j,2]*g1_t2_g2_t2(g_1,g_2) + b[j,3]*g1_g2_t3(g_1,g_2) + b[j,4]*g2_t4(g_1,g_2) 

  return weight

