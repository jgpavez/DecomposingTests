#!/usr/bin/env python
__author__ = "Pavez J. <juan.pavezs@alumnos.usm.cl>"


import ROOT
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.externals import joblib

import sys

import os.path
import pdb

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import preprocessing


'''
  Some usefull functions to print and load data
'''


def makePlotName(full, truth, f0=None, f1=None, type=None,
                 dir='/afs/cern.ch/user/j/jpavezse/systematics',
                 c1_g='', model_g='mlp'):
    if full == 'dec':
        return '{0}_{1}_f{2}_f{3}_{4}_{5}'.format(
            full, truth, f0, f1, model_g, type)
    else:
        return '{0}_{1}_{2}_{3}'.format(full, truth, model_g, type)


def loadData(type, k, j, folder=None, dir='', c1_g='', model_g='mlp'):
    if folder is not None:
        fk = np.loadtxt('{0}/{1}_{2}.dat'.format(folder, type, k))
        fj = np.loadtxt('{0}/{1}_{2}.dat'.format(folder, type, j))
    else:
        fk = np.loadtxt(
            '{0}/data/{1}/{2}/{3}_{4}.dat'.format(dir, model_g, c1_g, type, k))
        fj = np.loadtxt(
            '{0}/data/{1}/{2}/{3}_{4}.dat'.format(dir, model_g, c1_g, type, j))
    rng = np.random.RandomState(1111)

    num1 = fj.shape[0]
    num0 = fk.shape[0]
    traindata = np.zeros(
        (num0 + num1,
         fk.shape[1])) if len(
        fk.shape) > 1 else np.zeros(
            num0 + num1)
    targetdata = np.zeros(num0 + num1)
    traindata[:num1] = fj[:]
    traindata[num1:] = fk[:]
    targetdata[:num1].fill(1)
    targetdata[num1:].fill(0)

    return (traindata, targetdata)

def makePreliminaries():   
    # Preliminaries
    ROOT.gROOT.SetStyle('Plain')
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptFit(1)
    ROOT.gStyle.SetPalette(1)

    ROOT.gStyle.SetTitleX(0.5)
    ROOT.gStyle.SetTitleAlign(23)
    ROOT.gStyle.SetTitleBorderSize(0)

def printMultiFrame(
        w,
        obs,
        all_pdfs,
        name,
        legends,
        dir='/afs/cern.ch/user/j/jpavezse/systematics',
        model_g='mlp',
        setLog=False,
        y_text='',
        print_pdf=False,
        title='',
        x_text='x'):
    '''
      This just print a bunch of pdfs
      in a Canvas
    '''

    makePreliminaries()
    colors = [ROOT.kBlue, ROOT.kRed, ROOT.kBlack, ROOT.kGreen,
              ROOT.kYellow]
    style = [ROOT.kSolid, ROOT.kSolid, ROOT.kDashed, ROOT.kDashed]
    can = ROOT.TCanvas('c1')
    can.Divide(1, len(all_pdfs))
    legs = []
    frames = []
    for curr, pdf in enumerate(all_pdfs):
        x = w.var(obs[curr])
        can.cd(curr + 1)
        if curr != len(pdf) - 1:
            if curr == 0:
                ROOT.gPad.SetTopMargin(0.09)
            else:
                ROOT.gPad.SetTopMargin(0.01)
            ROOT.gPad.SetRightMargin(0.01)
        else:
            ROOT.gPad.SetTopMargin(0.01)
            ROOT.gPad.SetRightMargin(0.01)
        ROOT.gPad.SetLeftMargin(0.04)
        if setLog:
            ROOT.gPad.SetLogy(1)
        funcs = []
        line_colors = []
        line_styles = []
        for i, p in enumerate(pdf):
            funcs.append(p)
            line_colors.append(ROOT.RooFit.LineColor(colors[i]))
            line_styles.append(ROOT.RooFit.LineStyle(style[i]))
        frames.append(x.frame(ROOT.RooFit.Name(legends[curr][0]), ROOT.RooFit.
                              Title(legends[curr][0].split('_')[0])))
        for i, f in enumerate(funcs):
            if isinstance(f, str):
                funcs[0].plotOn(frames[-1],
                                ROOT.RooFit.Components(f),
                                ROOT.RooFit.Name(legends[curr][i]),
                                line_colors[i],
                                line_styles[i])
            else:
                f.plotOn(frames[-1],
                         ROOT.RooFit.Name(legends[curr][i]),
                         line_colors[i],
                         line_styles[i])
        legs.append(ROOT.TLegend(0.79, 0.73, 0.90, 0.87))
        for i, l in enumerate(legends[curr]):
            if i == 0:
                legs[-1].AddEntry(frames[-1].findObject(legends[curr]
                                                        [i]), l.split('_')[1], 'l')
            else:
                legs[-1].AddEntry(frames[-1].findObject(legends[curr]
                                                        [i]), l.split('_')[1], 'l')
        legs[-1].SetFillColor(0)
        legs[-1].SetBorderSize(0)
        legs[-1].SetTextSize(0.06)
        legs[-1].SetFillColor(0)
        legs[-1].SetBorderSize(0)
        frames[-1].SetTitleSize(0.06, "Y")
        frames[-1].SetTitleSize(0.06, "X")
        frames[-1].GetYaxis().CenterTitle(1)
        frames[-1].GetYaxis().SetTitleOffset(0.35)
        if curr == 0:
            frames[-1].SetTitle("{0};;{1}".format(title, y_text))
        elif curr == len(all_pdfs) - 1:
            frames[-1].SetTitle(";{0};{1}".format(x_text, y_text))
            frames[-1].GetXaxis().SetTitleOffset(0.25)
        else:
            frames[-1].SetTitle(";;{0}".format(y_text))

        frames[-1].Draw()
        legs[-1].Draw()
        ROOT.gPad.Update()
        can.Modified()
        can.Update()
    can.SaveAs('{0}/plots/{1}/{2}.png'.format(dir, model_g, name))
    if print_pdf:
        can.SaveAs('{0}/plots/{1}/{2}.pdf'.format(dir, model_g, name))


def printFrame(w, obs, pdf, name, legends,
               dir='/afs/cern.ch/user/j/jpavezse/systematics', model_g='mlp',
               title='', y_text='', x_text='', range=None, print_pdf=False
               ):
    '''
      This just print a bunch of pdfs
      in a Canvas
    '''
    # Preliminaries
    makePreliminaries()
    if len(obs) > 1:
        ROOT.gStyle.SetOptTitle(0)

    colors = [ROOT.kBlue, ROOT.kRed, ROOT.kGreen, ROOT.kBlack,
              ROOT.kYellow]
    x = []
    for var in obs:
        x.append(w.var(var))
    funcs = []
    line_colors = []
    for i, p in enumerate(pdf):
        funcs.append(p)
        line_colors.append(ROOT.RooFit.LineColor(colors[i]))

    can = ROOT.TCanvas('c1')
    can.Divide(1, len(obs))
    frame = []
    for var in x:
        frame.append(var.frame())
    legs = []
    for j, fra in enumerate(frame):
        can.cd(j + 1)
        if len(obs) == 1:
            ROOT.gPad.SetRightMargin(0.01)
        else:
            if j != len(obs) - 1:
                # ROOT.gPad.SetBottomMargin(0.01)
                if j == 0:
                    ROOT.gPad.SetTopMargin(0.09)
                else:
                    ROOT.gPad.SetTopMargin(0.01)
                ROOT.gPad.SetRightMargin(0.01)
            else:
                ROOT.gPad.SetTopMargin(0.01)
                ROOT.gPad.SetRightMargin(0.01)
            ROOT.gPad.SetLeftMargin(0.04)
        for i, f in enumerate(funcs):
            if isinstance(f, str):
                funcs[0].plotOn(
                    fra, ROOT.RooFit.Components(f), ROOT.RooFit.Name(
                        legends[i]), line_colors[i])
            else:
                f.plotOn(fra, ROOT.RooFit.Name(legends[i]), line_colors[i])
        legs.append(ROOT.TLegend(0.69, 0.73, 0.89, 0.87))

        for i, l in enumerate(legends):
            if i == 0:
                legs[-1].AddEntry(fra.findObject(legends[i]), l, 'l')
            else:
                legs[-1].AddEntry(fra.findObject(legends[i]), l, 'l')
        legs[-1].SetFillColor(0)
        legs[-1].SetBorderSize(0)
        legs[-1].SetFillColor(0)
        legs[-1].SetBorderSize(0)

        fra.GetYaxis().CenterTitle(1)

        fra.SetTitle(" ")
        if range is not None:
            fra.GetXaxis().SetRangeUser(range[0], range[1])
        fra.Draw()
        legs[-1].Draw()
    can.SaveAs('{0}/plots/{1}/{2}.png'.format(dir, model_g, name))
    if print_pdf:
        can.SaveAs('{0}/plots/{1}/{2}.pdf'.format(dir, model_g, name))


def saveFig(
        x,
        y,
        file,
        type='',
        labels=None,
        axis=None,
        dir='/afs/cern.ch/user/j/jpavezse/systematics',
        model_g='mlp',
        marker=False,
        marker_value=None,
        x_range=None,
        title='',
        multi=False,
        print_pdf=False,
        min_value=None):
    '''
        Helper to print a lof of different plots
    '''
    
    plt.ioff()
    fig, ax = plt.subplots()
    colors = ['b-', 'r-', 'k-']
    colors_rgb = ['blue', 'red', 'black']
    if type=='contour2':
        cs1 = plt.contour(
            x, y[0], y[1].transpose(), [
                0., 0.1, 0.5, 1., 5., 10., 20., 50.], label=labels[0])
        cs2 = plt.contour(
            x, y[0], y[2].transpose(), [
                0., 0.1, 0.5, 1., 5., 10., 20., 50.], linestyles="dashed", label=labels[1])
        plt.clabel(cs1, inline=1, fontsize=10)
        lines = [cs1.collections[0], cs2.collections[0]]
        plt.legend(lines, labels, frameon=False, fontsize=11)

        ax.set_xlabel('signal weight', fontsize=11)
        ax.set_ylabel('background weight', fontsize=11)
        ax.annotate(
            'min=({0:.2f},{1:.2f})'.format(
                min_value[0],
                min_value[1]),
            xy=(
                min_value[0] + 0.001,
                min_value[1] + 0.001),
            xytext=(
                min_value[0] + 0.03,
                min_value[1] + 0.03),
            arrowprops=dict(
                facecolor='red'))
        if marker:
            plt.axvline(marker_value[0], color='black')
            plt.axhline(marker_value[1], color='black')
        plt.legend(loc='upper left', frameon=False, numpoints=1)
    elif type=='contour':
        levels = [2.3,6.]
        cs1 = plt.contour(
            x,
            y[0],
            y[1].transpose(),
            levels,
            origin='lower',
            alpha=.5,
            colors=(
                '#ff0000',
                '#ff9900',
                '#999900',
                'w',
                '#009999',
                '#0099ff',
                '#0000ff',
                '#00ffff'))
        plt.clabel(cs1, inline=1, fontsize=10)
        lines = [cs1.collections[0]]
        ax.set_title('-2ln(Q), target=({0:.2f},{1:.2f})'.
            format(marker_value[0],marker_value[1]))
        ax.set_xlabel('Kazz',fontsize=11)
        ax.set_ylabel('Khazz',fontsize=11)
        ax.set_xlabel('signal weight', fontsize=11)
        ax.set_ylabel('background weight', fontsize=11)
        ax.annotate(
            'min=({0:.2f},{1:.2f})'.format(
                min_value[0],
                min_value[1]),
            xy=(
                min_value[0] + 0.001,
                min_value[1] + 0.001),
            xytext=(
                min_value[0] + 0.03,
                min_value[1] + 0.03),
            arrowprops=dict(
                facecolor='red'))
        if marker:
            plt.axvline(marker_value[0], color='black')
            plt.axhline(marker_value[1], color='black')

        labels = ['95% C.L','68.27% C.L']
        for i in range(len(labels)):
            cs1.collections[i].set_label(labels[i])
        plt.legend(loc='upper left', frameon=False, numpoints=1)
    elif type=='pixel':
        vals = np.flipud(y[1])
        im = plt.imshow(
            vals,
            extent=(
                y[0].min(),
                y[0].max(),
                x.min(),
                x.max()),
            interpolation='nearest',
            cmap=cm.gist_rainbow_r)

        CB = plt.colorbar(im, shrink=0.8, extend='both')
        ax.set_title(title)
        ax.set_xlabel('g2', fontsize=11)
        ax.set_ylabel('g1', fontsize=11)
    elif type=='scatter':
        if len(y) == 1:
            ax.scatter(x, y[0], marker='*', color='g')
            ax.set_xlabel(axis[0])
            ax.set_ylabel(axis[1])
        else:
            sc1 = ax.scatter(x, y[0], color='black')
            sc2 = ax.scatter(x, y[1], color='red')
            ax.legend((sc1, sc2), (labels[0], labels[1]))
            ax.set_xlabel('x')
            ax.set_ylabel('regression(score)')
    elif type=='hist2D':
        H, xedges, yedges, img = plt.hist2d(
            y[0][:450], y[1][:450], bins=12)
        extent = [xedges[0], xedges[-1], yedges[-1], yedges[0]]
        fig = plt.figure()
        plt.xlim([xedges[0], xedges[-1]])
        plt.ylim([1.2, yedges[-1]])
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel(axis[0])
        ax.set_ylabel(axis[1])
        im = ax.imshow(
            H,
            cmap=plt.cm.jet,
            interpolation='nearest',
            extent=extent)
        mean1, mean2 = (y[0][:450].mean(), y[1][:450].mean())
        std1, std2 = (y[0][:450].std(), y[1][:450].std())
        ax.annotate(
            'mean=[{0:.2f},{1:.2f}]\nstd=[{2:.2f},{3:.2f}]'.format(
                mean1, mean2, std1, std2), xy=(
                mean1, mean2), xytext=(
                mean1 - 0.01, mean2 - 0.1), arrowprops=dict(
                facecolor='red'), color='white')
        fig.colorbar(im, ax=ax)
        if marker:
            plt.axvline(marker_value[0], color='black')
            plt.axhline(marker_value[1], color='black')
    elif type=='hist':
        if len(y) == 1:
            if x_range is not None:
                ax.hist(
                    y[0], color='red', label=labels[0], bins=16, range=[
                        x_range[0], x_range[1]], histtype='step', alpha=0.5)
            else:
                ax.hist(
                    y[0],
                    color='red',
                    label=labels[0],
                    bins=20,
                    histtype='step',
                    alpha=0.5)
        else:
            # Just supporting two plots for now
            if x_range is not None:
                for i, ys in enumerate(y):
                    ax.hist(ys, color=colors_rgb[i], label=labels[i], bins=15, range=[
                        x_range[0], x_range[1]], histtype='step', normed=1, alpha=0.5)
            else:
                for i, ys in enumerate(y):
                    ax.hist(
                        ys,
                        color=colors_rgb[i],
                        label=labels[i],
                        bins=15,
                        histtype='step',
                        normed=1,
                        alpha=0.5)
            ax.legend(frameon=False, fontsize=11)
        if axis is not None:
            ax.set_xlabel(axis[0])
        else:
            ax.set_xlabel('x')
        ax.set_ylabel('count')
        if marker:
            plt.axvline(marker_value, color='black')
    elif type=='':
        if len(y) == 1:
            ax.plot(x, y[0], 'b')
            ax.annotate(
                'fit min 1.5285', xy=(
                    1.52857, 0), xytext=(
                    1.8, 200.), arrowprops=dict(
                    facecolor='red'))
        else:
            # Just supporting two plots for now
            linestyles = ['--', '--']
            markers = ['+', 'x']
            for k, ys in enumerate(y):
                ax.plot(x, ys, colors[k], label=labels[k])
            ax.legend(frameon=False, fontsize=11)
        if axis is not None:
            ax.set_ylabel(axis[1])
            ax.set_xlabel(axis[0])
        else:
            ax.set_ylabel('LR')
            ax.set_xlabel('x')
        if marker:
            plt.axvline(marker_value, color='black')

    plt.tight_layout()
    if print_pdf:
        fig.savefig('{0}/plots/{1}/{2}.pdf'.format(dir, model_g, file))
    fig.savefig('{0}/plots/{1}/{2}.png'.format(dir, model_g, file))
    plt.close(fig)
    plt.clf()

def saveMultiFig(x,y,file,labels=None, 
            dir='/afs/cern.ch/user/j/jpavezse/systematics',
            model_g='mlp',title='',print_pdf=False):
    plot_num = 311
    for k,ys in enumerate(y):
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

def makeMultiROC(
    all_outputs,
    targets,
    label,
    dir='/afs/cern.ch/user/j/jpavezse/systematics',
    model_g='mlp',
    true_score=None,
    print_pdf=False,
    title='',
    pos=[
        (0,
         1),
        (0,
         2),
        (1,
         2)]):
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
    for k, (outputs, target) in enumerate(zip(all_outputs, targets)):
        fpr, tpr, _ = roc_curve(target.ravel(), outputs.ravel())
        fprs.append(fpr)
        tprs.append(tpr)
        roc_auc = auc(fpr, tpr)
        roc_aucs.append(roc_auc)
        if true_score is not None:
            fpr_true, tpr_true, _ = roc_curve(
                target.ravel(), true_score[k].ravel())
            fpr_trues.append(fpr_true)
            tpr_trues.append(tpr_true)
            roc_auc_true = auc(fpr_true, tpr_true)
            roc_auc_trues.append(roc_auc_true)
        plt.subplot(plot_n)
        plt.plot(fprs[-1], tprs[-1], 'b-',
                 label='ROC curve trained (area = %0.2f)' % roc_aucs[-1])
        if true_score is not None:
            plt.plot(fpr_trues[-1], tpr_trues[-1], 'r-',
                     label='ROC curve true (area = %0.2f)' % roc_auc_trues[-1])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        if k == len(targets) - 1:
            plt.xlabel('Sensitivity', fontsize=11)
        plt.ylabel('1-Specificity', fontsize=11)
        plt.tick_params(axis='both', labelsize=10)
        if plot_n == 311:
            plt.title('{0}'.format(title))
        plt.legend(loc="lower right", frameon=False, fontsize=11)
        plt.text(0.62, 0.42, 'f{0}-f{1}'.format(pos[k][0], pos[k][1]))
        plot_n = plot_n + 1
    plt.savefig('{0}/plots/{1}/{2}.png'.format(dir, model_g, label))
    if print_pdf:
        plt.savefig('{0}/plots/{1}/{2}.pdf'.format(dir, model_g, label))
    plt.close()
    plt.clf()


def makeROC(outputs, target, label,
            dir='/afs/cern.ch/user/j/jpavezse/systematics', model_g='mlp'):
    '''
      make plots for ROC curve of classifier and
      test data.
    '''
    fpr, tpr, _ = roc_curve(target.ravel(), outputs.ravel())
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
    plt.savefig('{0}/plots/{1}/{2}.png'.format(dir, model_g, label))
    plt.plot()



def makeSigBkg(all_outputs, targets, label,
               dir='/afs/cern.ch/user/j/jpavezse/systematics', model_g='mlp',
               print_pdf=False, legends=None, title=''):
    '''
    make plots for ROC curve of classifier and
    test data.
    '''
    print 'Signal-Background plots'
    tprs = []
    fnrs = []
    aucs = []
    thresholds = np.linspace(0, 1.0, 150)
    fig = plt.figure()
    for k, (outputs, target) in enumerate(zip(all_outputs, targets)):
        fnrs.append(np.array([float(np.sum((outputs > tr) *
                                           (target == 0))) /
                              float(np.sum(target == 0)) for tr in thresholds]))
        fnrs[-1] = fnrs[-1].ravel()
        tprs.append(np.array([float(np.sum((outputs < tr) *
                                           (target == 1))) /
                              float(np.sum(target == 1)) for tr in thresholds]))
        tprs[-1] = tprs[-1].ravel()
        aucs.append(auc(tprs[-1], fnrs[-1]))
        plt.plot(tprs[-1], fnrs[-1], label='{0} (area = {1:.2f})'.format(
            legends[k], aucs[-1]))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('signal efficiency', fontsize=11)
    plt.ylabel('background rejection', fontsize=11)
    plt.tick_params(axis='both', labelsize=10)
    plt.legend(loc="lower left", frameon=False, fontsize=11)
    plt.tight_layout()
    plt.savefig('{0}/plots/{1}/{2}.png'.format(dir, model_g, label))
    if print_pdf:
        plt.savefig('{0}/plots/{1}/{2}.pdf'.format(dir, model_g, label))
    plt.close(fig)
    plt.clf()


