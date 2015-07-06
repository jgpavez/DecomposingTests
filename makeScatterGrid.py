import numpy as np
import scipy
from pandas import DataFrame
from seaborn import pairplot
import pdb
import matplotlib.pyplot as plt
from pandas.tools.plotting import parallel_coordinates

c0 = np.array([.0,.3, .7])
c1 = np.array([.1,.3, .7])
c1_g = ''
dir = ''
'''
def parallel_coordinates(frame, class_column, cols=None, ax=None, color=None,
                     use_columns=False, xticks=None, colormap=None,
                     **kwds):
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    n = len(frame)
    class_col = frame[class_column]
    class_min = np.amin(class_col)
    class_max = np.amax(class_col)

    if cols is None:
        df = frame.drop(class_column, axis=1)
    else:
        df = frame[cols]

    used_legends = set([])

    ncols = len(df.columns)

    # determine values to use for xticks
    if use_columns is True:
        if not np.all(np.isreal(list(df.columns))):
            raise ValueError('Columns must be numeric to be used as xticks')
        x = df.columns
    elif xticks is not None:
        if not np.all(np.isreal(xticks)):
            raise ValueError('xticks specified must be numeric')
        elif len(xticks) != ncols:
            raise ValueError('Length of xticks must match number of columns')
        x = xticks
    else:
        x = range(ncols)

    fig = plt.figure()
    ax = plt.gca()

    Colorm = plt.get_cmap(colormap)

    for i in range(n):
        y = df.iloc[i].values
        kls = class_col.iat[i]
        ax.plot(x, y, color=Colorm((kls - class_min)/(class_max-class_min)), **kwds)

    for i in x:
        ax.axvline(i, linewidth=1, color='black')

    ax.set_xticks(x)
    ax.set_xticklabels(df.columns)
    ax.set_xlim(x[0], x[-1])
    ax.legend(loc='upper right')
    ax.grid()

    bounds = np.linspace(class_min,class_max,10)
    cax,_ = mpl.colorbar.make_axes(ax)
    cb = mpl.colorbar.ColorbarBase(cax, cmap=Colorm, spacing='proportional', ticks=bounds, boundaries=bounds, format='%.2f')

    return fig
'''
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

def makeGrid(traindata,k,j):
    x,y = traindata 
    num = x.shape[0]
    df = DataFrame(np.hstack((x[num/2-250:num/2+250],y[num/2-250:num/2+250,None])),columns=range(x.shape[1])+["class"])
    _ = pairplot(df,vars=[1,3,4,5,9],hue="class",size=2.5)
    plt.title('f{0}-f{1}'.format(k,j))
    plt.savefig('dec_truth_{0}_{1}_grid.png'.format(k,j))

def makeParallel(traindata):
    x,y = traindata 
    num = x.shape[0]
    df = DataFrame(np.hstack((x[num/2-100:num/2+100],y[num/2-100:num/2+100,None])),columns=range(x.shape[1])+["class"])
    parallel_coordinates(df, 'class')
    plt.title('Parallel coordinates F0-f0')
    plt.savefig('paralell_coordinates_F0-f0.png')

#for k,c in enumerate(c0):
#    for j,c_ in enumerate(c1):
#       if k > j or k == j:
#            continue
#        makeGrid(loadData('train',k,j,folder='.'),k,j)

makeParallel(loadData('train','F0',0,folder='.'))

