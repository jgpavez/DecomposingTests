import numpy as np
import scipy
from pandas import DataFrame
from seaborn import pairplot
import pdb
import matplotlib.pyplot as plt

c0 = np.array([.0,.3, .7])
c1 = np.array([.1,.3, .7])
c1_g = ''
dir = ''

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
    plt.savefig('dec_truth_{0}_{1}_grid.png'.format(k,j))

for k,c in enumerate(c0):
    for j,c_ in enumerate(c1):
        if k > j or k == j:
            continue
        makeGrid(loadData('train',k,j,folder='.'),k,j)
