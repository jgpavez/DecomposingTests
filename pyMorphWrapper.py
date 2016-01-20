'''
Simple python Wrapper for EFT Morphing code
author: jpavezse
'''


from ctypes import *
import numpy as np
import pdb
import time
from itertools import combinations

#reading C++ library
lib = cdll.LoadLibrary('./RandomEFT/RandomEFT/libcMorphWrapper.so')
#lib.getWeights.restype = c_char_p
lib.getCrossSections.restype = c_char_p

class MorphingWrapper:
  def __init__(self):
    # Initialize the C object
    self.obj = lib.MorphingWrapperNew()
    self.mem_values = dict()

  def setStringData(self,data):
    # Set samples data in the form of string: 
    #'Nsamples Ncouplings types[P|D|S]*Ncouplings target sample1 sample2 ...'
    # where target are the N couplings of the sample to morph and sample* 
    # N couplings of each sample
    self.string_data = data
    s_data = create_string_buffer(data)
    lib.setSampleData(self.obj,s_data)
  
  def getStringData(self):
    string_data = '{0} {1} '.format(self.nsamples,self.ncouplings)
    string_data = string_data + ' '.join(str(x) for x in self.types) + ' '
    string_data = string_data + ' '.join(str(x) for x in self. morphed) + ' '
    string_data = string_data + ' '.join(str(x) for sample in self.basis for x in sample)
    return string_data

  def setSampleData(self,nsamples,ncouplings,types,morphed,samples,ncomb=20):
    lib.getWeights.restype = POINTER(c_float*(nsamples+2))
    self.nsamples = nsamples
    self.ncouplings = ncouplings
    self.types = types
    self.morphed = morphed
    self.samples = samples
    self.basis = samples[:self.nsamples]
    self.ncomb = ncomb 
    string_data = self.getStringData()
    self.setStringData(string_data)

  def resetBasis(self, sample):
    self.basis = sample
    string_data = self.getStringData()
    self.setStringData(string_data)

  def resetTarget(self, target):
    self.morphed = target
    string_data = self.getStringData()
    self.setStringData(string_data)

  def getWeights(self):
    # Return computed weights for each sample
    s = lib.getWeights(self.obj)
    results = [float(x) for x in s.contents]
    weights = results[:-2]
    self.det = results[-2]
    self.cond = results[-1]
    return weights
    #return [float(x) for x in s.split()]

  def getCrossSections(self):
    # Return cross section for each one of the samples 
    # The C++ version is not working

    #s = lib.getCrossSections(obj)
    return self.__computeCrossSections()
  
  def __computeCrossSections(self):
    # Compute Cross Section using analytical formula, only for VBF 2nSM for now
    # Defining constants
    ca =  0.70710678
    sa =  0.70710678
    c_ = [0.1220,0.05669,0.003246,0.00006837,0.04388,0.002685,0.00007918,0.00002289]
    osm = 0.1151

    def formula(ksm,khzz,kazz):
      return 4*osm*((ca**4)*((ksm**4) + c_[0]*(ksm**3)*khzz + c_[1]*(ksm**2)*(khzz**2) + c_[2]*ksm*(khzz**3) + c_[3]*(khzz**4)) +\
        (ca**2)*(sa**2)*(c_[4]*(ksm**2)*(kazz**2) + c_[5]*ksm*khzz*(kazz**2) + c_[6]*(khzz**2)*(kazz**2)) + (sa**4)*c_[7]*(kazz**4))

    cross_sections = []
    for sample in self.basis:
      ksm = sample[0]
      khzz = 16.247*sample[1]
      kazz = 16.247*sample[2]
      cross_sections.append(formula(ksm,khzz,kazz))

    return cross_sections

  def computeNeff(self,basis,samples):
    basis = [samples[b] for b in basis]
    self.resetBasis(basis)
    weights = np.array(self.getWeights())
    #print weights
    cross_sections = np.array(self.getCrossSections())
    n_tot = (np.abs(np.multiply(weights,cross_sections))).sum()
    n_eff = (np.multiply(weights,cross_sections)).sum()
    return np.abs(n_eff / n_tot)

  def computeStats(self,basis,samples):
    basis = [samples[b] for b in basis]
    self.resetBasis(basis)
    weights = np.array(self.getWeights())
    #print weights
    cross_sections = np.array(self.getCrossSections())
    n_tot = (np.abs(np.multiply(weights,cross_sections))).sum()
    n_eff = (np.multiply(weights,cross_sections)).sum()
    return (np.abs(n_eff / n_tot),self.det,self.cond)

  def evalMaxPair(self,x,samples,cvalues_1,cvalues_2,verb=False):
    val = 0.
    target = self.morphed[:]
    norm = len(cvalues_1) * len(cvalues_2)
    result = 0.
    for val1 in cvalues_1:
      for val2 in cvalues_2:
        target[1] = val1
        target[2] = val2
        self.resetTarget(target)
        r1 = self.computeNeff(x[0],samples)
        r2 = self.computeNeff(x[1],samples) 
        result += np.max(r1, r2)
        #result = (r1 + r2)/2.
    r1,det1,cond1 = self.computeStats(x[0],samples)
    r2,det2,cond2 = self.computeStats(x[1],samples)
    val = result/norm
    if cond1 > 100000. or cond2 > 100000.:
      val = 0.
    else:
      print val
    return val

  def dynamicMorphing(self,target=None,cvalues_1=None,cvalues_2=None):
    # This function will work in case you have more samples that needed
    rng = np.random.RandomState(1234)
    samples = self.samples[:]
    indexes = np.array([samples.index(x) for x in samples if x <> self.morphed])
    # This is the sample to fit (not necessarily the morphed when fitting)
    if target <> None:
      indexes = np.array([samples.index(x) for x in samples if x <> target])

    self.ncomb = 23
    ncomb_indexes = indexes[rng.choice(len(indexes),self.ncomb,replace=False)]

    #Considering condition number
    comb = list(combinations(ncomb_indexes,self.nsamples))
    comb = [comb[i] for i in rng.choice(len(comb),int(len(comb)/2),replace=False)] 
    start = time.time()
    comb = sorted(comb,key=lambda x: self.computeNeff(x,samples),reverse=True)[:int(len(comb)*0.3)]
    end = time.time()
    print 'elapsed time : {0}'.format(end-start)
    print len(comb)
    pairs = []
    for c in comb:
      print 'eval'
      l1 = [i for i in indexes if i not in c]
      comb2_len = self.nsamples*2 - len(indexes)
      comb2 = np.array(list(combinations(c,self.nsamples*2 - len(indexes))))[rng.choice(comb2_len,
              int(comb2_len*0.5),replace=False)]
      for c2 in comb2:
        pairs.append([c,tuple(l1 + list(c2))])
    pairs = [pairs[i] for i in rng.choice(len(pairs),int(len(pairs)*0.3),replace=False)]
    print len(pairs)
    start = time.time()

    best_result = 0.
    for p in pairs:
      result = self.evalMaxPair(p,samples,cvalues_1,cvalues_2)
      if result > best_result:
        best_result = result
        best = p
      if result > 0.55:
        best = p
        break
    end = time.time()

    self.mem_values.clear()
    self.resetTarget(self.morphed)
    result,det,cond = self.computeStats(best[0],samples)
    print 'Result: {0} ,Det: {1}, Cond: {2}'.format(result, det, cond)
    result,det,cond = self.computeStats(best[1],samples)
    print 'Result: {0} ,Det: {1}, Cond: {2}'.format(result, det, cond)
    print 'Elapsed time : {0}'.format(end-start)

    return best

