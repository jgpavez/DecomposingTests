'''
Simple python Wrapper for EFT Morphing code
author: jpavezse
'''


from ctypes import *
import numpy as np
import pdb
import time
from itertools import combinations

from deap import base
from deap import tools
from deap import creator
from deap import algorithms


#reading C++ library
lib = cdll.LoadLibrary('./RandomEFT/RandomEFT/libcMorphWrapper.so')
#lib.getWeights.restype = c_char_p
lib.getCrossSections.restype = c_char_p

  
def initInd(icls, size, samples):
  pop1 = np.zeros(len(samples))
  idx1 = np.random.choice(len(samples),size,replace=False)
  pop1[idx1] = 1
  pop2 = np.zeros(len(samples))
  idx2 = list(set(range(len(samples))) - set(idx1))
  idx2 = np.random.choice(idx2, size, replace=False)
  pop2[idx2] = 1
  return icls(np.append(pop1,pop2).astype(int))
 

def mutateInd(individual):
  # External mutation
  size = len(individual) / 2
  ind_arr = np.array(individual)
  i = np.random.choice(np.where(np.logical_and(ind_arr[:size] == 0, ind_arr[size:] == 0))[0])
  base  = np.random.choice((0,1))
  individual[np.random.choice(np.where(np.array(individual)[base*size:(base+1)*size] == 1)[0]) + base*size] = 0
  individual[i + size*base] = 1 
  # Internal Mutation
  ind_arr = np.array(individual)
  i = np.random.choice(np.where(np.logical_and(ind_arr[:size] == base, ind_arr[size:] == 1-base))[0])
  j = np.random.choice(np.where(np.logical_and(ind_arr[:size] == 1-base, ind_arr[size:] == base))[0])
  individual[i],individual[i+size] = individual[i+size],individual[i]
  individual[j],individual[j+size] = individual[j+size],individual[j]
  return individual,
    
def cxOver(ind1, ind2, section, base_size):
  size = len(ind1) / 2
  assert size > section
  base  = np.random.choice((0,1))
  choice = np.random.choice(size-section)
  mask = np.zeros(len(ind1)/2, np.bool)
  mask[choice:choice+section] = 1

  ind1_base = list()
  ind2_base = list()
    
  for base in (0,1):
    ind1_base.append(np.array(ind1[base*size:(base+1)*size]))
    ind2_base.append(np.array(ind2[base*size:(base+1)*size]))
    tmp = np.copy(ind2_base[-1][mask])
    ind2_base[-1][mask] = ind1_base[-1][mask]
    ind1_base[-1][mask] = tmp
 
    
  for inds in (ind1_base,ind2_base):
      for ind in inds:
        if sum(ind) > base_size: # Case in which we add 1s
            ind[np.random.choice(np.where(ind == 1)[0], sum(ind) - base_size,replace=False)] = 0      
        if sum(ind) < base_size: # Case in which we add 0s
            ind[np.random.choice(np.where(np.logical_and(inds[0] == 0, inds[1] == 0))[0], 
                                 base_size - sum(ind),replace=False)] = 1
  
  ind1[:] = list(np.append(ind1_base[0],ind1_base[1]))
  ind2[:] = list(np.append(ind2_base[0],ind2_base[1]))

  return ind1, ind2
  

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

  def setSampleData(self,nsamples,ncouplings,types,morphed=[1.,1.,1.],samples=None,ncomb=20,used_samples=30):
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
    self.used_samples = used_samples
    
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
    # TODO : Harcoded!
    #self.resetTarget(self.morphed)
    self.resetTarget(self.morphed)
    basis = [samples[b] for b in basis]
    self.resetBasis(basis)
    weights = np.array(self.getWeights())
    #print weights
    cross_sections = np.array(self.getCrossSections())
    n_tot = (np.abs(np.multiply(weights,cross_sections))).sum()
    n_eff = (np.multiply(weights,cross_sections)).sum()
    return np.abs(n_eff / n_tot)

  def evalMean(self,x,samples,cvalues_1,cvalues_2,verb=False):
    val = 0.
    target = self.morphed[:]
    norm = len(cvalues_1) * len(cvalues_2)
    result = 0.
    for val1 in cvalues_1:
      for val2 in cvalues_2:
        target[1] = val1
        target[2] = val2
        self.resetTarget(target)
        res,det,cond = self.computeStats(x,samples)
        result += (1./(val1**2 + val2**2)) * res
    r1,det1,cond1 = self.computeStats(x,samples)
    val = result/norm
    if cond1 > 100000.:
      val = 0.
    else:
      print val
    return val

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
        #result += np.max((r1, r2))
        result += r1 + r2
        #result = (r1 + r2)/2.
    r1,det1,cond1 = self.computeStats(x[0],samples)
    r2,det2,cond2 = self.computeStats(x[1],samples)
    #self.resetTarget(self.morphed)
    #r1 = self.computeNeff(x[0],samples)
    #r2 = self.computeNeff(x[1],samples) 
    #result += 2.*(r1 + r2)
    val = result/(norm)
    if cond1 > 100000. or cond2 > 100000.:
      val = 0.
    else:
      print val
    return val

  def dynamicMorphing(self,target=None,cvalues_1=None,cvalues_2=None):
    # This function will work in case you have more samples that needed
    rng = np.random.RandomState(1234)
    #rng = np.random.RandomState(1111)
    samples = self.samples[:]
    indexes = np.array([samples.index(x) for x in samples if x <> self.morphed])
    # This is the sample to fit (not necessarily the morphed when fitting)
    if target <> None:
      indexes = np.array([samples.index(x) for x in samples if x <> target])

    samples_indexes = indexes
    #samples_indexes = indexes[rng.choice(len(indexes), self.used_samples, replace=False)]
    #samples_indexes = sorted(indexes,key=lambda x: np.sqrt(np.sum((np.array(samples[x])-self.morphed)**2)))
    #samples_indexes = np.array((samples_indexes[:self.used_samples]))
    print len(samples_indexes)
    ncomb_indexes = samples_indexes[rng.choice(len(samples_indexes),self.ncomb,replace=False)]

    print 'Starting combinations'
    #Considering condition number
    comb = list(combinations(ncomb_indexes,self.nsamples))
    #comb = [comb[i] for i in rng.choice(len(comb),int(len(comb)/2),replace=False)] 
    start = time.time()
    print 'Start sorting'
    comb = sorted(comb,key=lambda x: self.computeNeff(x,samples),reverse=True)[:int(len(comb)*0.1)]
    #comb = sorted(comb,key=lambda x: self.computeNeff(x,samples),reverse=True)
    end = time.time()
    print 'Elapsed time combinations: {0}'.format(end-start)
    print 'Number of combinations: {0}'.format(len(comb))
    pairs = []
    for c in comb:
      print '.',
      #l1 = [i for i in samples_indexes if i not in c]
      l1 = np.array(list(set(samples_indexes) - set(c)))
      l1 = l1[rng.choice(len(l1),self.ncomb,replace=False)]
      #comb2_len = self.nsamples*2 - len(indexes)
      #comb2 = list(combinations(c,self.nsamples*2 - len(indexes)))
      #comb2 = np.array(comb2)[rng.choice(len(comb2),int(len(comb2)*0.05),replace=False)]
      #for c2 in comb2:
      #  pairs.append([c,tuple(l1 + list(c2))])
      if len(l1) <> self.nsamples:
        comb2 = list(combinations(l1, self.nsamples))
        comb2 = np.array(comb2)[rng.choice(len(comb2), int(len(comb2) * 0.03),replace=False)]
        for c2 in comb2:
          pairs.append([c,tuple(list(c2))])
      else:
        pairs.append([c,tuple(l1)])
    print ''
    pairs = [pairs[i] for i in rng.choice(len(pairs),int(len(pairs)*0.3),replace=False)]
    print 'Number of pairs: {0}'.format(len(pairs)) 
    start = time.time()
    best_result = 0.
    for p in pairs:
      try: 
        result = self.evalMaxPair(p,samples,cvalues_1,cvalues_2)
        if result > best_result:
          best_result = result
          best = p
        if result > 0.8:
          best = p
          best_result = result
          break
      except KeyboardInterrupt:
        print 'KeyboardInterrupt caught'
        break    
    end = time.time()
    print 'Elapsed time 2nd basis: {0}'.format(end-start)
    print 'Best Result : {0}'.format(best_result)
    print 'Results on target for each basis: '
    self.mem_values.clear()
    self.resetTarget(self.morphed)
    result,det,cond = self.computeStats(best[0],samples)
    print 'Result: {0} ,Det: {1}, Cond: {2}'.format(result, det, cond)
    result,det,cond = self.computeStats(best[1],samples)
    print 'Result: {0} ,Det: {1}, Cond: {2}'.format(result, det, cond)

    return best

  def simpleMorphing(self,target=None,cvalues_1=None,cvalues_2=None,c_eval=0):
    # TODO: Have to check the condition number and det
    # This function will work in case you have more samples that needed
    rng = np.random.RandomState(1234)
    samples = self.samples[:]
    indexes = np.array([samples.index(x) for x in samples if x <> self.morphed])
    # This is the sample to fit (not necessarily the morphed when fitting)
    if target <> None:
      indexes = np.array([samples.index(x) for x in samples if x <> target])
    #indexes = sorted(indexes,key=lambda x: np.sqrt(np.sum((np.array(samples[x])-self.morphed)**2)))
    #ncomb_indexes = indexes[:self.ncomb]
    ncomb_indexes = indexes[rng.choice(len(indexes),self.ncomb,replace=False)]
    #Considering condition number
    comb = list(combinations(ncomb_indexes,self.nsamples))
    comb = [comb[i] for i in rng.choice(len(comb),int(len(comb)/2),replace=False)] 
    start = time.time()
    comb = sorted(comb,key=lambda x: self.computeNeff(x,samples),reverse=True)[:int(len(comb)*0.3)]
    #comb = sorted(comb,key=lambda x: self.computeNeff(x,samples),reverse=True)[:int(len(comb)*0.3)]
    end = time.time()
    print 'elapsed time : {0}'.format(end-start)
    print len(comb)
    start = time.time()
    comb = [comb[i] for i in rng.choice(len(comb),int(len(comb)*0.1),replace=False)] 
    #best = sorted(pairs,key=lambda x: self.evalPair(x,samples,cvalues_1,cvalues_2))[-1]
    best_result = 0.
    for p in comb:
      result = self.evalMean(p,samples,cvalues_1,cvalues_2) if cvalues_2 <> None else\
                  self.evalMeanSingle(p,samples,cvalues_1,c_eval)
      if result > best_result:
        best_result = result
        best = p
      #if result > 0.70:
      #  res,det,cond = self.computeStats(p,samples)
      #  print 'COND: {0}'.format(cond)
      #  if cond < 100000.:
      #    break
    #best = p
    end = time.time()
    self.resetTarget(self.morphed)
    result,det,cond = self.computeStats(best,samples)
    print 'Result: {0} ,Det: {1}, Cond: {2}'.format(result, det, cond)
    print 'elapsed time : {0}'.format(end-start)
    #print 'n_eff: {0}'.format(self.computeNeff(best,samples))
    #self.resetBasis([samples[b] for b in best])

    return best

  def evaluateInd(self,individual,cvalues_1, cvalues_2):
    size = len(individual) / 2
    ind1,ind2 = np.array(individual[:size]),np.array(individual[size:])
    # Check that individual is a valid sample
    assert sum(ind1) == 15
    assert sum(ind2) == 15
    assert sum(np.logical_and(ind1,ind2)) == 0
    p = map(list,(np.where(ind1 == 1)[0], np.where(ind2 == 1)[0]))
    result = self.evalMaxPair(p,self.samples,cvalues_1,cvalues_2)
    return result,

  def evolMorphing(self,target=None,cvalues_1=None,cvalues_2=None,c_eval=0):

    res = initInd(np.array, 15, self.samples)
  
    toolbox = base.Toolbox()
    creator.create("FitnessMax", base.Fitness, weights=(1.0,), typecode='d')
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox.register("individual", initInd, creator.Individual, size=15, samples=self.samples)
    ind1 = toolbox.individual()
    ind2 = toolbox.individual()
      
    toolbox.register('evaluate', self.evaluateInd, cvalues_1=cvalues_1, cvalues_2=cvalues_2)

    #mutateInd(ind)
    toolbox.register('mutate', mutateInd)
    
    toolbox.register('mate', cxOver, section=3, base_size=15)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    #toolbox.mate(ind1,ind2)
    pop = toolbox.population(n=50)
    
    pop, stat = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=50)
    
    individual = pop[0]
    size = len(individual) / 2
    ind1,ind2 = np.array(individual[:size]),np.array(individual[size:])
    # Check that individual is a valid sample
    assert sum(ind1) == 15
    assert sum(ind2) == 15
    assert sum(np.logical_and(ind1,ind2)) == 0
    p = map(list,(np.where(ind1 == 1)[0], np.where(ind2 == 1)[0]))
    print self.evalMaxPair(p,self.samples,cvalues_1,cvalues_2)
    
    best = p
    self.mem_values.clear()
    self.resetTarget(self.morphed)
    result,det,cond = self.computeStats(best[0],self.samples)
    print 'Result: {0} ,Det: {1}, Cond: {2}'.format(result, det, cond)
    result,det,cond = self.computeStats(best[1],self.samples)
    print 'Result: {0} ,Det: {1}, Cond: {2}'.format(result, det, cond)
        
    return p

