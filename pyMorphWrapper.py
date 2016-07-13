'''
Python Wrapper for EFT Morphing code and Morphing Strategies
author: jpavezse
'''


from ctypes import *
import numpy as np
import pdb
import time
from itertools import combinations
import copy

from deap import base
from deap import tools
from deap import creator
from deap import algorithms
from scoop import futures
import multiprocessing


# reading C++ library
lib = cdll.LoadLibrary('./RandomEFT/RandomEFT/libcMorphWrapper.so')
#lib.getWeights.restype = c_char_p
lib.getCrossSections.restype = c_char_p

''' 
    Evolutionary algorithm functions
'''
 
def initInd(icls, size, samples):
    '''
        Random initialization of a pair of bases on the ea algorithm
        representation
    '''
    pop1 = np.zeros(len(samples))
    idx1 = np.random.choice(len(samples), size, replace=False)
    pop1[idx1] = 1
    pop2 = np.zeros(len(samples))
    idx2 = list(set(range(len(samples))) - set(idx1))
    idx2 = np.random.choice(idx2, size, replace=False)
    pop2[idx2] = 1
    return icls(np.append(pop1, pop2).astype(int))


def mutateInd(individual):
    '''
        Mutation operator, it has two phases:
        Internal mutation: Randomly change a choosed sample by an unused one (1<->0) 
            inside a base
        External mutation: Randomly swap used samples between the bases 
    '''
    # Internal mutation
    size = len(individual) / 2
    ind_arr = np.array(individual)
    i = np.random.choice(np.where(
            np.logical_and(ind_arr[
            :size] == 0, ind_arr[size:] == 0))[0])
    base = np.random.choice((0, 1))
    individual[np.random.choice(np.where(
            np.array(individual)[base * size:(
            base +1) *size] == 1)[0]) + base * size] = 0
    individual[i + size * base] = 1
    # External Mutation
    ind_arr = np.array(individual)
    i = np.random.choice(
        np.where(np.logical_and(ind_arr[
            :size] == base, ind_arr[size:] == 1 - base))[0])
    j = np.random.choice(
        np.where(np.logical_and(ind_arr[
            :size] == 1 - base, ind_arr[size:] == base))[0])
    individual[i], individual[i + size] = individual[i + size], individual[i]
    individual[j], individual[j + size] = individual[j + size], individual[j]
    return individual,


def cxOver(ind1, ind2, section, base_size):
    '''
        Crossover operator: Swap two randomly choosed slices between the 
        solutions ind1 and ind2 for each base, afterwards it fix the problems 
        introduced by that creating feasible solutions
    '''
    size = len(ind1) / 2
    assert size > section
    base = np.random.choice((0, 1))
    choice = np.random.choice(size - section)
    mask = np.zeros(len(ind1) / 2, np.bool)
    mask[choice:choice + section] = 1

    ind1_base = list()
    ind2_base = list()

    # Swap randomly choosed slices between two solutions (ind1, ind2) for each base
    for base in (0, 1):
        ind1_base.append(np.array(ind1[base * size:(base + 1) * size]))
        ind2_base.append(np.array(ind2[base * size:(base + 1) * size]))
        tmp = np.copy(ind2_base[-1][mask])
        ind2_base[-1][mask] = ind1_base[-1][mask]
        ind1_base[-1][mask] = tmp

    # Fix problems introduced when swapping the slices
    for inds in (ind1_base, ind2_base):
        for ind in inds:
            if sum(ind) > base_size:  # Case in which we add 1s
                ind[np.random.choice(np.where(ind == 1)[0], sum(
                    ind) - base_size, replace=False)] = 0
            if sum(ind) < base_size:  # Case in which we add 0s
                ind[np.random.choice(np.where(np.logical_and(inds[0] == 0, inds[1] == 0))[
                                     0], base_size - sum(ind), replace=False)] = 1

    ind1[:] = list(np.append(ind1_base[0], ind1_base[1]))
    ind2[:] = list(np.append(ind2_base[0], ind2_base[1]))

    return ind1, ind2

'''
    Morphing Class
'''

class MorphingWrapper:

    def __init__(self):
        # Initialize the C object
        lib.MorphingWrapperNew.restype = POINTER(c_char)
        self.obj = lib.MorphingWrapperNew()
        self.mem_values = dict()

    def setStringData(self, data):
        # Set samples data in the form of string:
        #'Nsamples Ncouplings types[P|D|S]*Ncouplings target sample1 sample2 ...'
        # where target are the N couplings of the sample to morph and sample*
        # N couplings of each sample
        self.string_data = data
        s_data = create_string_buffer(data)
        lib.setSampleData(self.obj, s_data)

    def getStringData(self):
        string_data = '{0} {1} '.format(self.nsamples, self.ncouplings)
        string_data = string_data + ' '.join(str(x) for x in self.types) + ' '
        string_data = string_data + \
            ' '.join(str(x) for x in self.morphed) + ' '
        string_data = string_data + \
            ' '.join(str(x) for sample in self.basis for x in sample)
        return string_data

    def setSampleData(
            self,
            nsamples,
            ncouplings,
            types,
            morphed=[1.,1.,1.],
            samples=None,
            ncomb=20,
            used_samples=30,
            use_alpha=False):
        lib.getWeights.restype = POINTER(c_float * (nsamples + 2))
        self.nsamples = nsamples # total number of samples to use
        self.ncouplings = ncouplings # number of couplings in each sample
        self.types = types
        self.morphed = morphed # target sample
        self.samples = samples # available samples
        self.basis = samples[:self.nsamples]
        self.ncomb = ncomb # used in dynamic morphing, subset of samples to make permutations
        string_data = self.getStringData()
        self.setStringData(string_data)
        self.used_samples = used_samples
        self.use_alpha = use_alpha

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
        # return [float(x) for x in s.split()]

    def getCrossSections(self):
        # Return cross section for each one of the samples
        # The C++ version is not working

        #s = lib.getCrossSections(obj)
        return self.__computeCrossSections()

    def __computeCrossSections(self):
        # Compute Cross Section using analytical formula, only for VBF 2nSM for now
        # Defining constants
        ca = 0.70710678
        sa = 0.70710678
        c_ = [
            0.1220,
            0.05669,
            0.003246,
            0.00006837,
            0.04388,
            0.002685,
            0.00007918,
            0.00002289]
        osm = 0.1151

        def formula(ksm, khzz, kazz):
            return 4 * osm * ((ca**4) * ((ksm**4) + c_[0] * (ksm**3) * khzz + c_[1] * (ksm**2) * (khzz**2) + c_[2] * ksm * (khzz**3) + c_[3] * (khzz**4)) + (
                ca**2) * (sa**2) * (c_[4] * (ksm**2) * (kazz**2) + c_[5] * ksm * khzz * (kazz**2) + c_[6] * (khzz**2) * (kazz**2)) + (sa**4) * c_[7] * (kazz**4))

        cross_sections = []
        for sample in self.basis:
            ksm = sample[0]
            khzz = 16.247 * sample[1]
            kazz = 16.247 * sample[2]
            cross_sections.append(formula(ksm, khzz, kazz))

        return cross_sections
    
    def compute_one_alpha_part(self, weights, xs):
        c1s_1 = np.multiply(weights,xs)
        c1s_1 = np.multiply(weights,c1s_1)
        alpha1 = c1s_1.sum()
        return alpha1
    
    def computeWeighted(self, bases, samples):
        n_effs = np.zeros(2)
        alpha = np.zeros(2)
        for i,basis in enumerate(bases):
            self.resetTarget(self.morphed)
            basis = [samples[b] for b in basis]
            self.resetBasis(basis)
            weights = np.array(self.getWeights())
            cross_sections = np.array(self.getCrossSections())
            n_tot = (np.abs(np.multiply(weights, cross_sections))).sum()
            n_eff = (np.multiply(weights, cross_sections)).sum()
            n_effs[i] = np.abs(n_eff/n_tot)
            alpha[i] = self.compute_one_alpha_part(weights,cross_sections) 
        alpha[0] /= (alpha[0] + alpha[1])
        alpha[1] /= (alpha[0] + alpha[1])
        return alpha[0]*n_effs[0] + alpha[1]*n_effs[1]
 

    def computeStats(self, basis, samples):
        # Compute n_eff, condition number and matrix determinant of solution
        basis = [samples[b] for b in basis]
        self.resetBasis(basis)
        weights = np.array(self.getWeights())
        cross_sections = np.array(self.getCrossSections())
        n_tot = (np.abs(np.multiply(weights, cross_sections))).sum()
        n_eff = (np.multiply(weights, cross_sections)).sum()
        
        return (np.abs(n_eff / n_tot), self.det, self.cond)

    
    def computeNeff(self, basis, samples):
        '''
            N_eff computation
        '''
        self.resetTarget(self.morphed)
        basis = [samples[b] for b in basis]
        self.resetBasis(basis)
        weights = np.array(self.getWeights())
        cross_sections = np.array(self.getCrossSections())
        
        n_eff = (np.multiply(weights, cross_sections)).sum()
        n_tot = (np.abs(np.multiply(weights, cross_sections))).sum()
        
        return np.abs(n_eff / n_tot)
    
    def evalMaxPair(self, x, samples, cvalues_1, cvalues_2, verb=False, alpha = False):
        '''
            Loss function
        '''
        val = 0.
        target = self.morphed[:]
        norm = len(cvalues_1) * len(cvalues_2)
        result = 0.
        # Sum of n_eff for each pair of samples in grid (cvalues_1 x cvalues_2)
        for val1 in cvalues_1:
            for val2 in cvalues_2:
                target[1] = val1
                target[2] = val2
                self.resetTarget(target)
                if alpha == False:
                    r1 = self.computeNeff(x[0], samples)
                    r2 = self.computeNeff(x[1], samples)
                    result += r1 + r2
                else:
                    result += self.computeWeighted(x,samples)
        r1, det1, cond1 = self.computeStats(x[0], samples)
        r2, det2, cond2 = self.computeStats(x[1], samples)
        val = result / (norm)
        # Check that condition number of the solution is not too bad
        if cond1 > 100000. or cond2 > 100000.:
            val = 0.
        else:
            print val
        return val
    
    def evaluateInd(self, individual, cvalues_1, cvalues_2):
        '''
            Loss function for evo algorithm
        '''
        size = len(individual) / 2
        ind1, ind2 = np.array(individual[:size]), np.array(individual[size:])
        
        # Check that individual is a valid sample
        assert sum(ind1) == 15
        assert sum(ind2) == 15
        assert sum(np.logical_and(ind1, ind2)) == 0
        
        # Convert from evolutionary alg representation ([0 1 0 1 ...]) to 
        # list of bases ([1,3,..],[12,15,..])
        p = map(list, (np.where(ind1 == 1)[0], np.where(ind2 == 1)[0]))
        # Evaluate Loss
        result = self.evalMaxPair(p, self.samples, cvalues_1, cvalues_2, alpha = self.use_alpha)
        return result,

    # Workaround for parallel processing
    def __call__(self, individual, cvalues_1, cvalues_2):
        return self.evaluateInd(individual, cvalues_1, cvalues_2)

    def evolMorphing(
            self,
            target=None,
            cvalues_1=None,
            cvalues_2=None,
            c_eval=0):
        ''' 
            Evolutionary morphing. 
        '''

        toolbox = base.Toolbox()
        creator.create(
            "FitnessMax",
            base.Fitness,
            weights=(
                1.0,
            ),
            typecode='d')
        
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox.register(
            "individual",
            initInd,
            creator.Individual,
            size=15,
            samples=self.samples)

        #toolbox.register('evaluate', self.evaluateInd, cvalues_1=cvalues_1, cvalues_2=cvalues_2)
        # Here I define self as the fitness function ... then __call__ will be called and then self.evaluateInd
        toolbox.register(
            'evaluate',
            self,
            cvalues_1=cvalues_1,
            cvalues_2=cvalues_2)

        # Defining algorithm operators
        toolbox.register('mutate', mutateInd)
        toolbox.register('mate', cxOver, section=3, base_size=15)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        #toolbox.register("select", tools.selRoulette)
        toolbox.register(
            "population",
            tools.initRepeat,
            list,
            toolbox.individual)
        
        # Parallel processing
        pool = multiprocessing.Pool()
        #toolbox.register("map", pool.map)

        halloffame = tools.HallOfFame(maxsize=10)

        mu_es, lambda_es = 3, 21

        pop_stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        pop_stats.register('avg', np.mean)
        pop_stats.register('max', np.max)
        pop = toolbox.population(n=50)
        start = time.time()
        
        # Running the evolutionary algorithm
        pop, logbook = algorithms.eaSimple(
            pop, toolbox, cxpb=0.5, mutpb=0.5, ngen=500, stats=pop_stats, halloffame=halloffame)

        end = time.time()
        halloffame.update(pop)

        individual = halloffame[0]
        size = len(individual) / 2
        ind1, ind2 = np.array(individual[:size]), np.array(individual[size:])
        # Check that individual is a valid sample
        assert sum(ind1) == 15
        assert sum(ind2) == 15
        assert sum(np.logical_and(ind1, ind2)) == 0
        
        p = map(list, (np.where(ind1 == 1)[0], np.where(ind2 == 1)[0]))
        print self.evalMaxPair(p, self.samples, cvalues_1, cvalues_2, alpha=self.use_alpha)

        best = p
        self.mem_values.clear()
        self.resetTarget(self.morphed)
        result, det, cond = self.computeStats(best[0], self.samples)
        print 'Result: {0} ,Det: {1}, Cond: {2}'.format(result, det, cond)
        result, det, cond = self.computeStats(best[1], self.samples)
        print 'Result: {0} ,Det: {1}, Cond: {2}'.format(result, det, cond)
        print 'elapsed time : {0}'.format(end - start)
        return p
