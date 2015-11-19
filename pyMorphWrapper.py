'''
Simple python Wrapper for EFT Morphing code

author: jpavezse
'''

from ctypes import *


#reading C++ library
lib = cdll.LoadLibrary('./RandomEFT/RandomEFT/libcMorphWrapper.so')
lib.getWeights.restype = c_char_p
lib.getCrossSections.restype = c_char_p

class MorphingWrapper:
  def __init__(self):
    # Initialize the C object
    self.obj = lib.MorphingWrapperNew()

  def setStringData(self,data):
    # Set samples data in the form of string: 
    #'Nsamples Ncouplings types[P|D|S]*Ncouplings target sample1 sample2 ...'
    # where target are the N couplings of the sample to morph and sample* 
    # N couplings of each sample

    s_data = create_string_buffer(data)
    lib.setSampleData(self.obj,s_data)
  
  def setSampleData(self,nsamples,ncouplings,types,morphed,samples):
    self.nsamples = nsamples
    self.ncouplings = ncouplings
    self.types = types
    self.morphed = morphed
    self.samples = samples
    string_data = '{0} {1} '.format(nsamples,ncouplings)
    string_data = string_data + ' '.join(str(x) for x in types) + ' '
    string_data = string_data + ' '.join(str(x) for x in morphed) + ' '
    string_data = string_data + ' '.join(str(x) for samples in samples_values for x in samples)
    self.setStringData(string_data)


  def getWeights(self):
    # Return computed weights for each sample

    s = lib.getWeights(self.obj)
    return [float(x) for x in s.split()]

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

    print self.samples
    cross_sections = []
    for sample in self.samples:
      ksm = sample[0]
      khzz = 16.247*sample[1]
      kazz = 16.247*sample[2]
      cross_sections.append(formula(ksm,khzz,kazz))

    return cross_sections

ncouplings = 3
nsamples = 15

samples_values = [[1.,0.,17.],[1.,1./3.,19.],[1.,1./5.,23.],[1.,1./7.,29.],[1.,1./11.,31.],
                 [1.,1./13.,37.],[1.,1./17.,41.],[1.,1./19.,43.],[1.,1./23.,47.],[1.,1./29.,73.],
                 [1.,1./31.,53.],[1.,1./41.,57.],[1.,1./43.,59.],[1.,1./47.,67.],[1.,1./53.,71.]];

#Test Cross Sections
#nsamples = 5
#samples_values = [[1.,-3./2.,-3./2.],[1.,-1.,-1./2.],[1.,-1.,0.],[1.,-1.,1./2.],[1.,0.,-1./2.],
#              [1.,0.,1./2.]]

types = ['S', 'S','S']
morphed = [1.,0.5,-3.]

morph = MorphingWrapper()
morph.setSampleData(nsamples,ncouplings,types,morphed,samples_values)
#s = morph.getWeights()
cs = morph.getCrossSections()

#print s
print cs
