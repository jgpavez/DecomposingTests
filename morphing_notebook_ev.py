'''

 How to use Distributiond Evolutionary
 EFT Morphing

'''

import numpy as np
import pdb 
from pyMorphWrapper import MorphingWrapper
    
import matplotlib.pyplot as plt
import matplotlib.cm as cm

if __name__ == "__main__":

    # Define fitting ranges
    g1_range = (-0.3557,0.2646)
    g2_range = (-0.34467,0.34467)
    
    # Intervals for each axis
    npoints = 15
    
    csarray1 = np.linspace(g1_range[0],g1_range[1],npoints)
    csarray2 = np.linspace(g2_range[0], g2_range[1], npoints)
    
    #[(5, 24, 32, 19, 4, 12, 22, 3, 33, 13, 35, 1, 7, 14, 11), (10, 28, 31, 17, 0, 20, 30, 6, 27, 15, 8, 18, 9, 25, 2)] 0.8-1.2
    
    # nsamples, ncomb
    nsamples = 15
    ncomb = 22
    # List of availables basis samples
    '''
    available_samples = [[1.0, 1.0, 0.5], [1.0, 0.0, -1.0], [1.0, 0.0, -0.5], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0], 
                         [1.0, 1.0, -0.5], [1.0, 0.5, 0.3333333333333333], [1.0, 0.3333333333333333, 0.2], 
                         [1.0, 0.3333333333333333, 0.14285714285714285], [1.0, 0.25, 0.16666666666666666], 
                         [1.0, 2.0, 0.0], [1.0, 2.0, 1.0], [1.0, 2.0, 2.0], [1.0, 2.0, -1.0], [1.0, -1.0, 0.0],
                         [1.0, -1.0, 0.5], [1.0, -1.0, -0.5], [1.0, -0.5, -0.3333333333333333], 
                         [1.0, -0.3333333333333333, -0.2], [1.0, -0.3333333333333333, -0.14285714285714285], 
                         [1.0, -0.25, -0.16666666666666666], [1.0, -2.0, 0.0], [1.0, -2.0, 1.0], [1.0, -2.0, -1.0], 
                         [1.0, -2.0, -2.0], [1.0, -1.5, -1.5]]
    print available_samples
    '''
    available_samples = [[1.0,cs1,cs2] for cs1 in np.linspace(-0.5,0.5,8) for cs2 in np.linspace(-0.5,0.5,8)]
    
    # Compute both bases
    morph = MorphingWrapper()
    # Define number of samples, number of couplings, types (S,P,D) and available samples
    # Using half of range as initial target (used only to make computation faster)
    target = [1.,0.,0.]
    morph.setSampleData(nsamples=nsamples,ncouplings=3,types=['S','S','S'],samples=available_samples,
          ncomb=ncomb,used_samples=36,morphed=target)
    # Obtain the bases by using smooth dynamic morphing
    
    indexes = morph.evolMorphing(cvalues_1 = csarray1,cvalues_2 = csarray2)
    # Obtain the bases by using smooth dynamic morphing
    #indexes = morph.dynamicMorphing(cvalues_1 = csarray1,cvalues_2 = csarray2)
    
    print indexes 
    # Best pair found using 11x11 samples and 2500 iterations in the evolutionary algorithm
    #indexes = [[4, 12, 17, 26, 31, 36, 51, 56, 60, 69, 75, 84, 104, 110, 120], [10, 11, 13, 16, 30, 37, 43, 46, 50, 59, 74, 82, 88, 91, 108]]
    #0.89, 11
    #indexes = [[0, 7, 11, 13, 18, 20, 22, 25, 27, 29, 42, 44, 55, 59, 61], [1, 3, 6, 19, 21, 23, 24, 26, 28, 35, 38, 41, 45, 58, 63]]
    # 0.88, 8
    
    # Save cross sections and couplings for each one of the points on the fitting space
    # Also compute the weighted n_eff
    for l,ind in enumerate(indexes): 
      ind = np.array(ind)
      morph.resetBasis([available_samples[int(k)] for k in ind]) 
      sorted_indexes = np.argsort(ind)
      indexes[l] = ind[sorted_indexes]
      for i,cs in enumerate(csarray1):
        for j,cs2 in enumerate(csarray2):
          target[1] = cs
          target[2] = cs2 
          morph.resetTarget(target)
            # Compute weights and cross section of each sample
          couplings = np.array(morph.getWeights())
          cross_section = np.array(morph.getCrossSections())
          couplings,cross_section = (couplings[sorted_indexes],
                            cross_section[sorted_indexes])
          # Save list of cross sections and weights for each samples and orthogonal bases
          all_couplings = np.vstack([all_couplings,couplings])  if i <> 0 or j <> 0 or l <> 0 else couplings
          all_cross_sections = np.vstack([all_cross_sections, cross_section]) if i <> 0 or j <> 0 or l <> 0 else cross_section
            
            
    # Now compute and plot the weighted n_eff in order to evaluate the models
    alpha = np.zeros([csarray1.shape[0],csarray2.shape[0],2])
    n_eff_ratio = np.zeros((csarray1.shape[0], csarray2.shape[0]))
    for i,cs in enumerate(csarray1):
      for j, cs2 in enumerate(csarray2):
        target[1] = cs
        target[2] = cs2
        print '{0} {1}'.format(i,j)
        print target
    
        c1s_1 = all_couplings[i*npoints + j]
        cross_section_1 = all_cross_sections[i*npoints + j]
        c1s_1 = np.multiply(c1s_1,cross_section_1)
        n_eff = c1s_1.sum()
        n_tot = np.abs(c1s_1).sum()
        n_eff_1 = n_eff / n_tot
    
        c1s_2 = all_couplings[npoints*npoints + i*npoints + j]
        cross_section_2 = all_cross_sections[npoints*npoints + i*npoints + j]
        c1s_2 = np.multiply(c1s_2,cross_section_2)
        n_eff = c1s_2.sum()
        n_tot = np.abs(c1s_2).sum()
        n_eff_2 = n_eff / n_tot
    
        # Compute weights for bases
        neff2 = 1./n_eff_2
        neff1 = 1./n_eff_1
        alpha1 = np.exp(-np.sqrt(neff1))
        alpha2 = np.exp(-np.sqrt(neff2))
        alpha[i,j,0] = alpha1/(alpha1 + alpha2)
        alpha[i,j,1] = alpha2/(alpha1 + alpha2)
    
        # Compute Bkg weights
        n_eff_ratio[i,j] = (alpha[i,j,0]*n_eff_1 + alpha[i,j,1]*n_eff_2)
        #if n_eff_ratio[i,j] < 0.25:
        #    n_eff_ratio[i,j] = 0.
    
        print 'Weighted eff for ({0},{1}): {2}'.format(cs,cs2,n_eff_ratio[i,j])
    
    fig,ax = plt.subplots()
        
    A = [available_samples[ind][1] for ind in indexes[0]]
    B = [available_samples[ind][2] for ind in indexes[0]]
    A2 = [available_samples[ind][1] for ind in indexes[1]]
    B2 = [available_samples[ind][2] for ind in indexes[1]]
    
    ax.set_title('Samples position')
    ax.set_xlabel('Kazz')
    ax.set_ylabel('Khzz')    
        
    vals = np.flipud(n_eff_ratio)
    im = plt.imshow(vals, extent=(csarray2.min(), csarray2.max(), csarray1.min(),csarray1.max()),interpolation='nearest', cmap=cm.gist_rainbow_r)
    CB = plt.colorbar(im, shrink=0.8, extend='both')
    
    plt.plot(A,B,'ro')
    plt.plot(A2,B2,'bo')
    for xy in zip(A, B):                                               
        ax.annotate('({0:.2f},{1:.2f})'.format(xy[0],xy[1]), xy=xy, textcoords='offset points') 
    for xy in zip(A2, B2):                                               
        ax.annotate('({0:.2f},{1:.2f})'.format(xy[0],xy[1]), xy=xy, textcoords='offset points') 
    
    plt.savefig('morph/evmorph_khzz_kazz_9_07_07_pop100_2.png')
    


