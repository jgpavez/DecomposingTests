## Decomposing tests between mixture models into their components 

This work is based on the section **5.4 Decomposing tests between mixture models into their components** 
of the paper [Approximating Likelihood Ratios with Calibrated Discriminative Classifiers]
(http://arxiv.org/abs/1506.02169) by Kyle Cranmer.

The analysis is divided in the next sections:
* Simple Case (1D)
  * Identifying the signal coefficient by fitting (1D)
* N-dimensions
  * Varying the signal presence
  * Identifying the signal coefficient by fitting (ND)
* N-dimensions Private Model
  * Identifying the signal coefficient by fitting (Private)
  * How training affect ratios
* Morphing Data for VBF Higgs production with 1 BSM coupling
* Dynamic Morphing Method for VBF Higgs production with 2BSM couplings


## Simple Case
We start with a simple model composed of three 1-Dim pdfs. This allow us to check visually the quallity of the 
Ratios. The pdfs are shown next

![decomposed model](https://github.com/jgpavez/systematics/blob/master/plots/mlp/1Dim/0.10/decomposed_model.png)

Those pdfs forms two mixture models, as shown in formula (21) of the paper of reference.
The first distribution correspond to the null signal hypothesis, meanwhile the second one 
correspond to the mixture model including the signal (in this case the signal corresponds to f0). 
Both distributions for coefficients **c0 = [ 0.,0.3,0.7]** and **c1 = [ 0.1,0.27,0.63]**
are shown in the next image.

![Full Model](https://github.com/jgpavez/systematics/blob/master/plots/mlp/1Dim/0.10/full_model.png)

We will use a simple *MLP* as the classifier. The *MLP* have shown good results in previous experiments 
([Link](https://github.com/jgpavez/systematics/blob/master/plots/comp_train_all_ratio.png)).

We will compare each ratio obtained by training pair-wise to the real ratio of each pair of functions.

![Decomposed Ratios](https://github.com/jgpavez/systematics/blob/master/plots/mlp/1Dim/0.10/all_dec_train_mlp_ratio.png)

The ROC curves obtained by varying the threshold on the trained and true ratios of each pair of functions are shown in the next image.

![Decomposed ROC](https://github.com/jgpavez/systematics/blob/master/plots/mlp/1Dim/0.10/all_comparison_mlp_roc.png)

In the next image the composed ratio using the formula (24) is compared to the ratio obtained by training the classifier 
in the full model and to the true ratio.

![All Ratios](https://github.com/jgpavez/systematics/blob/master/plots/mlp/1Dim/0.10/all_train_mlp_ratio.png)

Next, the Signal Efficiency - Background Rejection curves of each one of the ratios (composed, full trained and full truth) is shown.

![All ROC](https://github.com/jgpavez/systematics/blob/master/plots/mlp/1Dim/0.10/comp_all_mlp_sigbkg.png)

# Varying the signal presence 

We want to check how the composed and the full strategy are affected when the value of the 
signal coefficient become smaller.

In the next image the mixture models for coefficients for signal of **[0.1,0.05,0.01,0.005]** are 
shown

 0.10                   | 0.05
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/1Dim/0.10/full_model.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/1Dim/0.05/full_model.png" width="350" >
 0.01                   | 0.005
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/1Dim/0.01/full_model.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/1Dim/0.005/full_model.png" width="350" >

Now, we compare the real ratios, the composed ratios and the full trained ratios for each 
one of the cases

 0.10                   | 0.05
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/1Dim/0.10/all_train_mlp_ratio.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/1Dim/0.05/all_train_mlp_ratio.png" width="350" >
 0.01                   | 0.005
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/1Dim/0.01/all_train_mlp_ratio.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/1Dim/0.005/all_train_mlp_ratio.png" width="350" >

Next, the score histogram for each one of the pair-wise trained classifiers for signal 
and background is shown, notice that only histograms for k < j is shown (those plots are independant of the signal ratio)

![Scores 1Dim](https://github.com/jgpavez/systematics/blob/master/plots/mlp/1Dim/0.10/dec0_all_mlp_hist.png)

The ratio histograms for the composite, full trained and true cases is shown in the next image, those histograms are constructed over data sampled from the distribution of F0 background and f0 signal.

 0.10                   | 0.05
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/1Dim/0.10/ratio_comparison_mlp_hist.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/1Dim/0.05/ratio_comparison_mlp_hist.png" width="350" >
 0.01                   | 0.005
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/1Dim/0.01/ratio_comparison_mlp_hist.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/1Dim/0.005/ratio_comparison_mlp_hist.png" width="350" >


Finally, in the next image the Signal Efficiency - Background Rejection curves for the composed, full trained and true ratio are shown for each one of the cases

 0.10                   | 0.05
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/1Dim/0.10/comp_all_mlp_sigbkg.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/1Dim/0.05/comp_all_mlp_sigbkg.png" width="350" >
 0.01                   | 0.005
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/1Dim/0.01/comp_all_mlp_sigbkg.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/1Dim/0.005/comp_all_mlp_sigbkg.png" width="350" >

It can be seen that for very low signal presence the composed ratios are still working perfectly, on the other hand the full trained MLP is not able to reproduce the ratio at all.

## Identifying the signal coefficient by fitting

What we want to check now if it is possible to identify the signal coefficient 
**c1[0]**, leaving the c1[1]/c1[2] ratio constant, by using the likelihood of the ratios.
To do this we first train the decomposed model on data with a defined c1[0].
After doing this we use this model and compute the composed likelihood ratio for different 
values of c1[0] and the data obtained using the previously fixed c1[0]. It will be expected
 that the minimum corresponds to the real c1[0]. 

First, We will check how the likelihood are affected by the number of data generated to compute the likelihoods, for the case c1[0] = 0.05.

 100                   | 500
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/1Dim/0.05/100/comp_train_mlp_likelihood.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/1Dim/0.05/500/comp_train_mlp_likelihood.png" width="350" >
 1000                   | 5000
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/1Dim/0.05/1000/comp_train_mlp_likelihood.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/1Dim/0.05/5000/comp_train_mlp_likelihood.png" width="350" >


Now, we will check if it is possible to identify both, the signal coefficient c1[0] and the 
background coefficient c1[1] (using c1[2] = 1.-c1[0]-c1[1]). In this case we are using 100,500,1000 and 5000 samples to compute the likelihood and values for the coefficients of c1[0] = 0.05 and c1[1] = 0.285. Truth likelihoods are compared to trained Likelihoods.

 100                  | 500 
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/1Dim/0.05/100/comp_train_mlp_multilikelihood.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/1Dim/0.05/500/comp_train_mlp_multilikelihood.png" width="350" >
 1000                   | 5000 
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/1Dim/0.05/1000/comp_train_mlp_multilikelihood.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/1Dim/0.05/5000/comp_train_mlp_multilikelihood.png" width="350" >

It can be seen that again the method do a very good job on identifying the correct values for the coefficients.

#N-dimensions

Now, we will check the composition method in a 10-dim mixture model. 
The pdfs forms two mixture models, as shown in formula (21) of the paper of reference.
The first distribution correspond to the null signal hypothesis, meanwhile the second one 
correspond to the mixture model including the signal (in this case the signal corresponds to f0). 
In this case each pdf is a multivariate gaussian, the first three features of this gaussians are shown in the next
image
 
![decomposed model](https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.10/decomposed_model.png)

Those pdfs forms two mixture models, as shown in formula (21) of the paper of reference.
The first distribution correspond to the null signal hypothesis, meanwhile the second one 
correspond to the mixture model including the signal (in this case the signal corresponds to f0). 
Both distributions for coefficients **c0 = [ 0.,0.3,0.7]** and **c1 = [ 0.1,0.27,0.63]**
are shown in the next image.

![Full Model](https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.10/full_model.png)


Both distributions are represented with scatter plots for a subset of features (indicated in the columns and rows of the grids of plots) and for each 
one of the pairwise combinations (1-2,1-3 and 2-3).


# f1-f2
![scatter grid 1](https://github.com/jgpavez/systematics/blob/master/plots/mlp/private/dec_truth_0_1_grid.png)

#f1-f3
![scatter grid 2](https://github.com/jgpavez/systematics/blob/master/plots/mlp/private/dec_truth_0_2_grid.png)

#f2-f3
![scatter grid 3](https://github.com/jgpavez/systematics/blob/master/plots/mlp/private/dec_truth_1_2_grid.png)

Next, a parallel coordinates plot of the distributions F0 background - f0 signal is shown

![parallel](https://github.com/jgpavez/systematics/blob/master/plots/mlp/paralell_coordinates_F0-f0.png)

First, the ROC curves obtained by varying the threshold on the trained and true ratios on each pair of 
functions are shown in the next image.

![Decomposed ROC](https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.10/dec_comparison_mlp_roc.png)

Next, the Signal Efficiency - Background Rejection curves of each one of the ratios (composed, full trained and full truth) is shown.

![All ROC](https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.10/full_comparison_mlp_sigbkg.png)

# Varying the signal presence 

We want to check how the composed and the full strategy are affected when the value of the 
signal coefficient become smaller.

First, the score histogram for each one of the pair-wise trained classifiers for signal 
and background is shown, notice that only histograms for k < j is shown

![Scores 10Dim](https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.10/dec0_all_mlp_hist.png)

 0.10                   | 0.05
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.10/dec0_all_mlp_hist.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.05/dec0_all_mlp_hist.png" width="350" >
 0.01                   | 0.005
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.01/dec0_all_mlp_hist.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.005/dec0_all_mlp_hist.png" width="350" >

The ratio histograms for the composite, full trained and true cases is shown in the next image, those histograms are constructed over data sampled from the distribution of F0 background and f0 signal.

 0.10                   | 0.05
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.10/ratio_comparison_mlp_hist.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.05/ratio_comparison_mlp_hist.png" width="350" >
 0.01                   | 0.005
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.01/ratio_comparison_mlp_hist.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.005/ratio_comparison_mlp_hist.png" width="350" >


Finally, in the next image the Signal Efficiency - Background Rejection curves for the composed, full trained and true ratio are shown for each one of the cases

 0.10                   | 0.05
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.10/comp_all_mlp_sigbkg.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.05/comp_all_mlp_sigbkg.png" width="350" >
 0.01                   | 0.005
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.01/comp_all_mlp_sigbkg.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.005/comp_all_mlp_sigbkg.png" width="350" >

For this N dimensional case all ratios behave better, still is clear than for smaller signal coefficient the trained ratios are working almost perfectly for the composed 
case.

## Identifying the signal coefficient by fitting

What we want to check now if it is possible to identify the signal coefficient 
**c1[0]**, leaving the **c1[1]/c1[2]** ratio constant, by using the likelihood of the ratios.
To do this we first train the decomposed model on the N-dim data with a defined **c1[0]**.
After doing this we use this model and compute the composed likelihood ratio for different 
values of **c1[0]** and the data obtained using the previously fixed **c1[0]**. It will be expected
 that the minimum corresponds to the real **c1[0]**. 

First, We will check how the likelihood are affected by the number of data generated to compute the likelihoods, for the case c1[0] = 0.05.

 100                   | 500
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.05/100/comp_train_mlp_likelihood.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.05/500/comp_train_mlp_likelihood.png" width="350" >
 1000                   | 5000
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.05/1000/comp_train_mlp_likelihood.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.05/5000/comp_train_mlp_likelihood.png" width="350" >


Now, we will check if it is possible to identify both, the signal coefficient **c1[0]** and the 
background coefficient **c1[1]** (using c1[2] = 1.-c1[0]-c1[1]). In this case we are using 100,500,1000 and 5000 samples to compute the likelihood and values for the coefficients of c1[0] = 0.05 and c1[1] = 0.285. Truth likelihoods are compared to trained Likelihoods.

 100                  | 500 
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.05/100/comp_train_mlp_multilikelihood.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.05/500/comp_train_mlp_multilikelihood.png" width="350" >
 1000                   | 5000 
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.05/1000/comp_train_mlp_multilikelihood.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.05/5000/comp_train_mlp_multilikelihood.png" width="350" >

It can be seen that again the method do a very good job on identifying the correct values for the coefficients.

Next we will check if the estimated **c1[0]** and **c1[1]** are unbiased estimators of the 
real values. In the next images histograms of the estimated values for the likelihood 
obtained by the true and composed methods and values of **c1[0] = 0.05** and **c1[1] = 0.285** 
 are shown. 

 c1                  | c2
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.05/c1c2_train_mlp_c1_hist.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.05/c1c2_train_mlp_c2_hist.png" width="350" >

It can be seen that both methods obtain unbiased estimators of the true values.



#N-dimensions Private Model

We will check the composition method in a N-dim mixture model. But this time each pdf is also a sum of gaussians with fixed coefficients. 
We want to see how the method works in this harder case.

In this case each pdf is a sum of three different multivariate gaussians, the first three features of this model are shown in the next
image
 
![decomposed model](https://github.com/jgpavez/systematics/blob/master/plots/mlp/private/0.10/decomposed_model.png)

Those pdfs forms two mixture models, as shown in formula (21) of the paper of reference.
The first distribution correspond to the null signal hypothesis, meanwhile the second one 
correspond to the mixture model including the signal (in this case the signal corresponds to f0). 
Both distributions for coefficients **c0 = [ 0.,0.3,0.7]** and **c1 = [ 0.1,0.27,0.63]**
are shown in the next image.

![Full Model](https://github.com/jgpavez/systematics/blob/master/plots/mlp/private/0.10/full_model.png)



The following images show scatter plots between a subset of features for the pairwise combinations 1-2, 1-3 and 2-3
the number of the feature is indicated in the columns and rows of the grid of plots.

# f1-f2
![scatter grid 1](https://github.com/jgpavez/systematics/blob/master/plots/mlp/private/dec_truth_0_1_grid.png)

#f1-f3
![scatter grid 2](https://github.com/jgpavez/systematics/blob/master/plots/mlp/private/dec_truth_0_2_grid.png)

#f2-f3
![scatter grid 3](https://github.com/jgpavez/systematics/blob/master/plots/mlp/private/dec_truth_1_2_grid.png)

Next, a parallel coordinates plot of the distributions F0 background - f0 signal is shown

![parallel](https://github.com/jgpavez/systematics/blob/master/plots/mlp/private/paralell_coordinates_F0-f0.png)

Now we will to check how the composed and the full strategy are affected when the value of the 
signal coefficient become smaller for this *private* model.

In the next image the mixture models for coefficients for signal of **[0.1,0.05,0.01,0.005]** are 
shown for one of the features (this allow to see the amount of signal in each model).

 0.10                   | 0.05
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/private/0.10/full_model.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/private/0.05/full_model.png" width="350" >
 0.01                   | 0.005
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/private/0.01/full_model.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/private/0.005/full_model.png" width="350" >

Next, the score histogram for each one of the pair-wise trained classifiers for signal 
and background is shown, notice that only histograms for k < j is shown

![Scores private](https://github.com/jgpavez/systematics/blob/master/plots/mlp/private/0.10/dec0_all_mlp_hist.png)


The ratio histograms for the composite, full trained and true cases is shown in the next image, those histograms are constructed over data sampled from the distribution of F0 background and f0 signal.

 0.10                   | 0.05
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/private/0.10/ratio_comparison_mlp_hist.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/private/0.05/ratio_comparison_mlp_hist.png" width="350" >
 0.01                   | 0.005
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/private/0.01/ratio_comparison_mlp_hist.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/private/0.005/ratio_comparison_mlp_hist.png" width="350" >


Finally, in the next image the Signal Efficiency - Background Rejection curves for the composed, full trained and true ratio are shown for each one of the cases

 0.10                   | 0.05
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/private/0.10/comp_all_mlp_sigbkg.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/private/0.05/comp_all_mlp_sigbkg.png" width="350" >
 0.01                   | 0.005
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/private/0.01/comp_all_mlp_sigbkg.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/private/0.005/comp_all_mlp_sigbkg.png" width="350" >

It can be seen that even with this harder model where each single distribution is a sum of gaussians the composed ratios are working quite well even for small signal 
coefficients.


## Identifying the signal coefficient by fitting for the private model

We will check if even for this harder case it is possible to identify the signal and background 
coefficient by minimizing the likelihood. The method is the same that the one used in the previous 
section
First, We will check how the likelihood are affected by the number of data generated to compute the likelihoods, for the case c1[0] = 0.05.

 100                   | 500
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/private/0.05/100/comp_train_mlp_likelihood.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/private/0.05/500/comp_train_mlp_likelihood.png" width="350" >
 1000                   | 5000
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/private/0.05/1000/comp_train_mlp_likelihood.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/private/0.05/5000/comp_train_mlp_likelihood.png" width="350" >


Now, we will check if it is possible to identify both, the signal coefficient **c1[0]** and the 
background coefficient **c1[1]** (using c1[2] = 1.-c1[0]-c1[1]). In this case we are using 100,500,1000 and 5000 samples to compute the likelihood and values for the coefficients of c1[0] = 0.05 and c1[1] = 0.285. Truth likelihoods are compared to trained Likelihoods.

 100                  | 500 
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/private/0.05/100/comp_train_mlp_multilikelihood.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/private/0.05/500/comp_train_mlp_multilikelihood.png" width="350" >
 1000                   | 5000 
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/private/0.05/1000/comp_train_mlp_multilikelihood.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/private/0.05/5000/comp_train_mlp_multilikelihood.png" width="350" >

Again, the method works pretty good even with this harder model.

# How training affect ratios

Now, what we will study is how the quality of training affect the final ratios. We will keep a large amount of data to construct the likelihood ratios (30000 samples) but we will vary 
the amount of data used to train the classiffier. We will use 100, 1000, 10000, and 1000000 samples to train the classifier.

The histograms for the values of the coefficients c1[0] and c2[0] obtained by maximizing the log-likelihood on a dataset in each one of the cases are shown. 


c1[0] - 100                 | c1[0] - 1000
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/training/regularized/100/c1c2_train_mlp_c1_hist.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/training/regularized/1000/c1c2_train_mlp_c1_hist.png" width="350" >
c1[0] - 10000                | c1[0] - 100000
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/training/regularized/10000/c1c2_train_mlp_c1_hist.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/training/regularized/100000/c1c2_train_mlp_c1_hist.png" width="350" >

c1[1] - 100                   | c1[1] - 1000
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/training/regularized/100/c1c2_train_mlp_c2_hist.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/training/regularized/1000/c1c2_train_mlp_c2_hist.png" width="350" >
c1[1] - 10000                | c1[1] - 100000
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/training/regularized/10000/c1c2_train_mlp_c2_hist.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/training/regularized/100000/c1c2_train_mlp_c2_hist.png" width="350" >

For each one of the cases, the pairwise score distributions are shown next

 100                 | 1000
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/training/regularized/100/dec0_all_mlp_hist.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/training/regularized/1000/dec0_all_mlp_hist.png" width="350" >
 10000                | 100000
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/training/regularized/10000/dec0_all_mlp_hist.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/training/regularized/100000/dec0_all_mlp_hist.png" width="350" >

Also, the ROC curve for each pairwise case are shown next

 100                 | 1000
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/training/regularized/100/all_comparison_mlp_roc.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/training/regularized/1000/all_comparison_mlp_roc.png" width="350" >
 10000                | 100000
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/training/regularized/10000/all_comparison_mlp_roc.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/training/regularized/100000/all_comparison_mlp_roc.png" width="350" >


And the ratio histograms are shown next,

 100                   | 1000
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/training/0/ratio_comparison_mlp_hist.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/training/1000/ratio_comparison_mlp_hist.png" width="350" >
 10000                | 100000
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/training/10000/ratio_comparison_mlp_hist.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/training/100000/ratio_comparison_mlp_hist.png" width="350" >



With a relatively small amount of samples, the trained likelihood is able to approximate closely the true value of the coefficients **c1[0]** and **c1[1]** with only a small bias. 
With more data the approximation is very good.

Now we will check a sighly easier model, just moving a little the signal distribution in a way that the NN can recognize it easier.

c1[0] - 100                 | c1[0] - 500
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/training/easier/100/c1c2_train_mlp_c1_hist.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/training/easier/500/c1c2_train_mlp_c1_hist.png" width="350" >
c1[0] - 1000                | c1[0] - 10000
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/training/easier/1000/c1c2_train_mlp_c1_hist.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/training/easier/10000/c1c2_train_mlp_c1_hist.png" width="350" >

c1[1] - 100                   | c1[1] - 500
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/training/easier/100/c1c2_train_mlp_c2_hist.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/training/easier/500/c1c2_train_mlp_c2_hist.png" width="350" >
c1[1] - 1000                | c1[1] - 10000
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/training/easier/1000/c1c2_train_mlp_c2_hist.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/training/easier/10000/c1c2_train_mlp_c2_hist.png" width="350" >

Also, for each one of the cases, the pairwise score distributions are shown next

100                   | 500
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/training/easier/100/dec0_all_mlp_hist.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/training/easier/500/dec0_all_mlp_hist.png" width="350" >
1000                | 10000
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/training/easier/1000/dec0_all_mlp_hist.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/training/easier/10000/dec0_all_mlp_hist.png" width="350" >

The ROC curves for each one of the cases: 

100                   | 500
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/training/easier/100/all_comparison_mlp_roc.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/training/easier/500/all_comparison_mlp_roc.png" width="350" >
1000                | 10000
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/training/easier/1000/all_comparison_mlp_roc.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/training/easier/10000/all_comparison_mlp_roc.png" width="350" >


Finally the ratio histograms for this model are shown next,
 
100                   | 500
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/training/easier/100/ratio_comparison_mlp_hist.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/training/easier/500/ratio_comparison_mlp_hist.png" width="350" >
1000                | 10000
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/training/easier/1000/ratio_comparison_mlp_hist.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/training/easier/10000/ratio_comparison_mlp_hist.png" width="350" >


## Morphing Data for VBF Higgs production with 1 BSM coupling

Using [Morphing Methods](https://cds.cern.ch/record/2065188) one can construct an arbitrary sample using a set of base samples related by a set of coupling constants. The morphed sampled can be written as 
**S(g1,g2,...) = sum(Wi(g1,g2,...)Si(g1,g2,...))**, where g1,g2 are coupling constants, Si input samples and Wi its weights. The minimum number of input samples needed is related with the 
number of shared coupling constants in production and decay and the number of coupling constants only in production or decay. The morphed sample it is a mixture model and the ratio between 
two morphed samples **S(g1,g2,...)** and **S(g1',g2',...)** can be decomposed and approximated using discriminative classifiers as shown before. 

We will use data from VBF Higgs production with 1 
BSM coupling. For this only 5 base samples are needed (a good selection of base samples is needed in order to minimize statistical uncertainty). The base samples used for morphing are **S(1,0),S(1,2), S(1,1), S(1,3), S(0,1)**. As validation sample we use **S(1.,1.5)**. We also know that the weights corresponding to the validation sample with the base samples mentioned before are *W = [-0.0625, 0.5625, 0.5625, -0.0625, 0.5625]* and the cross section values for each base sample is *[0.1149,8.469,1.635, 27.40, 0.1882]*.
 
Some of the pairwise distributions are represented using scatter plots for a subset of features (indicated in the column and row of the grid of plots). 

 S(1,0)-S(0,1)  | S(1,0)-S(1,1)
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/master/plots/xgboost/ggf/dec_truth_S10_S01_grid.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/xgboost/ggf/dec_truth_S10_S11_grid.png" width="350" >
 S(1,0)-S(1,2)  |  S(1,1)-(0,1)
<img src="https://github.com/jgpavez/systematics/blob/master/plots/xgboost/ggf/dec_truth_S10_S12_grid.png" width="350">  |<img src="https://github.com/jgpavez/systematics/blob/master/plots/xgboost/ggf/dec_truth_S11_S01_grid.png" width="350">

We start by training a **Boosted Decision Tree** (using the library *xgboost*) in each pair of samples. The score distribution obtained for each one of the pairs is shown next

 S(1,0)-S(1,2),S(1,0)-S(1,1),S(1,0)-S(1,3) | S(1,0)-S(0,1),S(1,2)-S(1,1),S(1,2)-S(1,3)
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/master/plots/xgboost/ggf/dec0_all_xgboost_hist.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/xgboost/ggf/dec1_all_xgboost_hist.png" width="350" >
 S(1,2)-S(0,1),S(1,1)-S(1,3),S(1,3)-S(0,1)  | 
<img src="https://github.com/jgpavez/systematics/blob/master/plots/xgboost/ggf/dec2_all_xgboost_hist.png" width="350">  |

Following, the ROC curves for each one of the pairwise trained classifiers, using the ratio as discriminative variable are shown.

 S(1,0)-S(1,2),S(1,0)-S(1,1),S(1,0)-S(1,3) | S(1,0)-S(0,1),S(1,2)-S(1,1),S(1,2)-S(1,3)
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/master/plots/xgboost/ggf/all0_comparison_xgboost_roc.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/xgboost/ggf/all1_comparison_xgboost_roc.png" width="350" >
 S(1,2)-S(0,1),S(1,1)-S(1,3),S(1,3)-S(0,1)  | 
<img src="https://github.com/jgpavez/systematics/blob/master/plots/xgboost/ggf/all2_comparison_xgboost_roc.png" width="350"> | 

To evaluate the classification capacity of the algorithmn we compare the decomposed method with a BDT trained on the full data on the sample *S(1.,1.5)* as signal and *S(1,0)* (only SM) as background. 
The Signal Efficiency - Background Rejection curves for the decomposed method and the full trained method are shown next.

![All ROC](https://github.com/jgpavez/systematics/blob/master/plots/xgboost/ggf/comp_all_xgboost_sigbkg.png)

In this case, the full trained classifier behave better than the pairwise, mainly because some of the pairwise distributions are very similar and hard to classify, more analysis must be done on this
since there is a lot of room for improvement of the training. Anyway, it should be noted that meanwhile the full trained classifier is only optimum for the sample *S(1.,1.5)* and must be retrained 
for any new sample, the pairwise classifier is optimum for any combination of samples and there is no need of retraining (only a change of coupling constants in the formula is needed).

By using the morphing and the decomposed trained classifiers we are able to identify the coupling constants of an arbitrary sample by using **Maximum Likelihood**. In this case we will fit the coefficients for the sample **S(1.,1.5)** keeping the background distribution **S(1,0)** constant. 

We start by fitting *g1* and keeping *g2=1.5* constant and *g2* by keeping *g1=1.* constant. The plots for the fit are shown next

 g1                         | g2
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/master/plots/xgboost/ggf/comp_train_mlp_likelihood_g1.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/xgboost/ggf/comp_train_mlp_likelihood_g2.png" width="350" >

Both fits are pretty close to the real values, we can check that both are unbiased estimator of the fitted values in the next histograms, using 300 pseudo samples of size 5000.

 g1                         | g2
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/master/plots/xgboost/ggf/g1_train_mlp_hist.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/xgboost/ggf/g2_train_mlp_hist.png" width="350" >

It can be seen that the fit is working very good for the values of g1 and g2. The method is able to identify with high precision the real value of the coefficients.

Finally, we will study if it is possible to fit both values *g1,g2* at the same time by using Maximum Likelihood. The contour plot for a likelihood fit in both parameters is shown

![fit_g1g2](https://github.com/jgpavez/systematics/blob/master/plots/xgboost/ggf/comp_train_mlp_multilikelihood.png)

The mean values of the fit of both values for 450 pseudo samples of size 5000 are shown next

![hist c0](https://github.com/jgpavez/systematics/blob/master/plots/xgboost/ggf/g1g2_train_mlp_hist_pix.png)

Both values are pretty close to the real values (1.0 and 1.5), this shows the great capacity of the method to identify values of parameters on real data distributions by using Maximum Likelihood. 



## Dynamic Morphing Method for VBF Higgs production with 2BSM couplings

In the VBF Higgs production channel 15 samples are needed in order to morph any sample. Each sample is represented by 3 coupling constants. 

A main issue when ussing when using the Morphing method to find the coupling constants is the coverture of the choosed base on the coulings space. Statstical fluctuations 
of the morphed sample can increase a lot in some parts of the coupling space when using a wrong samples base. Due to this using a single basis on all the fitting space is not possible. A solution would to have more samples that the 15 needed and to choose different basis for all points in the couplings space minimizing the statistical fluctiations for each sample. The problem is that the transition between basis affect the fitting procedure.

We propose to use the sum of two bases choosed to minimize the statistical fluctuations in all the fitting space (in this case this is equivalent to maximize the n_eff value which is *sum(cross_section x coupling)*). In order to minimize fitting problems due to the transition between bases, the sum is weighted with weight *sqrt(n_eff)* for each basis. Results have shown that this method allow a relatively good n_eff in all fitting space and also minimize problems due to transition between bases. 

The position on the coupling space of the samples used are shown in the next image


![samples](https://github.com/jgpavez/systematics/blob/master/plots/xgboost/couplings_space.png)

Initial fit results for some target samples are shown next.


target=(1.,0.5)            | target=(1.,1.)
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/master/plots/xgboost/comp_train_mlp_multilikelihood_1.00_0.50.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/xgboost/comp_train_mlp_multilikelihood_1.00_1.00.png" width="350" >
target=(1.,-0.5)            | target=(1.,0.)
<img src="https://github.com/jgpavez/systematics/blob/master/plots/xgboost/comp_train_mlp_multilikelihood_1.00_-0.50.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/xgboost/comp_train_mlp_multilikelihood_1.00_0.00.png" width="350" >
target=(2.,2.)              | target=(0.33,0.2) 
<img src="https://github.com/jgpavez/systematics/blob/master/plots/xgboost/comp_train_mlp_multilikelihood_2.00_2.00.png" width="350"> | <img src="https://github.com/jgpavez/systematics/blob/master/plots/xgboost/comp_train_mlp_multilikelihood_0.33_0.20.png" width="350"> 
target=(0.5,0.33)              | target=(0.33,0.14) 
<img src="https://github.com/jgpavez/systematics/blob/master/plots/xgboost/comp_train_mlp_multilikelihood_0.50_0.33.png" width="350"> | <img src="https://github.com/jgpavez/systematics/blob/master/plots/xgboost/comp_train_mlp_multilikelihood_0.33_0.14.png" width="350"> 
target=(0.25,0.17)              | target=(-0.5, -0.33)
<img src="https://github.com/jgpavez/systematics/blob/master/plots/xgboost/comp_train_mlp_multilikelihood_0.25_0.17.png" width="350"> |<img src="https://github.com/jgpavez/systematics/blob/master/plots/xgboost/comp_train_mlp_multilikelihood_-0.50_-0.33.png" width="350">
target=(-0.33,-0.20))              | target=(-0.33,-0.14))
<img src="https://github.com/jgpavez/systematics/blob/master/plots/xgboost/comp_train_mlp_multilikelihood_-0.33_-0.20.png" width="350"> |<img src="https://github.com/jgpavez/systematics/blob/master/plots/xgboost/comp_train_mlp_multilikelihood_-0.33_-0.14.png" width="350">




It can be seen that fit results are very good for most of the samples, for samples not well covered by the full set of bases a bias can be seen but is expected that this can be solved with more base samples.


