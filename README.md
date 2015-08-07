## Decomposing tests between mixture models into their components 
## Multidimensional case

This work is based on the section **5.4 Decomposing tests between mixture models into their components** 
of the paper [Approximating Likelihood Ratios with Calibrated Discriminative Classifiers]
(http://arxiv.org/abs/1506.02169) by Kyle Cranmer.

We will check the composition method in a 2-dim mixture model. The pdfs conforming the mixture model 
are shown next

![decomposed model](https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/0.10/decomposed_model.png)

Those pdfs forms two mixture models, as shown in formula (21) of the paper of reference.
The first distribution correspond to the null signal hypothesis, meanwhile the second one 
correspond to the mixture model including the signal (in this case the signal corresponds to f0). 
Both distributions for coefficients **c0 = [ 0.,0.3,0.7]** and **c1 = [0.09090909,0.27272727,0.63636364]**
are shown in the next image.

![decomposed model](https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/0.10/full_model.png)

First, the ROC curves obtained by varying the threshold on the trained and true ratios on each pair of 
functions are shown in the next image.

![Decomposed ROC](https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/0.05/dec_comparison_mlp_roc.png)

Next, the Signal Efficiency - Background Rejection curves of each one of the ratios (composed, full trained and full truth) is shown.

![All ROC](https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/0.10/full_comparison_mlp_sigbkg.png)

# Varying the signal presence 

We want to check how the composed and the full strategy are affected when the value of the 
signal coefficient become smaller.

In the next image the mixture models for coefficients for signal of **[0.1,0.05,0.01,0.005]** are 
shown

 0.10                   | 0.05
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/0.10/full_model.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/0.05/full_model.png" width="350" >
 0.01                   | 0.005
<img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/0.01/full_model.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/0.005/full_model.png" width="350" >

Next, the score histogram for each one of the pair-wise trained classifiers for signal 
and background is shown, notice that only histograms for k < j is shown

 0.10                   | 0.05
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/dec0_all_mlp_hist.pdf" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/0.05/decomp_all_mlp_hist.png" width="350" >
 0.01                   | 0.005
<img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/0.01/decomp_all_mlp_hist.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/0.005/decomp_all_mlp_hist.png" width="350" >

The ratio histograms for the composite, full trained and true cases is shown in the next image, those histograms are constructed over data sampled from the distribution of F0 background and f0 signal.

 0.10                   | 0.05
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/0.10/ratio_comparison_mlp_hist.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/0.05/ratio_comparison_mlp_hist.png" width="350" >
 0.01                   | 0.005
<img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/0.01/ratio_comparison_mlp_hist.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/0.005/ratio_comparison_mlp_hist.png" width="350" >


Finally, in the next image the Signal Efficiency - Background Rejection curves for the composed, full trained and true ratio are shown for each one of the cases

 0.10                   | 0.05
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/0.10/full_comparison_mlp_sigbkg.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/0.05/full_comparison_mlp_sigbkg.png" width="350" >
 0.01                   | 0.005
<img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/0.01/full_comparison_mlp_sigbkg.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/0.005/full_comparison_mlp_sigbkg.png" width="350" >

It can be seen that for this 2-dim case and very low signal presence the composed ratios are still working perfectly, on the other hand the full trained MLP is not able to 
reproduce the ratio at all.

#N-dimensions

Now, we will check the composition method in a N-dim mixture model. The pdfs conforming the mixture model 
are shown next

![decomposed model](https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/10dim/0.10/decomposed_model.png)

Those pdfs forms two mixture models, as shown in formula (21) of the paper of reference.
The first distribution correspond to the null signal hypothesis, meanwhile the second one 
correspond to the mixture model including the signal (in this case the signal corresponds to f0). 
Both distributions for coefficients **c0 = [ 0.,0.3,0.7]** and **c1 = [0.09090909,0.27272727,0.63636364]**
are shown in the next image.

![decomposed model](https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/10dim/0.10/full_model.png)

The following images show scatter plots between a subset of features for the pairwise combinations 1-2, 1-3 and 2-3
the number of the feature is indicated in the columns and rows of the grid of plots.

# f1-f2
![scatter grid 1](https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/10dim/dec_truth_0_1_grid.png)

#f1-f3
![scatter grid 2](https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/10dim/dec_truth_0_2_grid.png)

#f2-f3
![scatter grid 3](https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/10dim/dec_truth_1_2_grid.png)

Next, a parallel coordinates plot of the distributions F0 background - f0 signal is shown

![parallel](https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/paralell_coordinates_F0-f0.png)

First, the ROC curves obtained by varying the threshold on the trained and true ratios on each pair of 
functions are shown in the next image.

![Decomposed ROC](https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/10dim/0.05/dec_comparison_mlp_roc.png)

Next, the Signal Efficiency - Background Rejection curves of each one of the ratios (composed, full trained and full truth) is shown.

![All ROC](https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/10dim/0.10/full_comparison_mlp_sigbkg.png)

# Varying the signal presence 

We want to check how the composed and the full strategy are affected when the value of the 
signal coefficient become smaller.

In the next image the mixture models for coefficients for signal of **[0.1,0.05,0.01,0.005]** are 
shown

 0.10                   | 0.05
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/10dim/0.10/full_model.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/10dim/0.05/full_model.png" width="350" >
 0.01                   | 0.005
<img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/10dim/0.01/full_model.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/10dim/0.005/full_model.png" width="350" >

Next, the score histogram for each one of the pair-wise trained classifiers for signal 
and background is shown, notice that only histograms for k < j is shown

 0.10                   | 0.05
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/10dim/0.10/decomp_all_mlp_hist.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/10dim/0.05/decomp_all_mlp_hist.png" width="350" >
 0.01                   | 0.005
<img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/10dim/0.01/decomp_all_mlp_hist.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/10dim/0.005/decomp_all_mlp_hist.png" width="350" >

The ratio histograms for the composite, full trained and true cases is shown in the next image, those histograms are constructed over data sampled from the distribution of F0 background and f0 signal.

 0.10                   | 0.05
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/10dim/0.10/ratio_comparison_mlp_hist.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/10dim/0.05/ratio_comparison_mlp_hist.png" width="350" >
 0.01                   | 0.005
<img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/10dim/0.01/ratio_comparison_mlp_hist.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/10dim/0.005/ratio_comparison_mlp_hist.png" width="350" >


Finally, in the next image the Signal Efficiency - Background Rejection curves for the composed, full trained and true ratio are shown for each one of the cases

 0.10                   | 0.05
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/10dim/0.10/full_comparison_mlp_sigbkg.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/10dim/0.05/full_comparison_mlp_sigbkg.png" width="350" >
 0.01                   | 0.005
<img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/10dim/0.01/full_comparison_mlp_sigbkg.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/10dim/0.005/full_comparison_mlp_sigbkg.png" width="350" >

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
<img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/0.05/100/comp_train_mlp_likelihood.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/0.05/500/comp_train_mlp_likelihood.png" width="350" >
 1000                   | 5000
<img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/0.05/1000/comp_train_mlp_likelihood.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/0.05/5000/comp_train_mlp_likelihood.png" width="350" >


Now, we will check if it is possible to identify both, the signal coefficient **c1[0]** and the 
background coefficient **c1[1]** (using c1[2] = 1.-c1[0]-c1[1]). In this case we are using 100,500,1000 and 5000 samples to compute the likelihood and values for the coefficients of c1[0] = 0.05 and c1[1] = 0.285. Truth likelihoods are compared to trained Likelihoods.

 100                  | 500 
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/0.05/100/comp_train_mlp_multilikelihood.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/0.05/500/comp_train_mlp_multilikelihood.png" width="350" >
 1000                   | 5000 
<img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/0.05/1000/comp_train_mlp_multilikelihood.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/0.05/5000/comp_train_mlp_multilikelihood.png" width="350" >

It can be seen that again the method do a very good job on identifying the correct values for the coefficients.

Next we will check if the estimated **c1[0]** and **c1[1]** are unbiased estimators of the 
real values. In the next images histograms of the estimated values for the likelihood 
obtained by the true and composed methods and values of **c1[0] = 0.05** and **c1[1] = 0.285** 
 are shown. 

 c1                  | c1
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/0.05/c1c2_train_mlp_c1_hist.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/0.005/c1c2_train_mlp_c1_hist.png" width="350" >
 c2                   | c2
<img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/0.05/c1c2_train_mlp_c2_hist.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/0.005/5000/c1c2_train_mlp_c2_hist.png" width="350" >

It can be seen that both methods obtain unbiased estimators of the true values.



#N-dimensions Private Model

We will check the composition method in a N-dim mixture model. But this time each pdf is also a sum of gaussians with fixed coefficients. 
We want to see how the method works in this harder case.

![decomposed model](https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/private/0.10/decomposed_model.png)

Both distributions for coefficients **c0 = [ 0.,0.3,0.7]** and **c1 = [0.10,0.27,0.63**
are shown in the next image.

![decomposed model](https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/private/0.10/full_model.png)

The following images show scatter plots between a subset of features for the pairwise combinations 1-2, 1-3 and 2-3
the number of the feature is indicated in the columns and rows of the grid of plots.

# f1-f2
![scatter grid 1](https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/private/dec_truth_0_1_grid.png)

#f1-f3
![scatter grid 2](https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/private/dec_truth_0_2_grid.png)

#f2-f3
![scatter grid 3](https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/private/dec_truth_1_2_grid.png)

Next, a parallel coordinates plot of the distributions F0 background - f0 signal is shown

![parallel](https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/private/paralell_coordinates_F0-f0.png)

Now we will to check how the composed and the full strategy are affected when the value of the 
signal coefficient become smaller for this *private* model.

In the next image the mixture models for coefficients for signal of **[0.1,0.05,0.01,0.005]** are 
shown

 0.10                   | 0.05
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/private/0.10/full_model.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/private/0.05/full_model.png" width="350" >
 0.01                   | 0.005
<img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/private/0.01/full_model.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/private/0.005/full_model.png" width="350" >

Next, the score histogram for each one of the pair-wise trained classifiers for signal 
and background is shown, notice that only histograms for k < j is shown

 0.10                   | 0.05
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/private/0.10/decomp_all_mlp_hist.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/private/0.05/decomp_all_mlp_hist.png" width="350" >
 0.01                   | 0.005
<img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/private/0.01/decomp_all_mlp_hist.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/private/0.005/decomp_all_mlp_hist.png" width="350" >

The ratio histograms for the composite, full trained and true cases is shown in the next image, those histograms are constructed over data sampled from the distribution of F0 background and f0 signal.

 0.10                   | 0.05
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/private/0.10/ratio_comparison_mlp_hist.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/private/0.05/ratio_comparison_mlp_hist.png" width="350" >
 0.01                   | 0.005
<img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/private/0.01/ratio_comparison_mlp_hist.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/private/0.005/ratio_comparison_mlp_hist.png" width="350" >


Finally, in the next image the Signal Efficiency - Background Rejection curves for the composed, full trained and true ratio are shown for each one of the cases

 0.10                   | 0.05
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/private/0.10/full_comparison_mlp_sigbkg.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/private/0.05/full_comparison_mlp_sigbkg.png" width="350" >
 0.01                   | 0.005
<img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/private/0.01/full_comparison_mlp_sigbkg.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/private/0.005/full_comparison_mlp_sigbkg.png" width="350" >

It can be seen that even with this harder model where each single distribution is a sum of gaussians the composed ratios are working quite well even for small signal 
coefficients.


## Identifying the signal coefficient by fitting for the private model

We will check if even for this harder case it is possible to identify the signal and background 
coefficient by minimizing the likelihood. The method is the same that the one used in the previous 
section
First, We will check how the likelihood are affected by the number of data generated to compute the likelihoods, for the case c1[0] = 0.05.

 100                   | 500
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/private/100/comp_train_mlp_likelihood.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/private/500/comp_train_mlp_likelihood.png" width="350" >
 1000                   | 5000
<img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/private/1000/comp_train_mlp_likelihood.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/private/5000/comp_train_mlp_likelihood.png" width="350" >


Now, we will check if it is possible to identify both, the signal coefficient **c1[0]** and the 
background coefficient **c1[1]** (using c1[2] = 1.-c1[0]-c1[1]). In this case we are using 100,500,1000 and 5000 samples to compute the likelihood and values for the coefficients of c1[0] = 0.05 and c1[1] = 0.285. Truth likelihoods are compared to trained Likelihoods.

 100                  | 500 
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/private/100/comp_train_mlp_multilikelihood.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/private/500/comp_train_mlp_multilikelihood.png" width="350" >
 1000                   | 5000 
<img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/private/1000/comp_train_mlp_multilikelihood.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/private/5000/comp_train_mlp_multilikelihood.png" width="350" >

Again, the method works pretty good even with this harder model.

# How training affect ratios

Now, what we will study is how the quality of training affect the final ratios. We will keep a large amount of data to construct the likelihood ratios (30000 samples) but we will vary 
the amount of data used to train the classiffier. We will use 0, 1000, 10000, and 1000000 samples to train the classifier.
The Likelihood ratios histograms are shown in the next image for each one of the cases and for a signal coefficient of **0.05**.

 0                   | 1000
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/training/harder/0/ratio_comparison_mlp_hist.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/training/harder/1000/ratio_comparison_mlp_hist.png" width="350" >
 10000                   | 100000
<img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/training/harder/10000/ratio_comparison_mlp_hist.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/training/harder/100000/ratio_comparison_mlp_hist.png" width="350" >


The signal efficiency - background rejection curves are shown next for each one of the cases.


 0                   | 1000
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/training/harder/0/full_comparison_mlp_sigbkg.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/training/harder/1000/full_comparison_mlp_sigbkg.png" width="350" >
 10000                   | 100000
<img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/training/harder/10000/full_comparison_mlp_sigbkg.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/training/harder/100000/full_comparison_mlp_sigbkg.png" width="350" >

Finally, the histograms for the values of the coefficients c1[0] and c2[0] obtained by maximizing the log-likelihood on a dataset in each one of the cases are shown. 


c1[0] - 0                   | c1[0] - 1000
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/training/harder/0/c1c2_train_mlp_c1_hist.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/training/harder/1000/c1c2_train_mlp_c1_hist.png" width="350" >
c1[0] - 10000                | c1[0] - 100000
<img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/training/harder/10000/c1c2_train_mlp_c1_hist.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/training/harder/100000/c1c2_train_mlp_c1_hist.png" width="350" >

c1[1] - 0                   | c1[1] - 1000
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/training/harder/0/c1c2_train_mlp_c2_hist.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/training/harder/1000/c1c2_train_mlp_c2_hist.png" width="350" >
c1[1] - 10000                | c1[1] - 100000
<img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/training/harder/10000/c1c2_train_mlp_c2_hist.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/multidim/plots/mlp/training/harder/100000/c1c2_train_mlp_c2_hist.png" width="350" >

With a relatively small amount of samples, the trained likelihood is able to approximate closely the true value of the coefficients **c1[0]** and **c1[1]** with only a small bias. 
With more data the approximation is very good.

