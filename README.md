## Decomposing tests between mixture models into their components

This work is based on the section **5.4 Decomposing tests between mixture models into their components** 
of the paper (ongoing work) [Approximating generalized likelihood ratio tests with calibrated discriminative classifiers]
(https://github.com/cranmer/parametrized-learning/blob/master/tex/parametrized-learning.pdf) by Kyle Cranmer.

We start with a simple model composed of three 1-Dim pdfs. Those pdfs are shown next

![decomposed model](https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.10/decomposed_model.png)

Those pdfs forms two mixture models, as shown in formula (21) of the paper of reference.
The first distribution correspond to the null signal hypothesis, meanwhile the second one 
correspond to the mixture model including the signal (in this case the signal corresponds to f0). 
Both distributions for coefficients **c0 = [ 0.,0.3,0.7]** and **c1 = [0.16666667,0.25,0.58333333]**
are shown in the next image.

![Full Model](https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.10/full_model.png)

First we compare the real ratios to the ratios obtained by training different classifiers (*Logistic, MLP and BDT*) and then composed using the formula (24) on the paper.

![All Classifiers](https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.10/comp_train_all_ratio.png)

The difference of each one of the composed ratios to the true ratio is shown in the next image.

![All Classifiers Differences](https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.10/comp_train_all_diff.png)

Given that *MLP* gives the best results, we will use this classifier for the next experiments.

Secondly, we compare each ratio obtained by training pair-wise to the real ratio of each pair of functions.

![Decomposed Ratios](https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.10/dec_comparison_mlp_ratio.png)

The ROC curves obtained by varying the threshold on the trained and true ratios of each pair of functions are shown in the next image.

![Decomposed ROC](https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.10/dec_comparison_mlp_roc.png)

In the next image the composed ratio using the formula (24) is compared to the ratio obtained by training the classifier 
in the full model and to the true ratio.

![All Ratios](https://github.com/jgpavez/systematics/blob/master/plots/full_comparison_mlp_ratio.png)

The difference of the composed ratio and the full trained ratio to the true ratio is shown in the image below.

![All Ratios Differences](https://github.com/jgpavez/systematics/blob/master/plots/full_comparison_mlp_diff.png)

Next, the Signal Efficiency - Background Rejection curves of each one of the ratios (composed, full trained and full truth) is shown.

![All ROC](https://github.com/jgpavez/systematics/blob/master/plots/full_comparison_mlp_sigbkg.png)

# Varying the signal presence 

We want to check how the composed and the full strategy are affected when the value of the 
signal coefficient become smaller.

In the next image the mixture models for coefficients for signal of **[0.2,0.1,0.05,0.01]** are 
shown

 0.20                   | 0.10
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.2/full_model.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.10/full_model.png" width="350" >
 0.05                   | 0.01
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.05/full_model.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.01/full_model.png" width="350" >

Now, we compare the real ratios, the composed ratios and the full trained ratios for each 
one of the cases

 0.20                   | 0.10
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.2/full_comparison_mlp_ratio.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.10/full_comparison_mlp_ratio.png" width="350" >
 0.05                   | 0.01
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.05/full_comparison_mlp_ratio.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.01/full_comparison_mlp_ratio.png" width="350" >

Finally, in the next image the Signal Efficiency - Background Rejection curves for the composed, full trained and true ratio are shown for each one of the cases

 0.20                   | 0.10
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.2/full_comparison_mlp_sigbkg.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.10/full_comparison_mlp_sigbkg.png" width="350" >
 0.05                   | 0.01
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.05/full_comparison_mlp_sigbkg.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.01/full_comparison_mlp_sigbkg.png" width="350" >

It can be seen that for very low signal presence the composed ratios are still working, on the other hand the full trained MLP is not able to reproduce correctly the ratio.

