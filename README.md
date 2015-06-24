## Decomposing tests between mixture models into their components

This work is based on the section **5.4 Decomposing tests between mixture models into their components** 
of the paper [Approximating Likelihood Ratios with Calibrated Discriminative Classifiers]
(http://arxiv.org/abs/1506.02169) by Kyle Cranmer.

We start with a simple model composed of three 1-Dim pdfs. Those pdfs are shown next

![decomposed model](https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.10/decomposed_model.png)

Those pdfs forms two mixture models, as shown in formula (21) of the paper of reference.
The first distribution correspond to the null signal hypothesis, meanwhile the second one 
correspond to the mixture model including the signal (in this case the signal corresponds to f0). 
Both distributions for coefficients **c0 = [ 0.,0.3,0.7]** and **c1 = [0.09090909,0.27272727,0.63636364]**
are shown in the next image.

![Full Model](https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.10/full_model.png)

First we compare the real ratios to the ratios obtained by training different classifiers (*Logistic, MLP and BDT*) and then composed using the formula (24) on the paper.

![All Classifiers](https://github.com/jgpavez/systematics/blob/master/plots/comp_train_all_ratio.png)

Given that *MLP* gives the best results, we will use this classifier for the next experiments.

Secondly, we compare each ratio obtained by training pair-wise to the real ratio of each pair of functions.

![Decomposed Ratios](https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.10/dec_comparison_mlp_ratio.png)

The ROC curves obtained by varying the threshold on the trained and true ratios of each pair of functions are shown in the next image.

![Decomposed ROC](https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.10/dec_comparison_mlp_roc.png)

In the next image the composed ratio using the formula (24) is compared to the ratio obtained by training the classifier 
in the full model and to the true ratio.

![All Ratios](https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.10/full_comparison_mlp_ratio.png)

Next, the Signal Efficiency - Background Rejection curves of each one of the ratios (composed, full trained and full truth) is shown.

![All ROC](https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.10/full_comparison_mlp_sigbkg.png)

# Varying the signal presence 

We want to check how the composed and the full strategy are affected when the value of the 
signal coefficient become smaller.

In the next image the mixture models for coefficients for signal of **[0.1,0.05,0.01,0.005]** are 
shown

 0.10                   | 0.05
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.10/full_model.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.05/full_model.png" width="350" >
 0.01                   | 0.005
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.01/full_model.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.005/full_model.png" width="350" >

Now, we compare the real ratios, the composed ratios and the full trained ratios for each 
one of the cases

 0.10                   | 0.05
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.10/full_comparison_mlp_ratio.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.05/full_comparison_mlp_ratio.png" width="350" >
 0.01                   | 0.005
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.01/full_comparison_mlp_ratio.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.005/full_comparison_mlp_ratio.png" width="350" >

Next, the score histogram for each one of the pair-wise trained classifiers for signal 
and background is shown, notice that only histograms for k < j is shown

 0.10                   | 0.05
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.10/decomp_all_mlp_hist.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.05/decomp_all_mlp_hist.png" width="350" >
 0.01                   | 0.005
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.01/decomp_all_mlp_hist.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.005/decomp_all_mlp_hist.png" width="350" >

The ratio histograms for the composite, full trained and true cases are shown in the next image, those histograms are constructed over data sampled from the distribution of F0 background and f0 signal.

 0.10                   | 0.05
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.10/ratio_comparison_mlp_hist.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.05/ratio_comparison_mlp_hist.png" width="350" >
 0.01                   | 0.005
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.01/ratio_comparison_mlp_hist.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.005/ratio_comparison_mlp_hist.png" width="350" >


Finally, in the next image the Signal Efficiency - Background Rejection curves for the composed, full trained and true ratio are shown for each one of the cases

 0.10                   | 0.05
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.10/full_comparison_mlp_sigbkg.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.05/full_comparison_mlp_sigbkg.png" width="350" >
 0.01                   | 0.005
<img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.01/full_comparison_mlp_sigbkg.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/master/plots/mlp/0.005/full_comparison_mlp_sigbkg.png" width="350" >

It can be seen that for very low signal presence the composed ratios are still working perfectly, on the other hand the full trained MLP is not able to reproduce the ratio at all.

# Checking how the training affect the ratios

Four major points will be studied
 * Using more training data to train the MLP.
 * Keep the same MLP, but changing the ammount of data used to create the histograms of score for each f_i.
 * Reuse the classifier trained for fi - fj in fj - fi.
 * Sample only once from each fi (other way woule be to obtain samples for each fi in each pair including fi).

First, we study how the ratios are affected by the number of samples used to train the MLP. We train the MLP with 1000, 10000, 100000 and 200000 samples.
The next image shows the results.

 1000                  | 10000
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/training_study/plots/mlp/1000/full_comparison_mlp_ratio.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/training_study/plots/mlp/10000/full_comparison_mlp_ratio.png" width="350" >
 100000                   | 200000
<img src="https://github.com/jgpavez/systematics/blob/training_study/plots/mlp/100000/full_comparison_mlp_ratio.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/training_study/plots/mlp/200000/full_comparison_mlp_ratio.png" width="350" >

Secondly, we keep the MLP fixed (using the one trained with 200000 samples) and change the number of samples used to create the score histograms for each one of the distributions.
Results for 1000 20000 10000 and 1000000 samples are shown next.

 1000                  | 2000
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/training_study/plots/mlp/1000_h/full_comparison_mlp_ratio.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/training_study/plots/mlp/2000_h/full_comparison_mlp_ratio.png" width="350" >
 10000                   | 100000
<img src="https://github.com/jgpavez/systematics/blob/training_study/plots/mlp/10000_h/full_comparison_mlp_ratio.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/training_study/plots/mlp/100000_h/full_comparison_mlp_ratio.png" width="350" >

Now, we are interested in check how the ratio are affected when not reusing the classifiers trained for i-j in the dataset j-i. In the previous cases we reuse the output of the classifier 
trained for i-j in the dataset j-i but with the complement of the output. Results using 10000 and 200000 samples for training and building histograms without reusing the classifiers 
are shown next.

 Not reusing classifier - 10000        | Not reusing classifier - 200000
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/training_study/plots/mlp/10000_nr/full_comparison_mlp_ratio.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/training_study/plots/mlp/200000_nr/full_comparison_mlp_ratio.png" width="350" >
 Reusing classifier - 10000        | Reusing classifier - 200000
<img src="https://github.com/jgpavez/systematics/blob/training_study/plots/mlp/10000/full_comparison_mlp_ratio.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/training_study/plots/mlp/200000/full_comparison_mlp_ratio.png" width="350" >


Finally, we will check how the results are effected if we use different samples from f_i for each pair. Commonly we would sample only one dataset from each f_i to construct each combined dataset, if we use different samples for each one of the combinations results are affected, as shown next.

 Not reusing classifier - 10000        | Not reusing classifier - 200000
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/systematics/blob/training_study/plots/mlp/10000_nr_nf/full_comparison_mlp_ratio.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/training_study/plots/mlp/200000_nr_nf/full_comparison_mlp_ratio.png" width="350" >
 Reusing classifier - 10000        | Reusing classifier - 200000
<img src="https://github.com/jgpavez/systematics/blob/training_study/plots/mlp/10000/full_comparison_mlp_ratio.png" width="350">  | <img src="https://github.com/jgpavez/systematics/blob/training_study/plots/mlp/200000/full_comparison_mlp_ratio.png" width="350" >

