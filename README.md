## Decomposing tests between mixture models into their components

This work is based on the section **5.4 Decomposing tests between mixture models into their components** 
of the paper (ongoing work) [Approximating generalized likelihood ratio tests with calibrated discriminative classifiers]
(https://github.com/cranmer/parametrized-learning/blob/master/tex/parametrized-learning.pdf)by Kyle Cranmer.

First we compare the ratios obtained by training different classifiers (*Logistic, Support Vector Classifier and Support Vector 
Regression*) and then composed using the formula (24) on the paper to the truth ratio.

![All Classifiers](https://github.com/jgpavez/systematics/blob/master/plots/composite_trained_all_ratio.png)

The difference of each one of the composed ratios to the truth ratio is shown in the next image.

![All Classifiers Differences](https://github.com/jgpavez/systematics/blob/master/plots/composite_trained_all_diff.png)

Given that *Logistic Regression* gives the best results, we will use this classifier for the next experiments.

Secondly, we compare each ratio obtained by training pair-wise to the real ratio of each pair of functions.

In the next image the composed ratio using the formula (24) is compared to the ratio obtained by training the classifier 
in the full model and to the truth ratio.

![All Ratios](https://github.com/jgpavez/systematics/blob/master/plots/full_comparison_logistic_ratio.png)

The difference of the composed ratio and the full trained ratio to the truth ratio is shown in the image below.

![All Ratios Differences](https://github.com/jgpavez/systematics/blob/master/plots/full_comparison_logistic_diff.png)

Next, the roc curves of each one of the ratios (composed, full trained and full truth) is shown.

![All ROC](https://github.com/jgpavez/systematics/blob/master/plots/full_comparison_logistic_ratio.png)



