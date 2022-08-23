#!/usr/bin/env python
# coding: utf-8

# # Lec 08-Data Splits - Models - Cross Validation
# 
# ## Outline
# - Bias/Variance
# - Regularization
# - Train/dev/test splits
# - Model selection and Cross validation

# ## Bias/Variance
# - underfit the data 
#   - high bias
#     - this data has strong bias that the data could be fit linearly
# - overfit the data
#   - high variance
#     - the prediction will have very high variance with slight modification in random draws of data
# 
# - Variance and bias gives an understanding as how to improve the algorithm
# 

# ## Regularization
# - this is used very often
# - the optimization objective for linear regression looks like:
# > $\min_{\theta}\sum\limits_{i=1}^{m}\Vert y^{(i)} - \theta^{T}x^{(i)} \Vert^{2} $  
# - to add regularization, we add an extra term
#   - by adding the regularization, we have added an incentive term for the algorithm to make the $\theta$ parameter smaller
# > $\min\limits_{\theta}\sum\limits_{i=1}^{m}\Vert y^{(i)} - \theta^{T}x^{(i)} \Vert^{2} + \frac{\lambda}{2} \Vert \theta \Vert^{2}$  
#   - if $\lambda$ is set to 0, we will be overfitting
#   - if $\lambda$ is set to a big number, then we will be forcing parameters to be too close to 0, we will be underfitting, with a very simple function
# - the optimization cost function for logistic regression looks like:
# > arg $\max\limits_{\theta}\sum\limits_{i=1}^{n}log\space p(y^{(i)}|x{(i)};\theta)$  
# - to add regularization, we add an extra term
# > arg $\max\limits_{\theta}\sum\limits_{i=1}^{n}log\space p(y^{(i)}|x^{(i)};\theta) - \frac{\lambda}{2} \Vert \theta \Vert^{2}$  

# ### Bayesian statistics and regularization
# 
# - Let S be the training set $S = \{(x^{(i)}, y^{(i)})\} _{i=1}^{m}$
# - Given a training set, we want to find the most likely value of $\theta$, by Bayes rule
# > $P(\theta|s) = \frac{P(s|\theta)P(\theta)}{p(s)}$  
# - To pick most likely value of $\theta$, given the data we saw
# > arg $\max_{\theta} P(\theta|s) = \text{arg }\max_{\theta} P(s|\theta)P(\theta)$  
#   - where the denominator is constant  
# - For logistic regression, the equation becomes  
# > arg $\max_{\theta} \left(\prod\limits_{i=1}^{m} P(y^{(i)} | x^{(i)}; \theta)\right) P(\theta) $
# - If you assume $P(\theta)$ is Gaussian $\theta \sim \mathbb N(0, \tau^{2}I)$, prior distribution of $\theta$  
# > $P(\theta) = \frac{1}{\sqrt{2\pi}(\tau^{2}I)^{1/2}} exp\left(-\frac{1}{2}\theta^{T}(\tau^{2}I)^{-1}\theta  \right)$  
# - The above is the prior distribution for $\theta$, and if we plug this in the estimate of $\theta$, take max and apply log, we will get the same regularization solution as above
# 
# - All of the above is based on frequentist interpretation
#   - Frequentist  
#     > arg $\max\limits_{\theta} P(S|\theta)$ - MLE - Maximum likelihood  
#   - Bayesian  
#     - based on prior distribution - after we have seen the data. 
#     - look at the data, compute the Bayesian posterior distribution of $\theta$ and pick a value of $\theta$ that's most likely  
#     > arg $\max\limits_{\theta} P(\theta|S)$ - MAP - Maximum a posteriori  

# ### Error vs Model complexity
# 
# - Assuming we dont consider regularization
# - plot a curve with model complexity on x-axis (with high degree polynomial on right side of curve) and training error on y-axis
# - we observe that training error improves or reduces with higher degree of complexity or more degree of polynomial
# - we also observed that the ability of algorithm to generalize goes down and then starts to go back up with increase in model complexity (generalization error)
#   - this curve is also true with regularization
#     - if $\lambda$(=infinite) is way too big, it will underfit
#     - if $\lambda$(=zero) is way too small, it will overfit
# - Let us try to find different procedures for finding this point in the middle
# 
# <img src="images/08_generalizationError.png" width=400 height=400>  
# $\tiny{\text{YouTube-Stanford-CS229-Andrew Ng}}$  
# 
# 

# ## Train/Dev/Test datasets
# - Given a dataset
# - we split data into train/dev/test sets
# - say we have 10000 examples
# - we are trying to find what is the polynomial we are trying to fit, or we are trying to choose $\lambda$, or we are trying to choose $\tau$ band-width parameter in locally weighted regression, or we are trying to choose value $C$ in SVM
#   - in all these problems, we have question of bias/variance trade-offs
# - Split dataset $S$ into $S_{train}$, $S_{dev}$, $S_{test}$
#   - Train each $model_{i}$ (option for different degree of polynomial) on "$S_{train}$"
#   - Get some hypothesis $h_{i}$
#   - Measure the error on "$S_{dev}$"
#   - Pick the one with lowest error on "$S_{dev}$"
#     - If you measure error on "$S_{train}$", we will end up choosing a complex polynomial to fit
#   - To publish a paper or report unbiased report, evaluate your algorithm on a separate "$S_{test}$" set

# ### Cross-validation
# 
# - Holdout cross validation set
# <br>
# 
# - Optimize performance on the dev set
# - Then to know how well is the algorithm performing, then evaluate the model on the test set
# - Be careful not to do is - Dont make decision based on the test set
#   - Because then your scientific data to the test set is no longer an unbiased estimate
# 
# <img src="images/08_crossValidationError.png" width=400 height=400>  
# $\tiny{\text{YouTube-Stanford-CS229-Andrew Ng}}$  
# 
# 

# ### k-fold Cross Validation
# - Small datasets - You have 100 examples
# - It is a waste of data if you apply 70-30 rule?
#   - $S_{train}$ = 70, $S_{dev}$ = 30
# - Procedure if you have a small dataset
#     - Say k=5, divide 100 examples into 5 subsets, 20 examples in each subset
#       - For d = 1,...,5 (for each degree of polynomial say 5)
#         - For i=1,..,k
#           - Train (fit parameters) on k-1 pieces
#           - Test on the remaining 1 piece
#         - Average
#       - pick the degree of polynomial that did best among all the runs
#       - we have now 5 classifiers
#       - say if we choose 2nd order polynomial
#       - refit the model once on all 100% of the data
# - Typically k=10 is used
# 
# - Even smaller 
#   - Leave-one-out CV

# ## Feature Selection
# - If you suspect that out of 10000 features only 50 are highly relevant
# - Preventive Maintenance for truck - say there are 10000 reasons why the truck may go down, but only 10 of them might be most relevant
#   - in such cases, feature selection might be the thing to go for
# - Many a times, one way to reduce overfitting is to try to find a small subset of features that are most useful for the task
#   - this takes judgement
#   - this cannot be applied to computer vision, as subset of pixels might be relevant
#   - but this can be applied to other type of problems, where looking at small subset of relevant features help in reducing overfitting
#   
# <br>
# 
# - Feature selection is a special case of model selection
# <br>
# 
# - Algorithm
#   - Start with empty set of feature F = $\phi$
#   - Repeat 
#     - Try adding each feature i to F and see which single feature addition most improves the dev set performance
#     - Go ahead and add that feature to F
#     
# - This can be computationally expensive
# - Another such method is backward search
#   - in this we start will all the features and remove one feature at a time
