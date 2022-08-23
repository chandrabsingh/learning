#!/usr/bin/env python
# coding: utf-8

# # Lec 06-Naive Bayes - SVM
# 
# ## Outline
# - Naive Bayes
#   - Laplace smoothing
#   - Element models
# - Comments on applying ML
# - SVM

# ## Naive Bayes 
# - is a generative type of algorithm
# - To build generative model, P(x|y) and P(y) needs to be modeled. Gaussian discriminative model uses Gaussian and Bernoulli respectively to model this. Naive Bayes uses a different model. 
# > - $ x_{j} = \mathbb 1$ {indicator - word j appears in email or not}  
# > - $ P(x|y) = \prod\limits_{j=1}^{n}P(x_{j}|y)$ - NB uses the product of conditional probabilities of individual features given in the class label y   
# - Parameters of NB model are:    
#   > - $P(y=1) = \phi_{y} $ - the class prior for y=1 
#   > - $P(x_{j}=1|y=0) = \phi_{j|y=0}$ - chances of the word appearing in non-spam email.    
#   > - $P(x_{j}=1|y=1) = \phi_{j|y=1}$ - chances of the word appearing in spam email.  
# 

# ### Maximum likelihood estimate  
# > $\phi_{y} = \frac{\sum\limits_{i=1}^{m}\mathbb 1 \{y^{(i)}=1\}}{m}$  
# > $\phi_{j|y=0} = \frac{\sum\limits_{i=1}^{m}\mathbb 1 \{x_{j}^{(i)}=1, y^{(i)}=0\}}{\sum\limits_{i=1}^{m}\mathbb 1 \{y^{(i)}=0\}}$

# ### Laplace smoothing
# - ML conference papers - NIPS - Neural Information Processing Systems
# 
# - __NB breaks__ - because the probability of some event that you have not seen is trained as 0, which is statistically wrong
# - the classifier gives wrong result, when it gets such event for the first time
#   - instead using __Laplace smoothing__ helps this problem
#     - add 1 each to pass and fail scenario
#     
# #### Maximum likelihood estimate  
# > $\phi_{x=i} = \frac{\sum\limits_{j=1}^{m}\mathbb 1 \{x^{(i)}=j\} + 1} {m + k}$  
#   - k is the size of dictionary
# > $\phi_{i|y=0} = \frac{\sum\limits_{i=1}^{m}\mathbb 1 \{x_{j}^{(i)}=1, y^{(i)}=0\} + 1}{\sum\limits_{i=1}^{m}\mathbb 1 \{y^{(i)}=0\} + 2}$

# ## Event models for text classification
# - The functionality can be generalized from binary to multinomial features instead 
#   - Multivariate Bernoulli event
#   - Multinomial event
#   
# - We can discretize the size of house and transform a Bernoulli event to Multinomial event
#   - a rule of thumb is to discretize variables into 10 buckets 
# 
# 
# <img src="images/06_multinomialEventSqFt.png" width=400 height=400>  
# $\tiny{\text{YouTube-Stanford-CS229-Andrew Ng}}$  
# 

# ## NB variation
# - for text classification
#   - if the email contains text "Drugs! Buy drugs now", 
#     - the feature vector will make binary vector of all words appearing in the email
#     - it will lose the information that the word "drug" appeared twice
#     - each featture only stores information $x_{j} \in \{0,1\}$
#     - instead of making a feature vector of 10000 words of dictionary, we can make a feature vector of 4 words(as above) holding word index and $x_{j} \in \{1, 2, ... ,10000\}$
#       - algorithm containing feature vector of 10000 is called __"Multivariate Bernoulli algorithm"__
#       - algorithm containing feature vector of 4 is called __"Multinomial algorithm"__
#         - Andrew McCallum used these 2 names
# 
# 
# 
# ### NB advantage
# - quick to implement
# - computationally efficient
# - no need to implement gradient descent
# - easy to do a quick and dirty type of work
# - SVM or logistic regression does better work in classification problems
# - NB or GDA does not result in very good classification, but is very quick to implement, it is quick to train, it is non-iterative
# 

# ## Support Vector Machines (SVM)
# - find classification in the form of non linear decision boundaries
# - SVM does not have many parameters to fiddle with
#   - they have very robust packages
#   - and does not have parameters like learning rate, etc to fiddle with
#   
# <br>
# 
# - Let the non-linear feature variables be mapped in vector form as below. This non-linear function can be viewed as a linear function over the variables $\phi(x)$
# - derive an algorithm that can take input features of the $x_{1}, x_{2}$ and map them to much higher dimensional set of feature and then apply linear classifier, similar to logistic regression but different in details. This allows us to learn very non-linear decision boundaries.  
# 
# <img src="images/06_svmBoundary.png" width=200 height=200>  
# $\tiny{\text{YouTube-Stanford-CS229-Andrew Ng}}$  
# 
# 
# ### Outline for SVM
# - Optimal margin classifier (separable case)
# 
# - Kernel
#   - Kernels will allow us to fit in a large set of features
#   - How to take feature vector say $\mathbb R^{2}$ and map it to $\mathbb R^{5}$ or $\mathbb R^{10000}$ or $\mathbb R^{\inf}$, and train the algorithm to this higher feature. Map 2-dimensional feature space to infinite dimensional set of features. What it helps us is relieve us from lot of burden of manually picking up features. ($x^{2}, \space x^{3}, \space \sqrt{x}, \space x_{1}x_{2}^{2/3}, ...$)
# 
# - Inseparable case

# ### Optimal margin classifier (separable case)
# - Separable case means data is separable
# 
# #### Functional margin of classifier
# 
# - How confidently and accurately do you classify an example
# - Using binary classification and logistic regression
# - In logistic classifier:
# > $h_{\theta}(x) = g(\theta^{T}x)$
# - If you turn this into binary classification, if you have this algorithm predict not a probability but predict 0 or 1, 
# > - Predict "1", if $\theta^{T}x > 0 \implies h_{\theta}(x) = g(\theta^{T}x) \ge 0.5$)  
# > - Predict "0", otherwise
#   - If $y^{(i)} = 1$, hope that $\theta^{T}x^{(i)} \gg 0$
#   - If $y^{(i)} = 0$, hope that $\theta^{T}x^{(i)} \ll 0$
#     - this implies that the prediction is very correct and accurate 
# 
# #### Geometric margin
# - assuming data is linearly separable, 
# - and there are two lines that separates the true and false variables
# - the line that has a bigger separation or geometric margin meaning a physical separation from the training examples is a better choice
# 
# - We need to prove the optimal classifier is the algorithm that tries to maximize the geometric margin
# - what SVM and low dimensional classifier do is pose an optimization problem to try and find the line that classify these examples to find bigger separation margin

# #### Notation
# - Labels: $y \in \{-1, +1\} $ to denote class labels
# - SVM will generate hypothesis output value as {-1, +1} instead of probability as in logistic regression
# - g(z) = $\begin{equation}
#     \begin{cases}
#       +1 & \text{if z $\ge$ 0}\\
#       -1 & \text{otherwise}
#     \end{cases}       
# \end{equation}$
#   - instead of smooth transition, here we have a hard transition
# 
# 
# Previously in logistic regression:
# > $h_{\theta}(x) = g(\theta^{T}x)$
# - where x is $\mathbb R^{n+1} $ and $x_{0} = 1$
# 
# In SVM:
# > $h_{W,b}(x) = g(W^{T}x + b)$
# - where x is $\mathbb R^{n} $ and b is $\mathbb R $ and drop $x_{0}=1$
# - The term 
# 
# Other way to think about is:  
# > $
# \begin{bmatrix}
# \theta_{0}\\
# \theta_{1}\\
# \theta_{2}\\
# \theta_{3}
# \end{bmatrix}
# $
# - where $\theta_{0}$ corresponds to b and $\theta_{1}, \theta_{2}, \theta_{3}$ corresponds to W

# #### Functional margin of hyperplane (cont)
# - Functional margin __wrt single training example__
#   - Functional margin of hyperplane defined by $(w, b)$ wrt a single training example $(x^{(i)}, y^{(i)})$ is
#   > $\hat{\gamma}^{(i)} = y^{(i)} (w^{T}x^{(i)} + b)$
#     - If $y^{(i)} = +1$, hope that $(w^{T}x^{(i)} + b) \gg 0$
#     - If $y^{(i)} = -1$, hope that $(w^{T}x^{(i)} + b) \ll 0$
#   - Combining these two statements:
#     - we hope that $\hat{\gamma}^{(i)} \gg 0$
#   - If $\hat{\gamma}^{(i)} \gt 0$, 
#     - implies $h(x^{(i)}) = y^{(i)}$
#   - In logistic regression 
#     - $\gt 0$ means the prediction is atleast a little bit above 0.5 or little bit below 0.5
#     - $\gg 0$ means the prediction is either very close to 1 or very close to 0
# - Functional margin __wrt entire training set__ (how well are you doing on the worst example in your training set)  
#   > $\hat{\gamma} = \min\limits_{i} \hat{\gamma}^{(i)}$
#   - where i = 1,..,m - all the training examples
#   - here the assumption is training set is linearly separable
#     - we can assume this kind of worst-case notion because we are assuming that the boundary  is linearly separable
#   - we can normalize (w, b), which does not change the classification boundary, it simply rescales the parameters
#   > $(w, b) \rightarrow (\frac{w}{\Vert w \Vert}, \frac{b}{\Vert b \Vert})$
#   - classification remains the same, with any rescale number
#   

# #### Geometric margin of hyperplane (cont)
# - Geometric margin __wrt single training example__
#   - Upper half of hyperplane has positive cases and the lower half negative cases
#   - Let $(x^{(i)}, y^{(i)})$ be a training example which the classifier is classifying correctly, by predicting it as $h_{w,b}(x) = +1$. It predicts the lower half as $h_{w,b}(x) = -1$.
#   - Let the distance from the plane to the training example be the geometric margin
#   - Geometric margin of hyperplane (w,b) wrt $(x^{(i)}, y^{(i)})$ be
#   > $\hat{\gamma} = \frac{(y^{(i)})(w^{T}x^{(i)} + b)}{\Vert w \Vert}$
#     - This is the Euclidean distance between training example and decision boundary
#     - For positive examples, $y^{(i)}$ is +1 and the equation reduces to
#     > $\hat{\gamma} = \frac{(w^{T}x^{(i)} + b)}{\Vert w \Vert}$
#   
# 
# <img src="images/06_geometricMargin1.png" width=400 height=400>  
# $\tiny{\text{YouTube-Stanford-CS229-Andrew Ng}}$  
# 
# 

# #### Relationship between functional and geometric margin
# > $\gamma^{(i)} = \frac{\hat{\gamma}^{(i)}}{\Vert w \Vert}$  
# > i.e., $\text{Geometric Margin} = \frac{\text{Functional Margin}}{\text{Norm of w}}$
# 
# #### Geometric margin of hyperplane (cont)
# - Geometric margin __wrt entire training example__
#   > $\gamma = \min\limits_{i} \gamma^{(i)}$
#   
# #### Optimal margin classifier (cont)
# - what the optimal margin classifier does is to choose w,b to maximize $\gamma$
# > $\max_{\gamma, w, b} \text{s.t.} \frac{(y^{(i)})(w^{T}x^{(i)} + b)}{\Vert w \Vert} \ge \gamma$ for i=1,...,m
#   - subject to for every single training example  must have the geometric margin greater than or equal to gamma
#   - this causes to maximize the worst-case geometric margin
#   - this is not a convex optimization problem and cannot be solved using gradient descent or local optima
#   - but this can be re-written/reformulated into a equivalent problem which is minimizing norm of w subject to the geometric margin
# > $\min_{w,b} \Vert w \Vert ^{2}   \\
# \text{s.t. } y^{(i)}(w^{T}x^{(i)} + b) \ge 1$
#     - This is a convex optimization problem and can be solved using optimization packages
# - All this study is for linear separable case only
#   - this is a baby SVM
# - Once we learn kernels and apply kernels with optimal margin classifier, we get solution for SVM
