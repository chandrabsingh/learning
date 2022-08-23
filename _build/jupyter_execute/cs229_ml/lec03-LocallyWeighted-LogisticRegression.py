#!/usr/bin/env python
# coding: utf-8

# # Lec 03-Locally Weighted Regression - Logistic Regression
# 
# ## Outline 
# - Linear Regression (recap)  
# - Locally Weighted regression  
# - Probabilistic interpretation  
# - Logistic Regression  
# - Newton's method  

# ## Notation  
# * $ (x^{(i)}, y^{(i)}) - i^{th} $ example  
# * $ x^{(i)} \in \mathbb R^{n+1}, y^{(i)} \in \mathbb R, x_{0} = 1 $  
# * m - # examples, n = #features  
# * $ h_{\theta}(x) = \sum\limits_{j=0}^{n} \theta_{j}x_{j} = \theta^{T}x $  
# * Cost function $ J(\theta) = \frac{1}{2} \sum\limits_{i=1}^m (h_{\theta}(x^{(i)}) - y^{(i)})^{2} $  
# 

# ## Feature selection algorithm 
# - what sort of feature do you want to fit  
#     - $\theta_{0} + \theta_{1}x_{1}$ 
#     - $\theta_{0} + \theta_{1}x_{1} + \theta_{2}x^{2}$  
#         * quadratic equation, it comes back to the axis
#     - $\theta_{0} + \theta_{1}x + \theta_{2}\sqrt x $ 
#         * if we dont want quadratic function to curve back
#     - $\theta_{0} + \theta_{1}x + \theta_{2}\sqrt x + \theta_{3} log(x)$ 
# - Once we have defined how we want to fit, we can apply the machinery of linear regression here
#     - Later, we will learn, what sort of feature ($x, \sqrt x, x^{2}$) is best to fit
# 
# 
# Different ways of adjusting this problem when regression does not fit in a single line, we try to solve it using locally weighted linear regression  
# 
# 

# ## Locally weighted regression   
# - "Parameteric" learning algorithm -   
#   - fit fixed set of parameters ($\theta_{i}$) to data  
#   - For example: Linear Regression
# - "Non-Parameteric" learning algorithm    
#   - Amount of data/parameter, you need to keep grows (linearly) with size of training set data
#   - to make predictions, we need to save lots of data
#   - For example: Locally weighted regression

# To evaluate h at certain value of input X:
# - For linear regression, 
# - fit $\theta$ to minimize the cost function, 
# > $\frac{1}{2}\sum\limits_{i}(y^{(i)}-\theta^{T}x^{(i)})^{2}$
# - and return $\theta^{T}X$
# - For locally weighted regression, __more weights are put on some local areas__, rather than focusing globally
# > $\sum\limits_{i=1}^{m}w^{(i)}(y^{(i)}-\theta^{T}x^{(i)})^{2}$ - __Eq 1__
# - where $w^{(i)}$ is a weighting function
# > $w^{(i)} = exp(\frac{-(x^{(i)} - x)^{2}}{2\tau^{2}})$
# - x is the location where we want to make prediction
# - $x^{i}$ is the ith training example
# - If $|x^{(i)} - x|$ is small, $ x^{(i)} \approx 1$
# - If $|x^{(i)} - x|$ is large, $ x^{(i)} \approx 0$
# - The equation 1 is similar to what we saw in linear regression LMS. The difference is $w^{(i)}$. 
#     - If the example $x^{(i)}$ is far from the prediction is to be made, the prediction is multiplied with error term 0
#     - If the example $x^{(i)}$ is close from the prediction is to be made, the prediction is multiplied with error term 1
# - If bandwidth $\tau$ is too large, the fitting will be over-smoothing. If bandwidth is too small, the fitting will be too jagged. This hyperparameter allows you to say, how many parameters would you choose to make a prediction
# - _depending on which point, you want to make a prediction, we focus on that locality_

# <img src="images/03_locallyWt.png" width=400 height=400 />   $\tiny{\text{YouTube-Stanford-CS229-Andrew Ng}}$
# 
# <img src="images/03_locallyWt2.png" width=400 height=200 />   $\tiny{\text{YouTube-Stanford-CS229-Andrew Ng}}$

# ## Logistic Regression
# 
# ### Probabilistic interpretation of Linear Regression
# 
# - Why least squares has squared errors?
#   - Assume that true price of every house $y^{(i)} = \theta^{T}x^{(i)} + \epsilon^{(i)}$, where first term is the true price and the second error term $\epsilon^{(i)}$ is due to unmodelled effects and random noise
#   -  where $\epsilon^{(i)} \sim N(0, \sigma^{2})$ - is distributed Gaussian with mean 0 and variance $\sigma^{2}$
#   - $ P(\epsilon^{(i)}) = \frac{1}{\sqrt (2\pi)\sigma}exp(- \frac{(\epsilon^{(i)})^{2}}{2\sigma^{2}} )$, which integrates to 1
#   - Assumption is $\epsilon^{(i)}$ is IID, error term of one house is not affected by price of other house in the next lane. It is not absolutely true, but is good enough
#   - this implies that $P(y^{(i)}| x^{(i)}; \theta) = \frac{1}{\sqrt (2\pi)\sigma} exp(- \frac{(y^{(i)} - \theta^{T}x^{(i)})^{2}}{2\sigma^{2}})$ 
#     - ";" means "parameterized by". Instead if ", $\theta$" would mean - "conditioned by $\theta$, which will be a random variable", which is not the case here
#     > i.e., $ (y^{(i)}| x^{(i)}; \theta) \sim N(\theta^{T}x^{(i)}, \sigma^{2})$

# #### Likelihood of parameter $\theta$
# - $L(\theta) = P(y^{(i)}| x^{(i)}; \theta) $
# 
# #### Difference between likelihood and probability?
# - likelihood of parameters - make the data as a fixed thing and vary parameters
# - probability of data - make the parameter as a fixed thing and vary the data

# #### Log likelihood
# > $l(\theta) = \text{log } L(\theta)\\
# = \text{log} \prod\limits_{i=1}^{m} \frac{1}{\sqrt (2\pi)\sigma} \exp(- \frac{(y^{(i)} - \theta^{T}x^{(i)})^{2}}{2\sigma^{2}})\\
# = \sum()
# $
# 
# <img src="images/03_logLikelihoodDerivation.png" width=600 height=400 />   
# $\tiny{\text{YouTube-Stanford-CS229-Andrew Ng}}$

# ### Classification
# - Here we used a setup of probabilistic assumptions, with the key assumption that Gaussian errors are iid, which turns out to be exactly like least squares algorithm
# - Use this framework and apply it to a different type of problem - classification problem
#   - make assumption about $P(y|x; \theta)$
#   - find max likelihood estimation
# - in classification, the value of __y is either 0 or 1__
# 
# #### Binary classification
# - $y \in \{0,1\}$
# - Why linear regression should not be used for binary classification?
#   - dont use linear regression to binary classification
#   - it will be a very bad fit especially when there are outliers
#   - the decision boundary will be very different
# 
# <img src="images/03_binaryClassificationRegression.png" width=400 height=200 />   $\tiny{\text{YouTube-Stanford-CS229-Andrew Ng}}$  

# #### Logistic Regression
# - Goal: __hypothesis function to output values between 0 and 1__
# > $$h_{\theta}(x) \in [0,1]$$
# - choose following form of hypothesis
# > $$h_{\theta}(x) = g(\theta^{T}x) = \frac{1}{1+e^{-\theta^{T}x}} $$
# - where g(z) is logistic function or sigmoid function
# > $$g(z) = \frac{1}{1+e^{-z}}$$
# - In Linear Regression, the hypothesis was
# > $$h_{\theta}(x) = \theta^{T}x $$
# - so in logistic regression, the hypothesis function uses this sigmoid function that force generate output between 0 and 1
# <br>
# 
# - Why did we specifically choose sigmoid function?
#   - there is a broader class of algorithms called generalized linear models(GLM) of which this is a special case

# #### How do we fit $\theta$
# - we fit the parameters using maximum likelihood
# - Let us assume
# > $P(y=1|x;\theta) = h_{\theta}(x)$
# > $P(y=0|x;\theta) = 1 - h_{\theta}(x)$
# - generalizing above, this can be written using
# > $y \in \{0,1\}$
# - as
# > $$P(y|x;\theta) = h(x)^{y}(1-h(x))^{(1-y)}$$
# - above equation is a form of if..else 
# 
# #### Log likelihood 
# 
# <img src="images/03_logisticLikelihoodEst.png" width=600 height=400 />   
# $\tiny{\text{YouTube-Stanford-CS229-Andrew Ng}}$  
# 
# 
# <img src="images/03_logisticLogLikelihoodEst.png" width=600 height=400 />   
# $\tiny{\text{YouTube-Stanford-CS229-Andrew Ng}}$  
# 
# - Choose $\theta$ so as to maximizes the log-likelihood $l(\theta)$
# - And then having chosen the value of $\theta$, when a new patient comes in, then use $h(\theta)$ to estimate the chance of this tumor in the new patient
# 
# - Update the parameter $\theta_{j}$ using "batch gradient ascent"
# > $\theta_{j} := \theta_{j} + \alpha\frac{\partial}{\partial \theta_{j}}l(\theta)$
# 
# - the difference between the equation above and one for linear regression is 
#   - instead of squared cost function, we are trying to optimize the log-likelihood function
#   - in least square, we tried to minimize the squared error. in logistic, we maximize the log likelihood
#   
# Note: learning rate $\alpha$ is missing from these equation below
# <img src="images/03_logisticVslinear.png" width=600 height=400 />   
# $\tiny{\text{YouTube-Stanford-CS229-Andrew Ng}}$  
#   
# <img src="images/03_logisticVslinear_gd.png" width=200 height=200 />   
# $\tiny{\text{YouTube-Stanford-CS229-Andrew Ng}}$  
# 
# - Plug in the value of $h(\theta)$ in log likelihood equation, and applying calculus/derivative, the gradient ascent equation becomes
# > $\theta_{j} := \theta_{j} + \alpha\sum\limits_{i=1}^{m}(y^{(i)} - h_{\theta}(x^{(i)}))x_{j}^{(i)}$
# - and for stochastic gradient ascent equation becomes
# > $\theta_{j} := \theta_{j} + \alpha(y^{(i)} - h_{\theta}(x^{(i)}))x_{j}^{(i)}$
# <br>
# 
# - This eqution looks similar to the one derived for linear regression. The difference is $h_{\theta}(x^{(i)})$ is now defined as a non-linear function of $\theta^{T}x^{(i)}$. 
# - Both linear regression and logistic regression are special case of GLM models

# ### Newton's method
# - Gradient descent takes lot of steps to converge
# - Newton's method allows lot bigger step
# - but each iteration is much more expensive
# 
# - 1-dimensional problem
# - Suppose we have function f and we want to find $\theta$, s.t. $f(\theta) = 0$
# - maximize $l(\theta)$ by finding $l'(\theta) = 0$
# 
# - Goal: find $f(\theta) = 0$
#   - take a random point $\theta^{(0)}$
#   - find tangent that touches the horizontal axis and this point becomes $\theta^{(1)}$
#   - repeat
#   
#   
# <img src="images/03_newtonMethodDia.png" width=600 height=400 />   
# $\tiny{\text{YouTube-Stanford-CS229-Andrew Ng}}$  
# 
# - Solve for the value of $\Delta$, and replace it to find value of $\theta$
#   
# <img src="images/03_newtonMethodDerv.png" width=600 height=400 />   
# $\tiny{\text{YouTube-Stanford-CS229-Andrew Ng}}$  
# 
# #### Quadratic convergence
# - In Newton's method, the convergence is quadratic
# 
# - When $\theta$ is a vector, i.e., $(\theta \in \mathbb R^{n+1})$
# - when n increases to say 1000 dimension, taking $H^{-1}$ gets difficult
# 
# - If the number of parameters is say 10,000 parameters, rather than dealing with 10Kx10k matrix inversion, stochastic gradient descent is a better choice
# - If the number of parameters is relatively smaller, Newton's method is a better choice
# 
# <img src="images/03_newtonMethodHigherDim.png" width=600 height=400 />   
# $\tiny{\text{YouTube-Stanford-CS229-Andrew Ng}}$  
# 
# 

# ## TODO
# - READ: Bayesian vs Frequentist statistics
