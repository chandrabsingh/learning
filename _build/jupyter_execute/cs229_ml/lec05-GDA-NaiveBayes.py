#!/usr/bin/env python
# coding: utf-8

# # Lec 05-GDA - Naive Bayes
# 
# ## Outline
# * Discriminative Learning Algorithms (earlier)
#   * GLM
#   * Logistic Regression
#   * Linear Regression
# * Generative Learning Algorithms (this lecture)
#   * Gaussian Discriminant Analysis (GDA)
#   * Generative and Discriminative learning algorithm - comparision
#   * Naive Bayes - build spam filter
# 

# ## Discriminative and Generative algorithm
# 
# - Difference between Discriminative and Generative algorithm
# * Say there are malignant and benign tumors, with 2 classes
#   * A discriminant learning algorithm like logistical regression will use gradient descent to find the line that separates the positive and negative class, tumor in this case. 
#     * GD - Randomly initialize the parameter, which eventually evolves to be the separating line/plane to differentiate positive and negative examples
#     * It searches for the separation, trying to maximize the likelihood
#   * A generative learning algorithm instead looks into each of the classes one at a time, and try to make a model of how all the malignant tumors looks like in isolation
#     * If a new classification example
#     * It builds a model of how to classify the classes in isolation and test
#     * so when new patient example comes, it compares it to malignant tumor model compared to benign tumor model and then say which class example does it looks like

# ## Generative Learning Algorithm  
# * Discrimative Learning Algorithm
#   * Learn __p(y|x)__ or 
#     * Learn ($h_{\theta}(x)$) = {0 or 1 directly} 
#     * or learn about some function mapping from x to y directly  
# * Generative Learning Algorithm:  
#   * 1) Learns __p(x|y)__ where  
#     * x denotes features  
#     * y denotes class  
#     * given that the tumor is malignant, what will the features gone be like?
#   * 2) Learns __p(y)__ - class prior 
#     - even before you see a patient what are the chances of having malignant tumors?
#   * Can solve p(y=1|x) using Bayes rule:
#     > $p(y=1|x) = \frac{p(x|y=1)p(y=1)}{p(x)} \\ $
#     > where $p(x)=p(x|y=1)p(y=1)+p(x|y=0)p(y=0)$

# Examples of Generative Learning Algorithm:
# * Discrete value feature - Email spam filter, twitter positive/negative feature
# * Continuous value feature - Tumor classification

# ## Gaussian Discriminant Analysis (GDA)  
# - Suppose $x \in \mathbb R^{n}$  (drop out the convention $x_{0}=1$, instead of n+1)
# - Assume P(x|y) is distributed Gaussian, features are Gaussian (i.e., conditioned on the tumors being malignant, distribution of the features is Gaussian)
#   > $Z \in N(\mu, \sum) $  
#   - where $Z \in \mathbb R^{n}, \mu \in \mathbb R^{n}, \sum \in \mathbb R^{nxn} $  
#   > $ E[Z] = \mu $  
#   > $Cov(Z) = E[(Z-\mu)(Z-\mu)^{T}]$  
#   > $= Ezz^{T} - (Ez)(Ez)^{T}$  
# - Probability distribution for multivariate normal distribution  
#   > $P\left(x;\mu,\sum\right) = \frac{1}{(2\pi)^{d/2}|\sum|^{1/2}}exp\left(-\frac{1}{2}(x-\mu)^{T}\sum^{-1}(x-\mu)\right)$  
#   
# ### Covariance/mean matrix 
# - Impact of shrinking the variance, ie, multiply covariance matrix by integer less than 1
#   - probability density becomes taller
#   - the area under the curve is always 1, it simply reduces the spread of Gaussian density
# - Impact of expanding the variance, ie, multiply covariance matrix by integer more than 1
#   - probability density spreads out
#   - the area under the curve is always 1, it simply widens the spread of Gaussian density
# - Standard multivariate distribution
#   - covariance matrix is identity matrix
#   - uncorrelated z1 and z2
# - For standard covariance matrix, the off diagonal matrix is 0. By increasing this to 0.8, 
#   - changes from round shaped distribution to flatter
#   - positively correlated z1 and z2
# - For standard covariance matrix, the off diagonal matrix is 0. By increasing this to -0.8, 
#   - changes from round shaped distribution to flatter
#   - negatively correlated z1 and z2
# - For standard Gaussian, the mean is centered at 0. By varying the mean
#   - shift the center of Gaussian density
# 
# 

# ### GDA Model
# Density of each feature is Gaussian - with parameters of GDA model are $\mu_{0}, \mu_{1}, \sum$. Note the convention of covariance matrix is to use the same $\sum$. 
# - benign tumor
#   > $P(x|y=0) = \frac{1}{(2\pi)^{d/2}|\sum|^{1/2}}exp\left(-\frac{1}{2}(x-\mu_{0})^{T}\sum^{-1}(x-\mu_{0})\right)$ 
# 
# - malignant tumor
# > $P(x|y=1) = \frac{1}{(2\pi)^{d/2}|\sum|^{1/2}}exp\left(-\frac{1}{2}(x-\mu_{1})^{T}\sum^{-1}(x-\mu_{1})\right)$  
# 
# - Assumption: the two Gaussian for positive and negative case have same covariance but different mean

# #### Model - class prior
# - Model P(y), where y is a Bernoulli random variable, which takes 0 or 1
# \begin{align} P(y) = \phi^{y}(1-\phi)^{(1-y)} \end{align}    
# 
# - which means $P(y=1) = \phi$. This is the way of writing if..else statement in mathematical terms  
# \begin{align} \mu_{0} \in \mathbb R^{n}, \mu_{1}\in \mathbb R^{n}, \sum\in \mathbb R^{nxn},\phi \in \mathbb R \end{align}  

# #### How to fit the parameter:  
# ##### Generative Learning Algorithm
# - Training set $\{x^{(i)}, y^{(i)}\}_{i=1}^{m}$
# - To fit the parameter, we will **maximize the joint likelihood**  and P(y)
#   - For generative models, we try to choose parameters such that P(x and y) is maximized. 
#   - Likelihood of parameters
#     > $L\left(\phi, \mu_{0}, \mu_{1}, \sum\right) $  
#     > $= \prod \limits_{i=1}^{m} P\left(x^{(i)}, y^{(i)}; \phi,\mu_{0},\mu_{1},\sum\right) $  
#     > $=\prod \limits_{i=1}^{m} P\left(x^{(i)}| y^{(i)};\mu_{0},\mu_{1},\sum\right) P(y^{(i)};\phi)$ 

# ##### Discriminant Learning Algorithm
# - Cost function that you **maximize the conditional likelihood**, such that 
#   - Likelihood of parameters
#     > $L\left(\theta\right) = \prod \limits_{i=1}^{m} P\left(y^{(i)}|x^{(i)}; \theta\right) $  
#     
#   - Difference
#     - For logistic models or generalized linear models, we try to choose parameter $\theta$, such that it maximizes P(y|x). 
#     - For generative learning algorithms, we try to choose parameter that maximize p(x|y)

# ##### Maximum likelihood estimate  
# - you choose the parameter $\phi, \mu_{0}, \mu_{1}, \sum$ that maximize the log likelihood $L\left(\phi, \mu_{0}, \mu_{1}, \sum\right) $
# - you take the derivative of log likelihood and set it to zero, to get the parameter values
# - The value of $\phi$ that maximizes the estimate is:
#   - $\phi$ is the estimate of probability of y being equal to 1
#   - what is the chance when the next patient walks into your clinic, that they have a malignant tumor? 
#   > $\phi = \frac{\sum\limits_{i=1}^{m} y^{(i)}}{m} = \frac{\sum\limits_{i=1}^{m} \mathbb 1 \{ y^{(i)} = 1 \} }{m} $, 
#     - where $\mathbb 1 \{true\} = 1 \space$ and $\mathbb 1 \{false\} = 0 $. 

# - The value of $\mu_{0}$ that maximizes the estimate is (benign tumors in training set):
# 
# > $\mu_{0} = \frac{\sum\limits_{i=1}^{m} \mathbb 1 \{ y^{(i)} = 0 \}.x^{(i)} }{\sum\limits_{i=1}^{m} \mathbb 1 \{ y^{(i)} = 0 \} } = \frac{\text{Sum of feature vectors for all the examples with y=0}}{\text{Sum of all examples with y=0}}$. 
# 
# <br>
# 
# - The value of $\mu_{1}$ that maximizes the estimate is (malignant tumors in training set):
# 
# > $\mu_{1} = \frac{\sum\limits_{i=1}^{m} \mathbb 1 \{ y^{(i)} = 1 \}.x^{(i)} }{\sum\limits_{i=1}^{m} \mathbb 1 \{ y^{(i)} = 1 \} } = \frac{\text{Sum of feature vectors for all the examples with y=1}}{\text{Sum of all examples with y=1}}$. 
# 
# <br>
# 
# - The value of $\Sigma$ that maximizes the estimate is:  
# > $\Sigma = \frac{1}{m}\sum\limits_{i=1}^{m}(x^{(i)} - \mu_{y^{(i)}})(x^{(i)} - \mu_{y^{(i)}})^{T}$

# #### How to make prediction
# Having fit these parameters, how to make a prediction
# - to make predict most likely class label, you choose $\max\limits_{y} p(y|x)$
# - by Bayes' rule
# > arg $ \max\limits_{y} p(y|x) = arg \max\limits_{y} \frac{p(x|y)p(y)}{p(x)} $  
# - since p(x) is a constant  
# > = arg $ \max\limits_{y} p(x|y)p(y) $
# 
# - If we care about prediction, computation can be saved by not calculating denominator
# - If we care about probability, then we must normalize the probability
# 
# ##### Questions
# 
# - What is arg min?
#   - $min(z-5)^{2} = 0$
#   - arg $min(z-5)^{2} = 5$ - the value that is required to achieve the smallest possible value
# 
# <br>
# 
# - Why do we choose to have a single covariance matrix but different mean for each class variable?
#   - the decision boundary ends up being linear, if we go with this setting
#   - by choosing covariance matrix for each class variable
#     - you will end up with decision boundary that is non-linear
#     - and the number of parameters will get doubled up
# 
# #### GDA vs Logistic Regression:  
# - How to plot 
# 
# <img src="images/05_gda_log_plot.png" width=400 height=400>  
# $\tiny{\text{YouTube-Stanford-CS229-Andrew Ng}}$  
# 
# - Assumptions
#   - GDA assumes that x|y=0 and x|y=1 are Gaussian and y is Bernoulli
#   - Logistic assumes that y=1|x is a sigmoid function
#   - GDA's assumption implies that p(y=1|x) is governed by logistic function
#   - But the implication in opposite direction is not true
#     - i.e., if you assume that p(y=1|x) is governed by logistic function, then this does not in any way assume that x|y is Gaussian  
# 
# - What this means is 
#   - GDA makes stronger set of assumptions
#   - Logistic makes weaker set of assumptions
#   
# - If you make strong modeling assumptions, and if your modeling assumptions are roughly correct, then your model will do better because you are telling more information to the algorithm.
#   - So if x|y is Gaussian, then GDA will do better because you are telling the algorithm that x|y is Gaussian and so it can be more efficient
#   
# - If the GDA assumptions are wrong, and if x|y is not at all Gaussian, then GDA assumptions will be bad set of assumptions to make and will make GDA perform badly.
# 
# - When is each of the algorithm superior
#   - If you make weaker assumptions as in logistic regression, then your algorithm will be more robust to modeling assumptions such as accidently assuming the data is Gaussian and it is not
#    
#   - But if there is fewer data, making stronger assumptions is better
# 
# <img src="images/05_gda_log1.png" width=400 height=400>  
# $\tiny{\text{YouTube-Stanford-CS229-Andrew Ng}}$  
# 
# - Given a choice between GDA and logistic regression, its better to use GDA because its very efficient algorithm, you compute mean, and covariance matrix and its done. There is no iterative process involved.  
# - When performance of all algorithm is better when the data is more. 
# - But the performance of very good algorithm will perform better with less data
# - Skill comes in assumptions more when the data is less, as what is the underlying distribution

# ## Naive Bayes - Generative Learning 
# - spam or not - text classification
# - based on the subject line, categorize the email
# 
# ### How do you categorize the feature vector "x"?  
# - make list of all words 
# - make a binary feature vector of all the words, if it exists in email or not. $x \in \{0,1\}^{n}$
# 
# 
# $x_{i} = \mathbb 1$ {word i appear in email or not}
# 
# In Naive Bayes, we will build a generative learning algorithm, so we need to model p(x|y) and p(y)  
# 
# - say there are n = 10000 words - possible combinations of X are $2^{10000}$ 
#   - excessive number of parameters, will not work
# 
# 
# - Assume $x_{i}^{'}s$ are conditionally independent given y  
# > $P(x_{1},x_{2},..,x_{10000}|y) = P(x_{1}|y)P(x_{2}|x_{1},y)P(x_{3}|x_{1},x_{2},y)..P(x_{10000}|...)$  
# - the above statement is true by chain rule of probability
# - using conditional independence assumption or Naive Bayes assumption, we get  
# > $ \stackrel{assume}{=} P(x_{1}|y)P(x_{2}|y)P(x_{3}|y)..P(x_{10000}|y)$  
# > $= \prod\limits_{i=1}^{n}p(x_{i}|y)$
# - This is a strong assumption
# 
# <br>
# 
# - The reasoning behind conditional independence assumption is: given the information y is spam, it does not matter which words appear in email as $x_{1}$ is accountNumber, $x_{2}$ is mortgage, $x_{n}$ is yield, 
# 
# 
# <img src="images/05_conditionalIndependence.png" width=200 height=200>  
# $\tiny{\text{YouTube-Stanford-CS229-Andrew Ng}}$  
# 
# 

# #### Parameters
#   - $\phi_{j|y=1} = p(x_{j}=1|y=1)$
#     - if a spam email(y=1 is spam, y=0 is non-spam), what is the chance of word j appearing in the email? 
#   - $\phi_{j|y=0} = p(x_{j}=1|y=0)$
#     - if a non-spam email, what is the chance of word j appearing in the email?
#   - $\phi_{y} = p(y=1)$
#     - what is the cost prior, i.e., what is the prior probability that the next email you receive in your inbox is spam email?
# 
# #### Joint likelihood estimate
# - $L(\phi_{y}, \phi_{j|y}) = \prod\limits_{i=1}^{m} p(x^{(i)}, y^{(i)}; \phi_{y}, \phi_{j|y})$
# 
# #### MLE
# - fraction of spam email
# > $\phi_{y} = \frac{\sum\limits_{i=1}^{m} \mathbb 1 \{y^{(i)} = 1\} }{m}$
# 
# - estimate that the chance of word j appearing 
#   - denominator - find all the spam emails 
#   - numerator - of all the spam emails, count what fraction of them had word j in it
# > $\phi_{j|y=1} = \frac{\sum\limits_{i=1}^{m} \mathbb 1 \{ x_{j}^{(i)} = 1, y^{(i)} = 1 \}} {\sum\limits_{i=1}^{m} \mathbb 1 \{y^{(i)} = 1\} }$
