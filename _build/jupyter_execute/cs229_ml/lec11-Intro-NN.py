#!/usr/bin/env python
# coding: utf-8

# # Lecture 11
# 
# ## Outline
# - Logistic Regression as NN
# - Neural Network

# ## Deep Learning
# - used widely in computer vision, NLP, speech recognition
# - computationally expensive, parallelize
# - data is growing and is available
# - algorithms are growing

# ## Logistic Regression as NN
# 
# ### Goal 1: find cats in images - binary classification
#   - 1 - presence of cat
#   - 0 - absence of cat
# - images can be represented in matrices - 64x64x3 (RBG filter for 3)
# - flatten pixels into vector
# - push vector into an logistic operation, which has two parts
#   - linear function: wx + b
#   - sigmoid function: a function that takes numbers in the range of $-\infty$ and $\infty$ and maps it to 0 to 1
# - the equation is $\hat{y} = \sigma(\theta^{T}x) = \sigma(wx+b)$
# - vector x size is (64x64x3,1) = (12288,1)
# - vector w size is (1,64x64x3) = (1,12288)
# - once we have the model, we will train it 
#   - initialize w(weights),b(bias)
#   - find the optimal w,b - means define loss function
#     - use a proxy which will be your loss function
#     - try to minimize this loss function, to find the right parameters
#     - logistic loss function $L = -[y\log\hat{y} + (1-y)\log(1-\hat{y}) ]$
#       - this comes from maximum likelihood function
#       - how can we minimize this function - using gradient descent
#         - $w = w - \alpha\frac{\partial L}{\partial w}$
#         - $b = b - \alpha\frac{\partial L}{\partial b}$
#         - Update it iteratively
#   - use it to predict 
# 
# 
# <img src="images/11_logisticFun.png" width=400 height=400>  
# $\tiny{\text{YouTube-Stanford-CS229-Andrew Ng/Kian Katanforoosh}}$  
# 

# - Eq1: neuron = linear + activation (linear=wx+b; activation=sigmoid function)
# - Eq2: model = architecture + parameter

# ### Goal 2: Find cat/lion/iguana in images
# - how will you modify network to this into effect
#   - add 2 more neurons
# - Notations
#   > $\hat{y}_{1} = a_{1}^{[1]} = \sigma(w_{1}^{[1]}x + b_{1}^{[1]})$  
#   > $\hat{y}_{2} = a_{2}^{[1]} = \sigma(w_{2}^{[1]}x + b_{2}^{[1]})$  
#   > $\hat{y}_{3}= a_{3}^{[1]} = \sigma(w_{3}^{[1]}x + b_{3}^{[1]})$  
#     - square bracket [1] - represent index of a layer
#     - in one layer, neurons dont communicate with each other
#     - the subscript number represents the neuron inside the layer
#     - $a_{2}^{[1]}$ represents the activation part of 2nd neuron of 1st layer
#     
# - How many parameters do you need to train 
#   - 3 x 12288
#   - images and labels( with cats or without)
#   - we can represent labels in the format
# $$
# \begin{equation} \
# \text{cat is represented as = }  \
# \begin{bmatrix} \
# 1 \\
# 0 \\
# 0 \\  
# \end{bmatrix}
# \end{equation}
# $$
#   - we can have images where there are both cats and lion in it
#     - how will nn detect which is what
#   - neuron will understand by itself, if cats are alone, or cats with lion are in image, or cats with iguana, or only lions - neuron will detect that by itself
#   - neurons are not communicating to each other in a layer
#   - neurons are trained independently from each other
#   - in the present form, it is simply 3 logistic regression, its not neural network yet
#   
# 
# <img src="images/11_logisticFun2.png" width=400 height=400>  
# $\tiny{\text{YouTube-Stanford-CS229-Andrew Ng/Kian Katanforoosh}}$   
# 
# - __How to train these function__
# $$L_{3N} = -\sum\limits_{k=1}^{3} \left[ y_{k}\log\hat{y}_{k} + (1-y_{k})\log(1-\hat{y}_{k}) \right]$$
# - sum loss for each neuron
# 
# - Derivative of loss function 
#   - if we take derivative of loss function wrt $w_{1}$, i.e., $\frac{\partial L_{3N}}{\partial w_{2}^{[1]}}$ the result will be the same as before, because derivative of other 2 terms will be zero
# $$ w = w - \alpha\frac{\partial L}{\partial w}$$
# 
# 
# 

# ### Goal 3: add constraint, unique animal in an image
# - use softmax function
# > $\frac{e^{z_{3}^{[1]}} }{\sum\limits_{k=1}^{3} e^{z_{k}^{[1]}}}$
# - neuron has a linear part and an activation
#   - let $z_{1}^{[1]}$ be the linear part of first neuron
#   - sum of the output of the network must add up to 1
# - instead of getting a probabilistic output for each $\hat{y}_{1}$, $\hat{y}_{2}$, $\hat{y}_{3}$, now we will get a probability distribution over all the classes
# - earlier we got probability output was if image is cat or not (0 to 1), if image is lion or not (0 to 1), if image is iguana or not (0 to 1)
# - now the vector will add up to 1
# - the 3 probabilities are dependent on each other - if there is no cat, no lion then it means there is iguana
# - this is __softmax multi-class regression__
# 
# <br>
# 
# - __Cross-entropy loss__
# > $L_{CE} = -\sum\limits_{k=1}^{3} y_{k}\log\hat{y}_{k}$
# - this binary loss function is used in multi-class classification problems

# ### Question
# - Instead of predicting cats or no cats, how will you predict the age of cat based on image
#   - Approach 1
#     - make several output nodes corresponding for each age of cat
#     - which network would you use - 2nd (tree NN) or 3rd (softmax NN)
#       - Softmax NN because you have unique node for each age
#   - Approach 2
#     - Use regression to predict age from 0 to certain number
#     - how will you modify the network to use regression
#       - Modification 1
#         - change the sigmoid to linear function. But using linear function, this whole setup will become linear regression
#         - instead use ReLU - the function is linear but for negative values it is zero
#         - ReLU - Rectified Linear Unit
#       - Modification 2
#         - modify the loss function to fit the regression type of problem with $(\hat{y} - y)$ or L2 type of loss $\Vert \hat{y} - y \Vert^{2}$
#         - the reason why we use this loss is because it is easier to optimize for regression task rather than for classification task
#     
#     

# ## Neural Networks
# 
# ### Goal 1: binary classification for cat image(1) or no cat(0)
# - Add more layers 
# - the output layer must have the same number of neurons as the number of classes to be for reclassification/regression
# - how many parameters does this network have? 
#   - For layer 3 - 3 neurons - 3n weights + 3 biases
#   - For layer 2 - 2 neurons - 2x3 (3 neurons connected to 2 neurons) weights + 2 biases
#   - For layer 1 - 1 neuron  - 2x1 (2 neurons connected to 1 neuron) weight  + 1 bias
# - Notation
#   - neurons within layer dont interact with each other
#   - 1st layer is called input layer
#   - last layer is called output layer as it interacts with output
#   - middle layers are called hidden layer, as the inputs and output are hidden from this layer
#     - there is abstraction of input to it
# - what is interesting about NN
#   - the first layer understands the fundamental of images which is edges
#   - each neuron within the first layer will understand some specific type of edge
#   - then these neurons in the first layer will pass information of  edge to the second layer neurons, which will help them to detect body parts say ears or mouth
#   - then they will construct face of cat to communicate if it is a cat or not
# - why is middle layers called hidden layer
#   - because we dont know what it will figure out
#   - it can understand complex information if it has enough data
#   
# 
# <img src="images/11_nn1.png" width=400 height=400>  
# $\tiny{\text{YouTube-Stanford-CS229-Andrew Ng/Kian Katanforoosh}}$   
# 

# ### House price prediction
# - Parameters
#   - #bedrooms
#   - size
#   - zip code
#   - wealth
# - If hand-engineered, we would have combined specific features to get specific information
#   - zip and wealth - gives information about - school quality
#   - #bedroom and size - gives information about - family size
#   - zip - gives information about - walkable or not
#     - in the next layer can use above features to predict price
# - but this example above was not fully connected or was hand-engineered
# 
# <img src="images/11_nn_housePrediction1.png" width=400 height=400>  
# $\tiny{\text{YouTube-Stanford-CS229-Andrew Ng/Kian Katanforoosh}}$   
# 
# - instead we want network to be __fully connected__ to figure out by itself, what feature information it can extract from data
# - this is the reason why neural networks are often called __black-box models__
#   - its hard to figure out which edge of a  neuron is detecting the weighted average of input features
# - this is often called __end-to-end learning__ because we have an input, a ground truth and we don't constrain the network in middle,  we let it learn by itself
# 
# <img src="images/11_nn_housePrediction2.png" width=400 height=400>  
# $\tiny{\text{YouTube-Stanford-CS229-Andrew Ng/Kian Katanforoosh}}$   
# 
# 

# ### Propagation equations (Forward propagation)
# 
# - it is forward because it goes from input to output
# 
# <img src="images/11_nn_propagationEq.png" width=400 height=400>  
# $\tiny{\text{YouTube-Stanford-CS229-Andrew Ng/Kian Katanforoosh}}$   
# 
# - what is the difference between sgd and gd?
#   - sgd updates weights and bias after you see each example, so direction of gradient is quite noisy, it doesn't represent the entire batch
#   - gd/bgd updates weights and bias after you see whole batch of examples, so gradient is much more precise

# - What happens for an input batch for m examples
#   - X is no more a single vector, but a matrix of m examples in the form of column vector
# > $\begin{pmatrix}|&|&|&|\\|&|&|&|\\x^{(1)}&x^{(2)}&...&x^{(m)}\\|&|&|&|\\|&|&|&|\end{pmatrix}$  
# - we want to parallelize the computation as much as possible by feeding it more batches of inputs
# - so now batch of m inputs will be fed
# - Notation
#   - [2] to denote 2nd layer
#   - (2) to denote 2nd batch of input/2nd training example/image
#   - z - lowercase to represent vector equations
#   - Z - uppercase to represent matrix form of batch equations
# 
# <img src="images/11_nn_propagationEq_Matrix.png" width=600 height=600>  
# $\tiny{\text{YouTube-Stanford-CS229-Andrew Ng/Kian Katanforoosh}}$   
# 

# ### Optimizing function
# - the parameters to optimize: $W^{[1]}, W^{[2]}, W^{[3]}, b^{[1]}, b^{[2]}, b^{[3]} $
# - define loss/cost function
# 
# #### Cost/Loss function
# - cost function depends on ($\hat{y}, y$)
# > $J(\hat{y}, y) = \frac{1}{m}\sum\limits_{i=1}^{m}L^{(i)}$
# - where 
# > $L^{(i)} = -[y^{(i)}\log\hat{y}^{(i)} + (1-y^{(i)})\log(1-\hat{y}^{(i)}) ]$  

# #### Optimize - Backward propagation
# - why is it called backward propagation
#   - which derivative would you like to start with? - $W^{[1]}, W^{[2]}, W^{[3]}$
#     - $W^{[3]}$
#   - Why $W^{[3]}$?
#     - which relation is easier to understand? - $W^{[3]}$ and loss function or $W^{[1]}$ and loss function
#      - $W^{[3]}$
#   - why?
#     - because $W^{[3]}$ happens much later in the network, so it is simple
#   - if you want to understand how much should $W^{[1]}$ move in order to make the loss move, its much more complicated than $W^{[3]}$, because there are lot more connections of $W^{[1]}$
#   - this is the reason why it is called backward propagation
#   
# - we first calculate derivative wrt $W^{[3]}$, then use the result in derivative wrt $W^{[2]}$ and finally wrt $W^{[1]}$
# - we use chain rule here
# - we use chain rule wrt parameters on which it has dependencies
# - we store the right values and we continue to back propagate
# - we need forward propagation equations to determine which path to take in chain rule
# - we need to choose the right path in proper way so that there's no cancellation
# - for example: $Z^{[2]}$ is connected to $W^{[2]}$, but $a^{[1]}$ is not connected to $W^{[2]}$
# 
# <img src="images/11_nn_backPropagationEq.png" width=400 height=400>  
# $\tiny{\text{YouTube-Stanford-CS229-Andrew Ng/Kian Katanforoosh}}$   
# 
# 
# 
