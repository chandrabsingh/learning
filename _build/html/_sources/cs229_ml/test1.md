# test1

## Outline
- Logistic Regression as NN
- Neural Network

## Deep Learning
- used widely in computer vision, NLP, speech recognition
- computationally expensive, parallelize
- data is growing and is available
- algorithms are growing

## Logistic Regression as NN

### Goal 1: find cats in images - binary classification
  - 1 - presence of cat
  - 0 - absence of cat
- images can be represented in matrices - 64x64x3 (RBG filter for 3)
- flatten pixels into vector
- push vector into an logistic operation, which has two parts
  - linear function: wx + b
  - sigmoid function: a function that takes numbers in the range of $-\infty$ and $\infty$ and maps it to 0 to 1
- the equation is $\hat{y} = \sigma(\theta^{T}x) = \sigma(wx+b)$
- vector x size is (64x64x3,1) = (12288,1)
- vector w size is (1,64x64x3) = (1,12288)
- once we have the model, we will train it 
  - initialize w(weights),b(bias)
  - find the optimal w,b - means define loss function
    - use a proxy which will be your loss function
    - try to minimize this loss function, to find the right parameters
    - logistic loss function $L = -[y\log\hat{y} + (1-y)\log(1-\hat{y}) ]$
      - this comes from maximum likelihood function
      - how can we minimize this function - using gradient descent
        - $w = w - \alpha\frac{\partial L}{\partial w}$
        - $b = b - \alpha\frac{\partial L}{\partial b}$
        - Update it iteratively
  - use it to predict 


<img src="images/11_logisticFun.png" width=400 height=400>  
$\tiny{\text{YouTube-Stanford-CS229-Andrew,Ng/Kian Katanforoosh}}$  

- Eq1: neuron = linear + activation (linear=wx+b; activation=sigmoid function)
- Eq2: model = architecture + parameter

#### Goal 2: Find cat/lion/iguana in images
- how will you modify network to this into effect
  - add 2 more neurons
- Notations
  > $\hat{y}_{1} = a_{1}^{[1]} = \sigma(w_{1}^{[1]}x + b_{1}^{[1]})$  
  > $\hat{y}_{2} = a_{2}^{[1]} = \sigma(w_{2}^{[1]}x + b_{2}^{[1]})$  
  > $\hat{y}_{3}= a_{3}^{[1]} = \sigma(w_{3}^{[1]}x + b_{3}^{[1]})$  
    - square bracket [1] - represent index of a layer
    - in one layer, neurons dont communicate with each other
    - the subscript number represents the neuron inside the layer
    - $a_{2}^{[1]}$ represents the activation part of 2nd neuron of 1st layer
    
- How many parameters do you need to train 
  - 3 x 12288
  - images and labels( with cats or without)
  - we can represent labels in the format

$$
\begin{equation}\
\text{cat is represented as =}\
\begin{bmatrix}\
1 \\
0 \\
0 \\
\end{bmatrix}\
\end{equation}\
$$

  - we can have images where there are both cats and lion in it
    - how will nn detect which is what
  - neuron will understand by itself, if cats are alone, or cats with lion are in image, or cats with iguana, or only lions - neuron will detect that by itself
  - neurons are not communicating to each other in a layer
  - neurons are trained independently from each other
  - in the present form, it is simply 3 logistic regression, its not neural network yet
  

<img src="images/11_logisticFun2.png" width=400 height=400>  
$\tiny{\text{YouTube-Stanford-CS229-Andrew Ng/Kian Katanforoosh}}$   


- __How to train these function__
$$ L_{3N} = -\sum\limits_{k=1}^{3}\left[ y_{k} \log\hat{y}_{k} + (1-y_{k})\log(1-\hat{y}_{k}) \right]  $$
- sum loss for each neuron

- Derivative of loss function 
  - if we take derivative of loss function wrt $w_{1}$, i.e., $\frac{\partial L_{3N}}{\partial w_{2}^{[1]}}$ the result will be the same as before, because derivative of other 2 terms will be zero
$$ w = w - \alpha\frac{\partial L}{\partial w} $$


