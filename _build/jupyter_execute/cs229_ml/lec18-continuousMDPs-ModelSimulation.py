#!/usr/bin/env python
# coding: utf-8

# # Lec 18-Continuous MDPs-Model Simulation
# 
# ## Outline
# - Continuous State MDPs 
#   - Discretization
#   - Models/Simulation
#   - Fitted value iteration

# ## Continuous state space 
# - In the last lecture we learnt about finite state models. 
# - The state space representation of car 
#   - kinematic model of car
#     - x - units in x direction
#     - y - units in y direction
#     - $\theta$ - orientation wrt some direction say North 
#   - dynamic model of car
#     - $\dot{x}$ - linear velocity in x direction
#     - $\dot{y}$ - linear velocity in y direction
#     - $\dot{\theta}$ - angular velocity 
# - it depends on how we want to model the car, if we are interested in modeling race car, we need to include the temperature as well, we will need to model the 4 car tires separately
# 
# - the state space representation of helicopter will have 3-dimensions instead of 2
# - inverted pendulum
# - the dimensions for the above problem statement will be $\mathbb R^{6}(\text{car}), \mathbb R^{12}(\text{helicopter}), \mathbb R^{4}(\text{pendulum})$

# - one easy way would be discretize continuous state space and solve it as discrete state space problem
# - its fine to solve low dimensional state space problem using discretization, but for higher dimensional problem we need to use fitted value iteration 
# - disadvantages of discretization:
#   - 1) Naive representation for $V^{*}$ and $\pi^{*}$
#     - in the house price prediction problem, fit a constant function
#     - the buckets will represent the price of a house as a function of the size 
#     - same applies to state-space problem, where we approximate the value function as a staircase function, as a function of set of stairs
#     - this can be applied if there is lot of data and very few input features, but it does not allow to fit a smoother function
#     - its not a very good representation
#     
#   - 2) Curse of dimensionality
#     - if state space $s \in \mathbb R^{n}$, and if we discretize each dimension into k values, then we get $k^{n}$ discrete states which grows exponentially 
#     - this is not a good representation in large dimensions say more than 8-10 
#     - way of reducing/taking control of exponential blow up would be to use _unequal grid_ for state space
#       - define finer or more number of state for more sensitive state-space and lesser number of state for less sensitive state-space
#       
# 
# <img src="images/18_discretizationHousePricePrediction.png" width=400 height=400>  
# $\tiny{\text{YouTube-Stanford-CS229-Andrew Ng}}$   
# 
# 

# ## Alternative approach
# - approximate $V^{*}$ directly without using discretization directly
#   - in linear regression problem
#     - approximate the value function where we learn V as a function of s
#     - approximate $y \approx \theta^{T}x$ or $\theta^{T} \phi(x)$  
#       - where $\phi(x)$ is the representation of features of x
#       - in the housing price problem, $\phi(x)$ is vector of x's the housing price $\{x_{1}, x_{2}, x_{1}^{2}, x_{1}x_{2}\}$, either as a function of real feature or complex set of features of house
#   - in fitted value iteration problem
#     - is a model where we will approximate $V^{*}(s)$, as a linear function of features of the state, with lot of modifications to approximate the value function
#     - approximate $V^{*}(s) \approx \theta^{T} \phi(s)$  
#     - once we have a good approximation of value function, we can compute the optimal action for every state

# - fitted value iteration problem works best with a model with simulator of the MDP
# - what does that mean?
# - Model (simulator) of a robot is a function that takes an input state s, an action a and outputs state s', drawn from the state space probability $P_{sa}$
# - for most of the MDP problems, the state space will be very high dimensional and the action space is much lower dimensional than the state space
#   - for car state s will be 6 dimensional, and action a will be 2 dimensional - steering and wheel
#   - for helicopter state s will be 12 dimensional, and action a will be 4 dimensional - left and right sticks with 2 controls each
#   - for pendulum state space s will be 4 dimensional and action a will be 1 - left or right
# - later we will see we will not need to discretize action space
#     
# 
# <img src="images/18_simulationModel.png" width=400 height=400>  
# $\tiny{\text{YouTube-Stanford-CS229-Andrew Ng}}$   
# 
# 

# ## How do we make a model
# - One way - use physics simulator package
#   - in pendulum case, 
#     - state space is 4 dim, use physics simulator package, plug in the dimension details, mass and other details, and the simulator will spit out how the state evolves from one time step to another time step
# - Second way - learn model from data
#   - more generally used
#   - lets say we have a helicopter and we want to build an autonomous controller for it
#     - start helicopter in some state $s_{0}$
#     - a human pilot will command the helicopter with some action $a_{0}$
#     - say in few seconds, the helicopter reaches a new position and orientation as $s_{1}$
#     - human pilot will keep on taking actions say every 10 times a second and we record its action and state. repeat for t seconds
#     - record multiple trajectories say m times
#     - apply supervised learning to predict $s_{t+1}$($s^{'}$) as function of state $s_{t}$($s$) and action $a_{t}$($a$)
#     - in linear regression version
#       - this model is fine for helicopter moving at slow speed
#       > $s_{t+1} = As_{t} + Ba_{t}$
#         - we can also plug in non-linear form of model depending on your choice of features of A as $\phi(s)$ and B as $\phi(a)$
#       - to fit the parameters A and B, that minimizes the squared difference of left and right hand side
#       > $\min\limits_{A,B}\sum\limits_{i=1}^{m}\sum\limits_{t=0}^{T}\|s_{t+1}^{(i)}-(As_{t}^{(i)} + Ba_{t}^{(i)}) \|^{2}$
#       - we can either model it as 
#         - deterministic
#           > $s_{t+1} = As_{t} + Ba_{t}$
#         - stochastic
#           > $s_{t+1} = As_{t} + Ba_{t} + \epsilon_{t}$
#           - where $\epsilon_{t} \sim N(0, \sigma^{2}I)$
#           - in this case, we will be sampling $\epsilon$ from a Gaussian vector and add it to the prediction of linear model 
#           - if we simulate the helicopter flying around, the simulator will generate random noise that adds or subtracts to the state space of helicopter as if there were little wind gusts blowing the helicopter around
#       - for high speed helicopters, this model will not work  
#   - this approach is called __model based reinforcement learning__
#     - to build a model of robot, we train the reinforcement learning algorithm in the simulator 
#     - then take the policy we learn in simulation
#     - apply back on the real robot
#     - this field is evolving quickly in robotics
#   - alternative is __model-free reinforcement learning__, 
#     - this is generally not much used as we cannot afford to run actual helicopter and learn from it
#     - instead model-free rl is generally used in video games 

# - if we design a deterministic simulators for robots and use it in a real robot, the odds of its success is very low
# - instead if we design a stochastic simulators and add noise term into it, the chances of its success increases for robotic control problems

# ## How do you approximate the value function
# - in order to apply, fitted value iteration
# - first step is to choose features $\phi(s)$ of the state s
# - approximate V(s)
# > $V(s) = \theta^{T}\phi(s)$
# - determine the features of $\phi(s)$ as nonlinear features that we suppose might be representative of value/how well the robot is doing
# - value of state is the expected payoff/discounted rewards from that state
# - the value function captures if the robot starts off in that state, how well would it do

# ## Fitted value algorithm
# - Value iteration
# > $\begin{equation}\\
# \begin{aligned}\\
# V(s) &= R(s) + \gamma\max\limits_{a}\sum\limits_{s'\in S}P_{sa}(s')V(s')\\
# &= R(s) + \gamma\max\limits_{a}E_{s'\sim P_{sa}}[V(s')]\\
# \end{aligned}\\
# \end{equation}\\
# $
# 
# - Fitted value iteration
#   - Choose set of states randomly $\{ s^{(1)}, s^{(2)}, ..., s^{(m)}\} \in S$
#   - Initialize $\theta := 0$
#     - in general
#       - we map $x \rightarrow y$, 
#       - as we have a finite set of examples that we see as values of x in the training set for predicting housing prices
#     - similarly
#       - here we map $s \rightarrow V(s)$
#       - we will use a certain set of states, and use that finite set of examples s in a linear regression to fit V(s)
#   - that is the use of sample above
#   - Repeat { - _Loop1_
#     - for i = 1,2,..,m {  - _Loop2_
#       - for each action $a \in A$ { - _Loop3_
#         - sample $\{ s'^{(1)}, s'^{(2)}, ..., s^{(k)}\} \sim P_{s^{(i)},a}$ - _Step4_
#         - set $q(a) = \frac{1}{k}\sum\limits_{j=1}^{k}[R(s^{(i)}) + \gamma V(s'_{j})]$ - which is the estimate of expectation - _Step5_
#       - }
#       - set $y^{(i)} = \max\limits_{a}q(a)$ - _Step6_
#     - }
#     - $\theta := \arg\min\limits_{\theta}\frac{1}{2}\sum\limits_{i=1}^{m}(\theta^{T}\phi(s^{(i)}) - y^{(i)})^{2}$ - _Step7_
#   - }
#   
#   - In _Loop2_, m represents how many times we fly the helicopter in order to build a model
#   - in _Step4_, we took k samples from $V(s'_{j})$ distribution, average V(s') in _Step5_
#     - we sampled k times because we use stochastic simulator here
#     - if we use a deterministic simulator instead, for a given state and action, it will always result in same state $s^{(i)},a \rightarrow s'$
#     - for deterministic simulator, we set k=1
#   - in _Step5_, we get $V(s'_{j})$ from using the parameters of $\theta^{T}\phi(s_{j}')$ from the last iteration of fitted value iteration
# 
# > $\begin{equation}\\
# \begin{aligned}\\
# V(s) &= R(s) + \gamma\max\limits_{a}\sum\limits_{s'\in S}P_{sa}(s')V(s')\\
# &= R(s) + \gamma\max\limits_{a}E_{s'\sim P_{sa}}[V(s')]\\
# &= \max\limits_{a}E_{s'\sim P_{sa}}[R(s) + \gamma V(s')]\\
# &= \max\limits_{a}E_{s'\sim P_{sa}}[q(a)]\\
# \end{aligned}\\
# \end{equation}\\
# $

# - In _Step6_, we set $y^{(i)} = \max\limits_{a}q(a)$ - is the estimate at right-hand side of value iteration
#   - in the original value iteration 
#     - $V(s^{(i)}) := y^{(i)}$
#   - in fitted value iteration 
#     - we want $V(s^{(i)}) \approx y^{(i)}$
#     - i.e., $\theta^{T}\phi(s^{(i)}) \approx y^{(i)}$

# - In _Step7_: run linear regression to choose the parameters $\theta$ that minimizes the squared error
# > $\theta := \arg\min\limits_{\theta}\frac{1}{2}\sum\limits_{i=1}^{m}(\theta^{T}\phi(s^{(i)}) - y^{(i)})^{2}$

# ## Some practical aspects
# - After we learn all the parameters, fitted VI gives approximation to $V^{*}$, which implicitly defines $\pi^{*}$
# > $\pi^{*}(s) = \arg\max\limits_{a}E_{s'\in P_{s,a}}[V^{*}(s')]$
# - in real world helicopter fly, we will have to choose control actions at 10Hertz, i.e., 10 times a second we are given the state, and we have to choose an action
# - same would be with a self driving car, where we choose a new action 10 times a second
# - but how do we compute this expectation and maximize this 10 times per second?
#   - we used samples $\{ s'^{(1)}, s'^{(2)}, ..., s^{(k)}\} \sim P_{s^{(i)},a}$ to approximate expectation
# - Tricks
#   - simulator is of the form
#     - $s_{t+1} = f(s_{t}, a_{t}) + \epsilon_{t}$, i.e., $(As_{t} + Ba_{t} + \epsilon_{t})$ - previous state and action plus some noise
#   - for deployment/runtime
#     - set $\epsilon_{t}=0$ and $k=1$ 
#     - in simulator we need to have noise and randomness
#     - but for real practical purposes, we can use discretized state and action V
