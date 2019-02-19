
# week1 paper reviews for ADF
scheduled date: Feb. 1 - Feb. 7  
<p align="right">  
<a href="README.md">back to table</a>;<a href="https://github.com/5loaves-2fish-12basckets/ADF_studies/">back to schedule</a>
</p>
 
> paper list:  
[Are Adversarial Examples inevitable](#Are-Adversarial-Examples-inevitable)    
[Adversarial Examples: Attacks and Defenses for Deep Learning](#Adversarial-Examples-Attacks-and-Defenses-for-Deep-Learning)    
[Explaining and Harnessing Adversarial Examples](#Explaining-and-Harnessing-Adversarial-Examples)  
[Intriguing properties of neural networks](#Intriguing-properties-of-neural-networks)    
[Towards Evaluating the Robustness of Neural Networks](#Towards-Evaluating-the-Robustness-of-Neural-Networks)  
[Towards Deep Learning Models Resistant to Adversarial Attacks](#Towards-Deep-Learning-Models-Resistant-to-Adversarial-Attacks)  

## Are Adversarial Examples inevitable? 
[paper link](https://openreview.net/pdf?id=r1lWUoA9FQ)
>This paper attemps to illustrate and characterize adversarial examples using mathematical models

#### selected key points
1. math setup: some data points lie in space $\Omega$. C partitions $\Omega$ into disjoint subsets. If there is $x,\hat{x}\in\Omega$ with $C(\hat{x})\neq c, C(x)=c$ and $d(x,\hat{x})\le\epsilon$. Then $\hat{x}$ is an adversarial example
2. This paper goes on to use above settings to derive upon simple models such as half of a sphere. 

#### branching points
1. The math setup is weird. It does not separate data points from the whole space, thus assuming that data points form a continuous space, rather than discrete points. In this condition, there will always be adversarial points due to denseness of real number.
2. A better setup: Let R be the full image space with n = w \*  h, $\Omega\_i$ is the set of discrete data points belonging to class i. C partitions $\rm I\!R^n$ into disjoing subspaces, each containing a $\Omega_i$. if any $x\in\Omega_i$ is closer to the boundary of C than $\epsilon$, there exist an adversarial example.


## Adversarial Examples: Attacks and Defenses for Deep Learning
[paper link](https://arxiv.org/pdf/1712.07107.pdf)
> This paper reviews the topic of adversarial examples.
> In addition to images, it also mentions language, malware detection etc.
> A concise review paper including: introduction, list of attacking methods, list of 6 defence methods (mostly failed already to later attacks), brief conclusion on future challenges.

#### selected key points
1. White box adversarial attack sample generated mostly by model gradients. Black box mostly by transfer attack (but also some with Evolution Algorithm)
2. Psychometric perceptual adversarial similarity score (PASS) is "Adversarial diversity and hard positive generation" introduced in to be consistent with human perception
3. attack review shows clear progress of how attacks evolved. defend not so clear. Some interesting points include:
    * FGSM one step attacks easy to implement but also easy to defend
    * universal perturbation adds onto various imgs to become adversarial example
    * Can also attack GAN, VAE to produce different result
    * printability can be considered when attempting for real world objects (ex glasses for adversarial on face detection)
    * Defence methods include reactive (detacting adversarial examples) and proactive (making models more robust)

#### branching points
1. Intriguing properties of neural networks is the first paper on adversarial attacks. Often refered by author as Szegedy et al. in this and other following papers
2. As of July 2018, most defenses target adversarial examples in computer vision, leaving other areas (Natural Language, Malware detection) in need of research efforts. However it seems that there is currently no robust defence method.
3. check out if PASS is resonable!
4. history attacks and defences to be sorted out!

## Explaining and Harnessing Adversarial Examples 
[paper link](https://arxiv.org/abs/1412.6572)
> Fast Gradient Sign Method; Adversarial training.
> Argues that linearity is the fundamental reason for adversarial examples.
> citing "Intriguing properties" a lot, possibly the second important paper for adversarial attack.

#### selected key points
1. previous papers argues non-linearity as cause. This paper base its hypothesis on linearity. Ex: $w^T\tilde{x}=w^Tx+w^T\eta$
The difference will still be large even if $\eta$ is the smallest decimal in each dimention if the dimention is large
2. Fast Gradient Sign Method:
$\eta=\epsilon sign(\nabla_xJ(\theta,x,y))$, J is the cost function
3. Author believes criticism for deep networks is misguided. Unlike shallow ones, deep networks are at leat able to represent functions that resists adversarial attacks. Following Szegedy et al. This paper also train on adversarial examples made with FGSM and reduce error rate from 94% to 84% (which is useless lol)
4. Training on noise added sample is useless
5. Under linear view, adversarial example form broad subspaces. This is clearly proven by adjusting $\epsilon$ along the direction $sign(\nabla_xJ(\theta,x,y))$
6. Generative model (discriminate real/fake(adversarial) data) is useless. Ensembling is also useless
7. Conclusion: linearity is key. Deep learning model is easy to train, and so is easy to attack.

#### branching points
1. nonlinear model families such as RBF networks can reduce vulnerability. What is RBF, why is it resistant to rubbish class examples!
2. Is it true that robust deep model is definitely possible? should be able to do $x+\eta\rightarrow x$ A deep learning model that modifies data as it classifies
3. If linearity is key, perhaps biologically neurons avoid this by being non-linear for single cell (full on/off) while linear for ensembles (how does ensemble and tuning curve work!!)
4. Perhaps each point works for itself and does not consider whether it forms a reasonable picture along side other examples (reflecting on rubbish class example as noise imgs generated from FGSM from random values blank)

## Intriguing properties of neural networks 
[paper link](https://arxiv.org/abs/1312.6199)
>The first paper on adversarial attack written by Szegedy et al.  
>2 intriguing properties: a. layer forms a space of semantic meanings (rather than each unit in layer hold meaning) b. local generalization assumption is broken by adversarial examples.

#### selected key points
1. (a layer in neural net forms a space of sementic meaning?) there is no distinction between individual high levl units and random linear combinations of high level unit. This suggest that it is the space rather than individual units that contains the semantic information in the high layers of neural networks
2. let $\phi(x)$ be activation values of some layer, $x$ are the some input img. expeiment show that for any random direction $v\in\rm I\!R^n$ (including $\hat{e}_i$) $x'=arg\max\limits_{x\in I}\langle\phi(x),v\rangle$  will select out examples semantically related to each other.
3. it is argued ("Learning deep architectures in ai") that the deep stack of non-linear layers between input output of neural net are a way for the model to encode a non-local generalization prior over the input-space. Which means it is possible for deep learning model to assign regions of input space without training example to correct class (ex same object with different viewpoint and are far in pixel space)
4. local generalization, meaning that all data close to training data point will be assigned to same class is assumed above. However the assumption is false as this paper found that deep neural nets learn fairly discontinuous input-output mappings such that small perturbation can cause many networks to error (there are adversarial examples and it is transferable to different networks)
5. To solve: Minimize $\Vert r\Vert_2$ s.t. $f(x+r)=l, x+r \in [0,1]^m$ approximate by box-constrained L-BFGS, i.e. by line-serach to find minimum c>0? Minimize $c|r|+loss_f(x+r,l)$ subject to $x+r \in [0,1]^m$
6. adding gaussian noise with same distance does not effect error as much as adversarial perturbation
7. adversarial example generizes across models trained with disjoing subset within same training set (MNIST)
8. Calculates upper bound of (possibility for adversarial example/instability of nerual net)? and suggests that 'simple regularization of the parameters, consisting in penalizing each Lipschitz bound, might help improve the generalisation  error of the networks'

#### branching points
1. what is L-BFGS?
2. Lipschitz bound? Spectral Analysis of Unstability does not seem to give a clear answer.

## Towards Evaluating the Robustness of Neural Networks 
[paper link](https://arxiv.org/abs/1608.04644)
> Explain how defensive distillation protects from previous attacks and show that it is not robust against proposed attacks algorithms (C & W attacks for $L_0, L_2, L_\infty$ distance metrics).  
> notation: full network func. is $F(x)=softmax(Z(x))=y$

#### selected key points
1. distillation: train second model with output of first model (it will be softer than one hot vector and contains info), reduces previous attack success rate from 95% to 0.5%
2. prior attack methods: L-BFGS, FGSM, JSMA(greedy algorithm to pick pixel to modify), DeepFool(approximate as linear iteratively to move data point close to boundary)
3. because constraint C()... is highly non-linear, define f such that $C(x+\delta)=t \iff f(x+\delta)\le0$. (all possible functions listed below)
4. The paper used constrants to ensure that modification yieds valid images (discrete pixel (0-255), change of variable$\delta_i=\frac{1}{2}(\tanh(w_i)+1)-x_i$)
 
#### formulations
Following Szegedy et al.'s formulation for adversarial examples:  
mimimize $D(x,x+\delta)$  
such that $C(x+\delta)=t, x+\delta\in [0,1]^n$  
(C is classifyer func; t is some class; [0,1] is range for img)  
Now we have:  
minimize $D(x,x+\delta)$   
such that $f(x+\delta\le 0), x+\delta\in[0,1]^n$  
alternatively:  
minimize $D(x,x+\delta)+c\cdot f(x+\delta)$  
such that $x+\delta\in[0,1]^n$  
c>0, empirically smallest c has best result  
finally $L_2$ attack is:  
minimize $\Vert\frac{1}{2}(tanh(w)+1)-x\Vert^2_2+c\cdot f(\frac{1}{2}(tanh(w)+1))$   


$f(x')=\max(\max Z_{i:i\neq t}(x')-Z_t(x'),-\kappa)$, $\kappa$ is for confidence level  
$L_0$ attack is to iteratively run $L_2$ attack and remove pixel i with lower 
$\nabla f_i(x+\delta)\cdot\delta_i$ value  
$L_\infty$ attack is by minimize $c\cdot f(x+\delta)+\sum\limits_i[(\delta_i-\tau)^+]$, with $\tau =\tau x 0.9$ after each successful iteration   
defensive distillation use $softmax_i(x,T)=\frac{e^{x_i/T}}{\sum_je^x_j/T}$  
than train teacher network at T=T, teacher generate soft label, train student at T=T with soft label, test student at T=1  
Thus T will meddle with gradients and previous attacks will fail  

#### list of notations and functions
* $L_p\equiv\Vert x-x'\Vert_p\equiv\bigg(\sum\limits_{i=1}^n|(x-x')_i|^p\bigg)^\frac{1}{p}$
* $L_0=$ how many pixel changed
* $L_2=$ Euclidean distance
* $L_\infty=$ maximum change of any coordinate
* $f_1(x')=-loss_{F,t}(x')+1$  
* $f_2(x')=(\max\limits_{i\neq t}(F(x')_i)-F(x')_t)^+$  
* $f_3(x')=softplus(\max\limits_{i\neq t}(F(x')_i)-F(x')_t)-\log(2)$  
* $f_4(x')=(0.5-F(x')_t)^+$  
* $f_5(x')=-\log(2F(x')_t-2)$  
* $f_6(x')=(\max\limits_{i\neq t}(Z(x')_i)-Z(x')_t)^+$  
* $f_7(x')=softplus(\max\limits_{i\neq t}(Z(x')_i)-Z(x')_t)^+$  
* $(\cdot)^+\equiv\max(e,0), softplus(x)=\log(1+\exp(x))$ , loss is cross-entropy  

#### branching points
1. should checkout L-BFGS JSMA DeepFool in future
2. this is a good example of formulating math to produce deep learning success!
3. not clear on how the last part on gradients yet!

## Towards Deep Learning Models Resistant to Adversarial Attacks
[paper link](https://arxiv.org/abs/1706.06083)
> Believes that size is all, if size is big enough adversarial proof networks is achievable


#### selected key points
1. propose that projected gradient descent(PGD) as a universal "first-order adversary", basically is $FGSM^k$: $x^{t+1}=\Pi_{x+S}(x^t+\alpha sgn(\nabla_xL(\theta,x,y))$
2. find model capacity plays important role (uses Resnet model for cifar10)
3. conjugate saddle point problem as central objective:
$\min\limits_\theta\rho(\theta)$, where $\rho(\theta)=\rm I\!E_{(x,y)\sim D}\big[\max\limits_{\delta\in S}L(\theta,x+\delta,y)\big]$
which is to minimize (adversarial loss)
4. experiment PGD with multiple random restarts find that adversarail loss tend plateaus around same value, (the fact that deep learning plateaus around same value for training, is believed to have multiple same value local minima)
5. need to solve saddle point problem + show that value is small

#### branching points
1. which paper beats this??
2. Danskin's theorm states that gradients at inner maximizers corresponds to descent directions for the saddle point problem
3. it seems wierd to say that data points of different class would be closer than $\varepsilon$ since the adversarial examples can mostly be undetected by human.
4. if cifar need resnet than imagenet...
