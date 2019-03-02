
# week2 paper reviews for ADF
scheduled date: Feb. 8 - Feb. 14    

1. contour context
2. nonlinearity
3. other topic continuing prev week papers
> paper list1:  
[Extending Adversarial Attacks and Defenses to Deep 3D Point Cloud Classifiers](#Extending-Adversarial-Attacks-and-Defenses-to-Deep-3D-Point-Cloud-Classifiers)
[Attention, Please! Adversarial Defense via Attention Rectification and Preservation](#Attention-Please-Adversarial-Defense-via-Attention-Rectification-and-Preservation)

> paper list2:  
[Sparse DNNs with Improved Adversarial Robustness](#Sparse-DNNs-with-Improved-Adversarial-Robustness)
[Neural Networks with Structural Resistance to Adversarial Attacks](#Neural-Networks-with-Structural-Resistance-to-Adversarial-Attacks)
[A New Family of Neural Networks Provably Resistant to
Adversarial Attacks](#A-New-Family-of-Neural-Networks-Provably-Resistant-to-Adversarial-Attacks)

> paper list3:  
> (quick check on PASS)  
> [Adversarial Diversity and Hard Positive Generation](#Adversarial-Diversity-and-Hard-Positive-Generation)  
>(check if Towards deep learning ... is deafeated as of 2018)  
>[Wild patterns: Ten years after the rise of adversarial machine learning](#Wild-patterns-Ten-years-after-the-rise-of-adversarial-machine-learning)  

## Extending Adversarial Attacks and Defenses to Deep 3D Point Cloud Classifiers   
[paper link](https://arxiv.org/abs/1901.03006)
>This paper investigates adversarail attack and defence on 3D Point Cloud Classifiers, and found it to be attackable, and also defendable to some degree
>(Jan. 2019)

#### selected key points
1. Point cloud classifiers classifies 'point cloud data' i.e. a bunch of points forming 3D representation. The only info is position, therefore adversarial samples moves points around and is more easily detectible due to outliers that change **overall shape**. Can be prevented by a. projecting to surface, and b. clip norm of neighboring distance.
2. Defend by a. adversarial training (Ian Goodfellow) b. Remove outlier (high avg dist. to neighbor); Remove salient point 
3. Attacks are successful (~90%), but defence can break attacks success rate in half
#### branching points
1. If deep learning is as good as human brain, than adversarial examples that does not change **overall shape** should have no effect, and those with few outliers should have minimum effect.
2. In (7. Discussion): max-pooling operations ... hide a subset of pts so thay have zero gradients and could not be perturbed ... and after outliers, salient points removed (defence procedures) those points are exposed and can represent overall shape of point cloud
3. Best defensive method is by removing outliers (hinting on **overall shape?**; data is not provided)

## Attention, Please! Adversarial Defense via Attention Rectification and Preservation
[paper link](https://arxiv.org/abs/1811.09831)
>This paper applies insights gained from attention analysis on deep learning models to build strong adversarial defence (that does not deviate attention much when subject to adversarial example) and also apply to new attack (that considers attention)
>(Nov. 2018)

#### selected key points
1. Prev. defences mostly within three group: (a.) Denoising, transforming adversarial sample before feeding into the model (b.) Model modification and (c.) Adversarial training
2. (c.) is applicable to various different algorithms. It has 3 factors: (i) focus on robust features [Tsipras et al.](https://arxiv.org/abs/1805.12152) found that adversarial attack amplify importance of low confidence features (ii) reduce feature distribution divergence (iii) more robust if train on stronger advdersarial examples
3. visual attention -> region of interest. Successful attack deviates and scatters the attention area
4. Attention-baseed Adversarial Defense (AAD): (i)rectification, correct the classifcation? (ii) perservation, align attention area btw adversarial and original imgs (iii) selection of strong adversarial samples based on attention
5. IoU(truth, attention)<0.5 would likely be attacked.  Samples on which attack fails have higher IoU
6. IoU(ori, adv) small for attacks that succeed.
7. iter attack can shrink and scatter IoU step by step, but after (2) iteration IoU continue to drop but success rate will not climb, and adversarial example will become noticeable (which is bad for attack)

##### 3 part AAD model
1. rectification
help model focus attention on groud truth area (unavailable) --> rectify completeness of attention area (not scattered) by: $$x^m = x\odot(1-Att(x))$$ $$Loss = - Distance(f(x), f(x^m))$$
2. perservation
reduce shifting of attention for adversarial example. $$Loss = Distance(g(x), g(x_{adv}))$$This will give total loss $$Loss = \alpha L_c(x) +\beta L_r(x) +\gamma L_p(x)$$

3. select strong adversarial sample (that deviates attention more)

##### result
1. against StepLL, R+StepLL, Iter-LL have improvements better than [Madry et al.](https://arxiv.org/abs/1706.06083) (last paper of week1! Towards..., was state of the art)

#### more key points
1. $\alpha,\beta,\gamma$ ratio is different in MNIST and cifar10 setting (8,4,1 is generally best)
2. applied to attack by adding term to encourage attention deviation
#### branching points
1. A production of Beijing, China
2. Attention area $Att(x)\equiv Grad-CAM(x) > \kappa$ 
Check out [Grad-Cam](https://arxiv.org/abs/1610.02391)! 
3. How to defeat this?
4. Attention based attack vs defence is not evaluated!!



---
## Sparse DNNs with Improved Adversarial Robustness
[paper link](https://papers.nips.cc/paper/7308-sparse-dnns-with-improved-adversarial-robustness.pdf)
>Pruning a model saves computation/memory costs. This paper suggests that it can also help defend adversarial attack and discusses inefficiency and unrobustness of DNN together.


#### selected key points
1. DNN (basically deep learning) has redundent feature representations, and could improve for mobile application "[Predicting parameters in deep learning](https://arxiv.org/abs/1306.0543)" On the other hand there is the adversarial issue.
2. sparsity of connection or sparsity of neuron activity


#### branching points
1. Is a sparse DNN equivalent to a non linear DNN as coined in this paper?
2. The conclution here seems to be coherent with the paper above that adversarial attack utilizes less confidence connections/predictions/areas
3. The rest of this paper (after 3.) is skipped

## Neural Networks with Structural Resistance to Adversarial Attacks
[paper link](https://arxiv.org/abs/1809.09262)
>This paper promotes RBFI to replace ReLu! Following GoodFellow et al. (2014) that states local linearity may be key, the author aims to create a highly non-linear deep leaning model.

#### selected key points
1. (linearity issue) $\sum_{i=1}^nx_iw_i\rightarrow\sum_{i=1}^n(x_i+sgn(w)\epsilon)w_i$ will have output perturbation of $\epsilon\sum_{i=1}^n\vert w_i\vert$ which will be huge for large n
2. (gaussian) radial basis function is $\phi(x)=e^{-(\varepsilon r)^2}$ RBFI follows this and adds (i) remove radial symmetry and allow RBFIs to scale each component (sensitivity) individually; (ii) calculate distance not in $l_2$ but in $l_\infty$, which is maximum of difference, so we have n=1. (I in RBFI is infinity)
3. pseudo gradient!
4. result: similar to relu/sigmoid for MNIST, superb for adversarial MNIST (90% vs 2%)!
5. history: FGSM idea based on linearity,
training on FGSM ad-examples isn't good enough (label leaking).
Iterative I-FGSM and ensemble method won NIPS 2017, also using small random perturbations before FSGM is good
carlini and wagner show FGSM rely on local gradients, and propose using more general optimization to find attack.
6. Finally, Madry et al. find PGD to be strongest attack, and training on such examples though computationally exspensive, provides good defence.

##### RBFI units explained:
to find small output variant under small input variant in infinity norm  
* a single unit in a layer of DNN can be written as $\sum_ix_iw_i$, which can be interperted as distance to a plane perpendicular to to vector w, scaled by $\vert w \vert$ 
* For $\infty$ norm, pt to pt distance is more reasonable. So the author constructs $$N_\gamma(u,w)(x)=exp(-\Vert u \odot (x-w)\Vert^2_\gamma), \gamma=\infty$$This becomes $$N_\infty(u,w)(x)=exp\big(-\max\limits_{1\le i\le n}(u_i(x_i-w_i))^2\big)$$ which works like an AND gate. Created similarly is the OR gate/unit:$$N_\infty(u,w)(x)=1-exp\big(-\max\limits_{1\le i\le n}(u_i(x_i-w_i))^2\big)$$And layer, Or layer or Mixed layer can be constructed with these units
* Sensitivity  
Normal relu/sigmoid \(x 1/4) unit have $pert.=\epsilon\times\sum\limits_{i=1}^n\Vert w\Vert_1$
For RBFI unit we have $\epsilon\times$ ... unclear!!! but perhaps change addition to multiplication

(aborted because next paper has same essence but better wording)



#### branching points
1. reproduce this!!!https://github.com/lucadealfaro/rbfi
2. Is this because previous adversarial attack utilize normal gradients. Did this paper utilize 'pseudo gradient' for adversarial as well?


## A New Family of Neural Networks Provably Resistant to Adversarial Attacks
(using new template for paper review starting here)
* **Authors:** Rakshit Agrawal, Luca de Alfaro, David Helmbold
* [**paper link**](https://arxiv.org/abs/1902.01208)
* **publish date:** Feb 2019
#### summary
> This work uses a new unit with $l_\infty$ distance for forward pass and similar differentiable function for backward pass (pseudogradient descent) to create a non linear model that resists adversarial attack for $\epsilon<0.5$
##### chapter-wise note
(1. Introduction)
* Goodfellow et al. suggests (local) linearity is key. Consider $\sum^n_ix_iw_i$ can have $n\epsilon\vert w\vert$ output change which snowballs through layers
* MWD unit activate as $$\mathcal{U}(u,w)(x)=\exp\big(-\max\limits_{1\le i\le n}(u_i(x_i-w_i))^2\big)$$ which is for x $l_\infty$ distance from w multiply non-negative weight $u_i$ for coordinate i
* Found pseudogradient as proxy for training efficiently
(2. Related Work)   
* Goodfellow et al. mentioned RadialBasisFunction
* Condsiders FGSM, I-FGSM, and PGD (Madry et al.)
(3. MWD Networks)
* MWD units output ~ 1 only when x close to w => AND gate
* $1-\mathcal{U}$ as NAND gate, use AND layer; NAND layer or mixed
* gradient$\Rightarrow$pseudogradients: $$\frac{d}{dz}e^{-z}=-e^{-z}\Rightarrow -\frac{1}{\sqrt{1+z}}$$$$\frac{d}{dz}\max z_i=\begin{cases}1 & z_i=y\\0 & z_i<y\end{cases}\Rightarrow e^{z_i-y}=\begin{cases}1&z_i=y\\e^{-\delta}&z_i<y\end{cases}$$
(7. Results)
* can use gradient, but will be very slow, experiment show training with gradient is not significantly better.
* MWD Networks significantly higher than Relu trained with adversarial example for $0<\epsilon<0.5$
#### branching points and thoughts
1. is it possible to go from various kind of pseudogradient (this work) to using reinforcement, or a somewhat local incentives or different kind of propagation of error.
2. what about all/non as in bio neurons? 

---

## Adversarial Diversity and Hard Positive Generation
* **Authors:** Andras Rozsa, Ethan M. Rudd, Terrance E. Boult
* [**paper link**](https://arxiv.org/abs/1605.01775)
* **date:** May 2016
#### summary
> This work promotes psycometric perceptual adversarial similarity score (PASS) as a measure of how close an adversarial example is to its original counterpart, as considered by 'human'  
(3. PASS)  
* human can correctly classify --> should consider noticeble differece, excluding small perturbation, small rotation/translation
* measurement: 1. align, by enhanced correlation coefficient  2. measure difference, by structural difference, regional structural similarity index $RSSIM(x,y)=L(x,y)^\alpha C(x,y)^\beta S(x,y)^\gamma$ corresponding to luminance, contrast, and structure. (can be calculated by mean, variance and covariance of that spot)

#### branching points
* This doesn't seem useful


## Wild patterns: Ten years after the rise of adversarial machine learning
* **Authors:** Battista Biggio, Fabio Roli
* [**paper link**](https://www.sciencedirect.com/science/article/pii/S0031320318302565)
* **date:** Jul 2018
#### summary
> This work summarize 10 years of development on the topic of adversarial attack. Regarding the work of Madry et al., the current state of the art defence has not been defeated yet.

* Robust optimization formulates adversarial learning as a minimax problem in which the inner problem maximizes the training loss by manipulating the traning points under worst-case, bounded pertutrbations, while the outer problem trains the learning algorithm to minimize the corresponding worst-case training loss. 


#### branching point
* would be useful to go through other parts of this review, has a good summarizing plot.
