
# week1 paper reviews for ADF
scheduled date: Feb. 1 - Feb. 7    

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
1. math setup: some data points lie in space <img src="/doc/week01/tex/9432d83304c1eb0dcb05f092d30a767f.svg?invert_in_darkmode&sanitize=true" align=middle width=11.87217899999999pt height=22.465723500000017pt/>. C partitions <img src="/doc/week01/tex/9432d83304c1eb0dcb05f092d30a767f.svg?invert_in_darkmode&sanitize=true" align=middle width=11.87217899999999pt height=22.465723500000017pt/> into disjoint subsets. If there is <img src="/doc/week01/tex/e20fb9e8fe38c394ec5147172b66858f.svg?invert_in_darkmode&sanitize=true" align=middle width=58.05915389999999pt height=22.831056599999986pt/> with <img src="/doc/week01/tex/f4f80403985b065178a5b49565ade7a4.svg?invert_in_darkmode&sanitize=true" align=middle width=135.57884835pt height=24.65753399999998pt/> and <img src="/doc/week01/tex/eb0705ec2688793a826a8d4f1bc3adda.svg?invert_in_darkmode&sanitize=true" align=middle width=76.02725625pt height=24.65753399999998pt/>. Then <img src="/doc/week01/tex/f84e86b97e20e45cc17d297dc794b3e8.svg?invert_in_darkmode&sanitize=true" align=middle width=9.39498779999999pt height=22.831056599999986pt/> is an adversarial example
2. This paper goes on to use above settings to derive upon simple models such as half of a sphere. 

#### branching points
1. The math setup is weird. It does not separate data points from the whole space, thus assuming that data points form a continuous space, rather than discrete points. In this condition, there will always be adversarial points due to denseness of real number.
2. A better setup: Let R be the full image space with n = w*h, <img src="/doc/week01/tex/8fea593a0c14ea6d2a4d7d5bad42b661.svg?invert_in_darkmode&sanitize=true" align=middle width=16.52307854999999pt height=22.465723500000017pt/> is the set of discrete data points belonging to class i. C partitions <img src="/doc/week01/tex/c92edb5070aa26dcc6479898701e3881.svg?invert_in_darkmode&sanitize=true" align=middle width=22.55721599999999pt height=22.465723500000017pt/> into disjoing subspaces, each containing a <img src="/doc/week01/tex/8fea593a0c14ea6d2a4d7d5bad42b661.svg?invert_in_darkmode&sanitize=true" align=middle width=16.52307854999999pt height=22.465723500000017pt/>. if any <img src="/doc/week01/tex/7347bc852243ba765700bf6517dbb79a.svg?invert_in_darkmode&sanitize=true" align=middle width=46.00920389999999pt height=22.465723500000017pt/> is closer to the boundary of C than <img src="/doc/week01/tex/7ccca27b5ccc533a2dd72dc6fa28ed84.svg?invert_in_darkmode&sanitize=true" align=middle width=6.672392099999992pt height=14.15524440000002pt/>, there exist an adversarial example.


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
1. previous papers argues non-linearity as cause. This paper base its hypothesis on linearity. Ex: <img src="/doc/week01/tex/fa9db3f8b9652d5618ff3e11f982d6d3.svg?invert_in_darkmode&sanitize=true" align=middle width=137.25004589999998pt height=27.6567522pt/>
The difference will still be large even if <img src="/doc/week01/tex/1d0496971a2775f4887d1df25cea4f7e.svg?invert_in_darkmode&sanitize=true" align=middle width=8.751954749999989pt height=14.15524440000002pt/> is the smallest decimal in each dimention if the dimention is large
2. Fast Gradient Sign Method:
<img src="/doc/week01/tex/7ada38b3a0784c5a6f9bfbd9714bdd4b.svg?invert_in_darkmode&sanitize=true" align=middle width=168.07954185pt height=24.65753399999998pt/>, J is the cost function
3. Author believes criticism for deep networks is misguided. Unlike shallow ones, deep networks are at leat able to represent functions that resists adversarial attacks. Following Szegedy et al. This paper also train on adversarial examples made with FGSM and reduce error rate from 94% to 84% (which is useless lol)
4. Training on noise added sample is useless
5. Under linear view, adversarial example form broad subspaces. This is clearly proven by adjusting <img src="/doc/week01/tex/7ccca27b5ccc533a2dd72dc6fa28ed84.svg?invert_in_darkmode&sanitize=true" align=middle width=6.672392099999992pt height=14.15524440000002pt/> along the direction <img src="/doc/week01/tex/7c58fbe3788e939d4e33e1611e56bc65.svg?invert_in_darkmode&sanitize=true" align=middle width=130.73758335pt height=24.65753399999998pt/>
6. Generative model (discriminate real/fake(adversarial) data) is useless. Ensembling is also useless
7. Conclusion: linearity is key. Deep learning model is easy to train, and so is easy to attack.

#### branching points
1. nonlinear model families such as RBF networks can reduce vulnerability. What is RBF, why is it resistant to rubbish class examples!
2. Is it true that robust deep model is definitely possible? should be able to do <img src="/doc/week01/tex/46386043b67d3c435f66d319d6c8c37a.svg?invert_in_darkmode&sanitize=true" align=middle width=73.20370364999998pt height=19.1781018pt/> A deep learning model that modifies data as it classifies
3. If linearity is key, perhaps biologically neurons avoid this by being non-linear for single cell (full on/off) while linear for ensembles (how does ensemble and tuning curve work!!)
4. Perhaps each point works for itself and does not consider whether it forms a reasonable picture along side other examples (reflecting on rubbish class example as noise imgs generated from FGSM from random values blank)

## Intriguing properties of neural networks 
[paper link](https://arxiv.org/abs/1312.6199)
>The first paper on adversarial attack written by Szegedy et al.  
>2 intriguing properties: a. layer forms a space of semantic meanings (rather than each unit in layer hold meaning) b. local generalization assumption is broken by adversarial examples.

#### selected key points
1. (a layer in neural net forms a space of sementic meaning?) there is no distinction between individual high levl units and random linear combinations of high level unit. This suggest that it is the space rather than individual units that contains the semantic information in the high layers of neural networks
2. let <img src="/doc/week01/tex/1dd66ca1cb582bf5f23f25067f3537c2.svg?invert_in_darkmode&sanitize=true" align=middle width=31.974965549999986pt height=24.65753399999998pt/> be activation values of some layer, <img src="/doc/week01/tex/332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode&sanitize=true" align=middle width=9.39498779999999pt height=14.15524440000002pt/> are the some input img. expeiment show that for any random direction <img src="/doc/week01/tex/32202cea73dfe26c6e71d5983f049483.svg?invert_in_darkmode&sanitize=true" align=middle width=51.20619569999999pt height=22.465723500000017pt/> (including <img src="/doc/week01/tex/512eb3865853f108dbe121991aab0b00.svg?invert_in_darkmode&sanitize=true" align=middle width=12.30503669999999pt height=22.831056599999986pt/>) <img src="/doc/week01/tex/2a4b403f4f51aa9eaaca17a67b302385.svg?invert_in_darkmode&sanitize=true" align=middle width=154.87445655pt height=24.7161288pt/>  will select out examples semantically related to each other.
3. it is argued ("Learning deep architectures in ai") that the deep stack of non-linear layers between input output of neural net are a way for the model to encode a non-local generalization prior over the input-space. Which means it is possible for deep learning model to assign regions of input space without training example to correct class (ex same object with different viewpoint and are far in pixel space)
4. local generalization, meaning that all data close to training data point will be assigned to same class is assumed above. However the assumption is false as this paper found that deep neural nets learn fairly discontinuous input-output mappings such that small perturbation can cause many networks to error (there are adversarial examples and it is transferable to different networks)
5. To solve: Minimize <img src="/doc/week01/tex/6bdcb9217783085e13142cedce026621.svg?invert_in_darkmode&sanitize=true" align=middle width=30.86392154999999pt height=24.65753399999998pt/> s.t. <img src="/doc/week01/tex/9fd483a72f24002bfe1c95644c2bbe92.svg?invert_in_darkmode&sanitize=true" align=middle width=196.40571059999996pt height=24.65753399999998pt/> approximate by box-constrained L-BFGS, i.e. by line-serach to find minimum c>0? Minimize <img src="/doc/week01/tex/eca2444947c7873ff3a5274708936010.svg?invert_in_darkmode&sanitize=true" align=middle width=143.10508244999997pt height=24.65753399999998pt/> subject to <img src="/doc/week01/tex/15cc94e7a887c9ed06460ad1818a9cca.svg?invert_in_darkmode&sanitize=true" align=middle width=101.99187239999999pt height=24.65753399999998pt/>
6. adding gaussian noise with same distance does not effect error as much as adversarial perturbation
7. adversarial example generizes across models trained with disjoing subset within same training set (MNIST)
8. Calculates upper bound of (possibility for adversarial example/instability of nerual net)? and suggests that 'simple regularization of the parameters, consisting in penalizing each Lipschitz bound, might help improve the generalisation  error of the networks'

#### branching points
1. what is L-BFGS?
2. Lipschitz bound? Spectral Analysis of Unstability does not seem to give a clear answer.

## Towards Evaluating the Robustness of Neural Networks 
[paper link](https://arxiv.org/abs/1608.04644)
> Explain how defensive distillation protects from previous attacks and show that it is not robust against proposed attacks algorithms (C & W attacks for <img src="/doc/week01/tex/520cec487bc94f89c1e322ceb516a49d.svg?invert_in_darkmode&sanitize=true" align=middle width=76.02749549999999pt height=22.465723500000017pt/> distance metrics).
> notation: full network func. is <img src="/doc/week01/tex/cfccc8a9d574b13be59f70c02d6cf47d.svg?invert_in_darkmode&sanitize=true" align=middle width=198.82623585pt height=24.65753399999998pt/>

#### selected key points
1. distillation: train second model with output of first model (it will be softer than one hot vector and contains info), reduces previous attack success rate from 95% to 0.5%
3. prior attack methods: L-BFGS, FGSM, JSMA(greedy algorithm to pick pixel to modify), DeepFool(approximate as linear iteratively to move data point close to boundary)
4. Following Szegedy et al.'s formulation for adversarial examples:
mimimize <img src="/doc/week01/tex/e5c226c2b46c02c0dc2bad3fe69cbbba.svg?invert_in_darkmode&sanitize=true" align=middle width=80.96678699999998pt height=24.65753399999998pt/>
such that <img src="/doc/week01/tex/439ca16cf78f7660aa1a7e63ff6d4d9e.svg?invert_in_darkmode&sanitize=true" align=middle width=196.79209439999997pt height=24.65753399999998pt/>
(C is classifyer func. t is some class [0,1] is range for img)
5. because constraint C()... is highly non-linear, define f such that <img src="/doc/week01/tex/0e96665aed58340b4d8316bd4e87b827.svg?invert_in_darkmode&sanitize=true" align=middle width=229.53352784999996pt height=24.65753399999998pt/>  
list of possible fs:  
6. Now we have:
minimize <img src="/doc/week01/tex/e5c226c2b46c02c0dc2bad3fe69cbbba.svg?invert_in_darkmode&sanitize=true" align=middle width=80.96678699999998pt height=24.65753399999998pt/>  
such that <img src="/doc/week01/tex/f7cf9bad9afe90a00bf212449465b80a.svg?invert_in_darkmode&sanitize=true" align=middle width=195.96798209999997pt height=24.65753399999998pt/>  
alternatively:  
minimize <img src="/doc/week01/tex/4104b755748abdcf4f3311c7afeb45ec.svg?invert_in_darkmode&sanitize=true" align=middle width=180.06086009999999pt height=24.65753399999998pt/>  
such that <img src="/doc/week01/tex/f285dcd8e7161b763bfbb2171dface08.svg?invert_in_darkmode&sanitize=true" align=middle width=98.50816139999999pt height=24.65753399999998pt/>  
c>0, empirically smallest c has best result  
7. The paper used constrants to ensure that modification yieds valid images (discrete pixel (0-255), change of variable<img src="/doc/week01/tex/955efaa45bc69d8793f5934a85d784ef.svg?invert_in_darkmode&sanitize=true" align=middle width=183.33056774999997pt height=27.77565449999998pt/>)
8. final <img src="/doc/week01/tex/4327ea69d9c5edcc8ddaf24f1d5b47e4.svg?invert_in_darkmode&sanitize=true" align=middle width=17.73978854999999pt height=22.465723500000017pt/> attack is:  
minimize <img src="/doc/week01/tex/0a17dc13a821746521ea193125ed35d8.svg?invert_in_darkmode&sanitize=true" align=middle width=336.08506800000004pt height=27.77565449999998pt/>  
<img src="/doc/week01/tex/8ddfab6febb07b5bffe0d1ebd4039120.svg?invert_in_darkmode&sanitize=true" align=middle width=335.4472308pt height=24.7161288pt/>  
<img src="/doc/week01/tex/5c62da39aa7289df62d937cb24a31161.svg?invert_in_darkmode&sanitize=true" align=middle width=9.47111549999999pt height=14.15524440000002pt/> is for confidence level  
<img src="/doc/week01/tex/cc96eb8a40f81e8514147d06c9e8ad92.svg?invert_in_darkmode&sanitize=true" align=middle width=17.73978854999999pt height=22.465723500000017pt/> attack is iteratively run <img src="/doc/week01/tex/4327ea69d9c5edcc8ddaf24f1d5b47e4.svg?invert_in_darkmode&sanitize=true" align=middle width=17.73978854999999pt height=22.465723500000017pt/> attack and remove pixel i with lower <img src="/doc/week01/tex/e60f762ec5757d0a022eb1845f4ee06b.svg?invert_in_darkmode&sanitize=true" align=middle width=103.01738699999997pt height=24.65753399999998pt/> value  
<img src="/doc/week01/tex/3cc58aeae18bc28014beb059bf644895.svg?invert_in_darkmode&sanitize=true" align=middle width=24.292324649999987pt height=22.465723500000017pt/> attack is by  
minimize <img src="/doc/week01/tex/47f896721d19c058e2365caa764d302e.svg?invert_in_darkmode&sanitize=true" align=middle width=191.19365429999996pt height=28.92705090000002pt/>, with <img src="/doc/week01/tex/feea067ecd7d1c19447f32aec232adf7.svg?invert_in_darkmode&sanitize=true" align=middle width=60.18831719999999pt height=21.18721440000001pt/> after each successful iteration   
9. defensive distillation use  
<p align="center"><img src="/doc/week01/tex/e2c6d52723e45fd6f7d466fc546146f4.svg?invert_in_darkmode&sanitize=true" align=middle width=197.73956565pt height=44.16107355pt/></p>  
than train teacher network at T=T, teacher generate soft label, train student at T=T with soft label, test student at T=1  
Thus T will meddle with gradients and previous attacks will fail  
 

#### details notations and functions
* <img src="/doc/week01/tex/d1d5ef430a55b92632050eab0b4a45fd.svg?invert_in_darkmode&sanitize=true" align=middle width=267.66034844999996pt height=58.97244539999998pt/>
* <img src="/doc/week01/tex/03e9cccda02af9b900ae6e17fb149d1f.svg?invert_in_darkmode&sanitize=true" align=middle width=35.91323504999999pt height=22.465723500000017pt/> how many pixel changed
* <img src="/doc/week01/tex/9ed5482ad04488c19a8da277a691b870.svg?invert_in_darkmode&sanitize=true" align=middle width=35.91323504999999pt height=22.465723500000017pt/> Euclidean distance
* <img src="/doc/week01/tex/3c3c2df637165e6d9fe8c32da59a1b75.svg?invert_in_darkmode&sanitize=true" align=middle width=42.465751349999984pt height=22.465723500000017pt/> maximum change of any coordinate
* <img src="/doc/week01/tex/f8fcdb844475abcb6e81c735939ea17e.svg?invert_in_darkmode&sanitize=true" align=middle width=179.01034304999996pt height=24.7161288pt/>  
* <img src="/doc/week01/tex/76927801a0a8c4f158f1e7b7f869ce11.svg?invert_in_darkmode&sanitize=true" align=middle width=241.03244054999996pt height=26.17730939999998pt/>  
* <img src="/doc/week01/tex/f3e6fdd09258634dc345c11bb580818f.svg?invert_in_darkmode&sanitize=true" align=middle width=355.31151479999994pt height=25.936003499999995pt/>  
* <img src="/doc/week01/tex/6bb262db48aa512a6ddf0a5e7de5d3cf.svg?invert_in_darkmode&sanitize=true" align=middle width=173.53892159999998pt height=26.17730939999998pt/>  
* <img src="/doc/week01/tex/af86abad4c90a1640085cd69bdf48d6f.svg?invert_in_darkmode&sanitize=true" align=middle width=195.63932684999997pt height=24.7161288pt/>  
* <img src="/doc/week01/tex/d65636ecf70cccd58ec6d78a9b8bb1ee.svg?invert_in_darkmode&sanitize=true" align=middle width=240.11913255pt height=26.17730939999998pt/>  
* <img src="/doc/week01/tex/b6e12a45051e883a56085767d6fa7ffa.svg?invert_in_darkmode&sanitize=true" align=middle width=302.16083699999996pt height=26.17730939999998pt/>  
* <img src="/doc/week01/tex/6f1a6392b78010cb337131966ba9ec5e.svg?invert_in_darkmode&sanitize=true" align=middle width=339.809976pt height=26.17730939999998pt/> , loss is cross-entropy  

#### branching points
1. should checkout L-BFGS JSMA DeepFool in future
2. this is a good example of formulating math to produce deep learning success!
3. not clear on how the last part on gradients yet!

## Towards Deep Learning Models Resistant to Adversarial Attacks
[paper link](https://arxiv.org/abs/1706.06083)
> Believes that size is all, if size is big enough adversarial proof networks is achievable


#### selected key points
1. propose that projected gradient descent(PGD) as a universal "first-order adversary", basically is <img src="/doc/week01/tex/4cdaef488ac882fb61498ee7c824b0ff.svg?invert_in_darkmode&sanitize=true" align=middle width=61.81169444999998pt height=27.91243950000002pt/>: <img src="/doc/week01/tex/2620f873c064b80f3e19f8ac59c34c43.svg?invert_in_darkmode&sanitize=true" align=middle width=270.94992870000004pt height=26.76175259999998pt/>
2. find model capacity plays important role (uses Resnet model for cifar10)
3. conjugate saddle point problem as central objective:
<img src="/doc/week01/tex/bc1815d0f95aac938944c854140a4c3b.svg?invert_in_darkmode&sanitize=true" align=middle width=59.59484849999999pt height=24.65753399999998pt/>, where <img src="/doc/week01/tex/20b89846e10758be24386b65981271fd.svg?invert_in_darkmode&sanitize=true" align=middle width=254.81753384999996pt height=27.94539330000001pt/>
which is to minimize (adversarial loss)
4. experiment PGD with multiple random restarts find that adversarail loss tend plateaus around same value, (the fact that deep learning plateaus around same value for training, is believed to have multiple same value local minima)
5. need to solve saddle point problem + show that value is small

#### branching points
1. which paper beats this??
2. Danskin's theorm states that gradients at inner maximizers corresponds to descent directions for the saddle point problem
3. it seems wierd to say that data points of different class would be closer than <img src="/doc/week01/tex/9ae7733dac2b7b4470696ed36239b676.svg?invert_in_darkmode&sanitize=true" align=middle width=7.66550399999999pt height=14.15524440000002pt/> since the adversarial examples can mostly be undetected by human.
4. if cifar need resnet than imagenet...


